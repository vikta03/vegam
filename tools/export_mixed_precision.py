"""Export the trained NextChess model with explicit per-operation precision.

Same equations and the same FP16-rounded EMA weights as the deployed export;
only the arithmetic type of each operation is chosen here, so a strongly typed
TensorRT build follows it exactly. Large matrix products stay FP16. Optional FP32
islands:
  residual  residual stream and every LayerNorm (FP32 params = FP16 values)
  logits    attention scores plus static/adaptive bias summed in FP32
  gelu      feed-forward GELU evaluated in FP32
  heads     final LayerNorm output, pooling and all five heads in FP32
  foldbias  adaptive bias folded into the score product (exact algebraic rewrite):
            [q/sqrt(d), left*g] . [k, right] replaces q.k/sqrt(d) + (left*g).right
"""
import argparse
import json
import math
from pathlib import Path
import sys

import onnx
import torch
from torch import nn
from torch.nn import functional as F

N = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(N))
from nextchess.model import Config, Net


def h(t):
    return t.detach().half()


def f(t):
    return t.detach().half().float()  # FP16-rounded value held in FP32


class Mixed(nn.Module):
    def __init__(self, net, islands, move_capacity=256, fp32_blocks=None):
        super().__init__()
        if move_capacity not in (32, 64, 128, 256):
            raise ValueError('move capacity must be 32, 64, 128 or 256')
        self.move_capacity = move_capacity
        c = net.config
        self.c, self.islands = c, islands
        self.fp32_blocks = set(range(c.blocks) if fp32_blocks is None else fp32_blocks) if 'residual' in islands else set()
        if not self.fp32_blocks.issubset(range(c.blocks)):
            raise ValueError('FP32 block index out of range')
        self.depth = c.width // c.heads
        R = 'residual' in islands
        rt = (lambda t: f(t)) if R else (lambda t: h(t))
        self.input_w, self.input_b = nn.Parameter(h(net.input.weight)), nn.Parameter(h(net.input.bias))
        self.position = nn.Parameter(f(net.position) if 0 in self.fp32_blocks else h(net.position))
        self.blocks = []
        for i, b in enumerate(net.blocks):
            nt = f if i in self.fp32_blocks else h
            p = dict(
                ln1_w=nt(b.norm1.weight), ln1_b=nt(b.norm1.bias), ln2_w=nt(b.norm2.weight), ln2_b=nt(b.norm2.bias),
                qkv=h(b.qkv.weight), out=h(b.out.weight), bias=h(b.bias),
                ff1_w=h(b.ff[0].weight), ff1_b=h(b.ff[0].bias), ff2_w=h(b.ff[2].weight), ff2_b=h(b.ff[2].bias),
                comp_w=h(b.compress.weight), comp_b=h(b.compress.bias),
                g1_w=h(b.gates[0].weight), g1_b=h(b.gates[0].bias), g2_w=h(b.gates[2].weight), g2_b=h(b.gates[2].bias),
                left=h(b.left), right=h(b.right))
            if 'qkvcompress' in islands:
                # The QKV and adaptive compression projections read the same
                # normalized squares. Pack their unchanged learned rows into
                # one matrix product; zero rows align its output to 16.
                p['qkv_comp_w'] = torch.cat([h(b.qkv.weight), h(b.compress.weight),
                                            torch.zeros(8, c.width, dtype=torch.float16)])
                p['qkv_comp_b'] = torch.cat([torch.zeros(3*c.width, dtype=torch.float16),
                                            h(b.compress.bias), torch.zeros(8, dtype=torch.float16)])
            for k, v in p.items():
                self.register_parameter(f'b{i}_{k}', nn.Parameter(v))
            self.blocks.append({k: f'b{i}_{k}' for k in p})
        H = 'heads' in islands
        ht = (lambda t: f(t)) if H else (lambda t: h(t))
        for name, t in dict(norm_w=net.norm.weight, norm_b=net.norm.bias,
                            gp_w=net.global_proj[0].weight, gp_b=net.global_proj[0].bias,
                            state_w=net.state.weight, state_b=net.state.bias,
                            outcome_w=net.outcome.weight, outcome_b=net.outcome.bias,
                            duration_w=net.duration.weight, duration_b=net.duration.bias,
                            src_w=net.source.weight, src_b=net.source.bias,
                            dst_w=net.destination.weight, dst_b=net.destination.bias,
                            ctx_w=net.context.weight, ctx_b=net.context.bias,
                            promo=net.promotion.weight, castle=net.castle.weight,
                            j1_w=net.joint[0].weight, j1_b=net.joint[0].bias,
                            j2_w=net.joint[2].weight, j2_b=net.joint[2].bias).items():
            self.register_parameter(name, nn.Parameter(ht(t)))
        if not H and R:  # final LayerNorm reads the FP32 stream
            self.norm_w.data, self.norm_b.data = f(net.norm.weight), f(net.norm.bias)

    def P(self, block, key):
        return getattr(self, self.blocks[block][key])

    def block(self, i, x):
        c, I, P = self.c, self.islands, lambda k: self.P(i, k)
        x = x.to(torch.float32 if i in self.fp32_blocks else torch.float16)
        b = x.shape[0]
        z = F.layer_norm(x, (c.width,), P('ln1_w'), P('ln1_b'), 1e-5).half()
        if 'qkvcompress' in I:
            packed = F.linear(z, P('qkv_comp_w'), P('qkv_comp_b'))
            qkv = packed[:, :, :3*c.width]
            compressed = packed[:, :, 3*c.width:3*c.width+8]
        else:
            qkv = F.linear(z, P('qkv'))
            compressed = F.linear(z, P('comp_w'), P('comp_b'))
        q, k, v = qkv.reshape(b, 64, 3, c.heads, self.depth).permute(2, 0, 3, 1, 4).unbind(0)
        g = torch.tanh(F.linear(F.gelu(F.linear(compressed.flatten(1),
                                                 P('g1_w'), P('g1_b'))), P('g2_w'), P('g2_b')))
        g = g.reshape(b, c.heads, 1, c.rank)
        if 'foldbias' in I:
            qa = torch.cat([q / math.sqrt(self.depth), P('left').unsqueeze(0) * g], -1)
            ka = torch.cat([k, P('right').unsqueeze(0).expand(b, -1, -1, -1)], -1)
            scores = torch.matmul(qa, ka.transpose(-1, -2))
            scores = scores.float() + P('bias').float() if 'logits' in I else scores + P('bias')
        else:
            adaptive = torch.matmul(P('left').unsqueeze(0) * g, P('right').transpose(-1, -2).unsqueeze(0))
            qk = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.depth)
            if 'logits' in I:
                scores = qk.float() + (P('bias').float() + adaptive.float())
            else:
                scores = qk + (P('bias').unsqueeze(0) + adaptive)
        attn = torch.softmax(scores.float(), -1).half()
        attended = torch.matmul(attn, v).transpose(1, 2).reshape(b, 64, -1)
        a = F.linear(attended, P('out'))
        x = x + (a.float() if x.dtype == torch.float32 else a)
        z = F.layer_norm(x, (c.width,), P('ln2_w'), P('ln2_b'), 1e-5).half()
        u = F.linear(z, P('ff1_w'), P('ff1_b'))
        u = F.gelu(u.float()).half() if 'gelu' in I else F.gelu(u)
        u = F.linear(u, P('ff2_w'), P('ff2_b'))
        return x + (u.float() if x.dtype == torch.float32 else u)

    def forward(self, features, moves, legal):
        I = self.islands
        # Optional specialized decoder. The native caller must route positions
        # with more legal moves to the complete engine. Board computation and
        # every real move retain the same learned equations and parameters.
        if self.move_capacity < 256:
            moves = moves[:, :self.move_capacity]
            legal = legal[:, :self.move_capacity]
        x = F.linear(features.half(), self.input_w, self.input_b)
        x = x.float() + self.position if 0 in self.fp32_blocks else x + self.position
        for i in range(len(self.blocks)):
            x = self.block(i, x)
        dt = torch.float32 if 'heads' in I else torch.float16
        x = F.layer_norm(x.to(self.norm_w.dtype), (self.c.width,), self.norm_w, self.norm_b, 1e-5).to(dt)
        context = F.gelu(F.linear(x.mean(1), self.gp_w, self.gp_b))
        src = torch.gather(F.linear(x, self.src_w, self.src_b), 1, moves[:, :, 0, None].expand(-1, -1, 64))
        dst = torch.gather(F.linear(x, self.dst_w, self.dst_b), 1, moves[:, :, 1, None].expand(-1, -1, 64))
        gc = F.linear(context, self.ctx_w, self.ctx_b)[:, None, :]
        u = F.gelu(src + dst + src * dst / 8 + gc + F.embedding(moves[:, :, 2], self.promo)
                   + F.embedding(moves[:, :, 3], self.castle))
        joint = F.linear(F.gelu(F.linear(u, self.j1_w, self.j1_b)), self.j2_w, self.j2_b)
        logits = joint[:, :, 0].float().masked_fill(~legal, -1e9)
        actions = joint[:, :, 1:].float()
        if self.move_capacity < 256:
            logits = F.pad(logits, (0, 256-self.move_capacity), value=-1e9)
            actions = F.pad(actions, (0, 0, 0, 256-self.move_capacity))
        return (logits, F.linear(context, self.state_w, self.state_b).float(), actions,
                F.linear(context, self.duration_w, self.duration_b).float(),
                F.linear(context, self.outcome_w, self.outcome_b).float())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path,
                   default=N / 'runs/epochs23-20260919-pro5000-continue/downloads/final/latest.pt')
    p.add_argument('--metadata-from', type=Path, default=N / 'packages/wide-epoch3-20260919/model.onnx')
    p.add_argument('--islands', default='residual,logits,gelu,heads')
    p.add_argument('--move-capacity', type=int, choices=(32, 64, 128, 256), default=256,
                   help='specialized legal-move capacity; requires guarded native fallback')
    p.add_argument('--fp32-blocks', help='experimental residual/LN FP32 block indices (comma-separated); default all')
    p.add_argument('out', type=Path)
    a = p.parse_args()
    islands = set(filter(None, a.islands.split(',')))
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    net = Net(Config(**ck['config'])); net.load_state_dict(ck['ema']); net.eval()
    fp32_blocks = None if a.fp32_blocks is None else set(int(v) for v in a.fp32_blocks.split(',') if v)
    model = Mixed(net, islands, a.move_capacity, fp32_blocks).eval()
    x = torch.zeros(1, 64, 120); moves = torch.zeros(1, 256, 4, dtype=torch.int64)
    legal = torch.ones(1, 256, dtype=torch.bool)
    names = ['policy_logits', 'wdl_logits', 'action_wdl_logits', 'duration_params', 'outcome_logits']
    tmp = a.out.with_suffix('.tmp.onnx')
    with torch.no_grad():
        torch.onnx.export(model, (x, moves, legal), str(tmp), input_names=['features', 'moves', 'legal'],
                          output_names=names, opset_version=18, dynamo=False,
                          dynamic_axes={k: {0: 'batch'} for k in ['features', 'moves', 'legal', *names]})
    m = onnx.load(str(tmp)); onnx.checker.check_model(m)
    source = onnx.load(str(a.metadata_from), load_external_data=False)
    del m.metadata_props[:]
    for prop in source.metadata_props:
        m.metadata_props.append(prop)
    m.metadata_props.append(onnx.StringStringEntryProto(key='precision_islands', value=json.dumps(sorted(islands))))
    if a.move_capacity < 256:
        m.metadata_props.append(onnx.StringStringEntryProto(key='move_capacity', value=str(a.move_capacity)))
    if fp32_blocks is not None:
        m.metadata_props.append(onnx.StringStringEntryProto(key='fp32_blocks', value=json.dumps(sorted(fp32_blocks))))
    onnx.save(m, str(a.out)); tmp.unlink()
    print(json.dumps(dict(out=str(a.out), islands=sorted(islands), move_capacity=a.move_capacity)))


if __name__ == '__main__':
    main()
