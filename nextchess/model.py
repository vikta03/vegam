"""Square transformer with adaptive relations and a joint legal action decoder."""
from dataclasses import dataclass,asdict
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    width:int=256
    blocks:int=8
    heads:int=8
    ffn:int=384
    rank:int=16
    adaptive:bool=True
    action_values:bool=True
    channels:int=120


class Block(nn.Module):
    def __init__(self,c):
        super().__init__();self.heads=c.heads;self.depth=c.width//c.heads;self.rank=c.rank
        self.norm1=nn.LayerNorm(c.width);self.norm2=nn.LayerNorm(c.width)
        self.qkv=nn.Linear(c.width,c.width*3,bias=False);self.out=nn.Linear(c.width,c.width,bias=False)
        self.ff=nn.Sequential(nn.Linear(c.width,c.ffn),nn.GELU(),nn.Linear(c.ffn,c.width))
        self.bias=nn.Parameter(torch.zeros(c.heads,64,64));self.adaptive=c.adaptive
        # Execution choice only; checkpoint parameters and exported equations
        # are identical. Manual remains the inference/export control.
        self.attention_impl='manual'
        if c.adaptive:
            self.compress=nn.Linear(c.width,8)
            self.gates=nn.Sequential(nn.Linear(64*8,128),nn.GELU(),nn.Linear(128,c.heads*c.rank),nn.Tanh())
            self.left=nn.Parameter(torch.randn(c.heads,64,c.rank)*0.02)
            self.right=nn.Parameter(torch.randn(c.heads,64,c.rank)*0.02)

    def forward(self,x):
        z=self.norm1(x);b=x.shape[0]
        q,k,v=self.qkv(z).reshape(b,64,3,self.heads,self.depth).permute(2,0,3,1,4).unbind(0)
        bias=self.bias.unsqueeze(0)
        if self.adaptive:
            g=self.gates(self.compress(z).flatten(1)).reshape(b,self.heads,1,self.rank)
            bias=bias+torch.matmul(self.left.unsqueeze(0)*g,self.right.transpose(-1,-2).unsqueeze(0))
        if self.attention_impl == 'sdpa':
            attended=F.scaled_dot_product_attention(q,k,v,attn_mask=bias,dropout_p=0.)
        else:
            attn=torch.softmax((torch.matmul(q,k.transpose(-1,-2))/math.sqrt(self.depth)+bias).float(),dim=-1).to(v.dtype)
            attended=torch.matmul(attn,v)
        x=x+self.out(attended.transpose(1,2).reshape(b,64,-1))
        return x+self.ff(self.norm2(x))


class Net(nn.Module):
    def __init__(self,c=Config()):
        super().__init__();self.config=c
        if c.width%c.heads:raise ValueError('width must divide heads')
        self.input=nn.Linear(c.channels,c.width);self.position=nn.Parameter(torch.randn(1,64,c.width)*.02)
        self.blocks=nn.ModuleList(Block(c) for _ in range(c.blocks));self.norm=nn.LayerNorm(c.width)
        self.global_proj=nn.Sequential(nn.Linear(c.width,128),nn.GELU())
        self.state=nn.Linear(128,3);self.outcome=nn.Linear(128,3);self.duration=nn.Linear(128,2)
        self.source=nn.Linear(c.width,64);self.destination=nn.Linear(c.width,64)
        self.context=nn.Linear(128,64);self.promotion=nn.Embedding(5,64);self.castle=nn.Embedding(3,64)
        self.joint=nn.Sequential(nn.Linear(64,64),nn.GELU(),nn.Linear(64,4))

    def set_attention(self, implementation):
        if implementation not in ('manual','sdpa'): raise ValueError('unsupported attention implementation')
        for block in self.blocks: block.attention_impl=implementation

    def forward(self,features,moves,legal):
        x=self.input(features.to(self.input.weight.dtype))+self.position
        for block in self.blocks:x=block(x)
        x=self.norm(x);context=self.global_proj(x.mean(1))
        src=torch.gather(self.source(x),1,moves[:,:,0,None].expand(-1,-1,64))
        dst=torch.gather(self.destination(x),1,moves[:,:,1,None].expand(-1,-1,64))
        g=self.context(context)[:,None,:]
        joint=self.joint(F.gelu(src+dst+src*dst/8+g+self.promotion(moves[:,:,2])+self.castle(moves[:,:,3])))
        logits=joint[:,:,0].float().masked_fill(~legal,-1e9)
        return logits,self.state(context).float(),joint[:,:,1:].float(),self.duration(context).float(),self.outcome(context).float()


def losses(out,batch,action_values=True):
    logits,state,action,duration,outcome=out
    def mean_mask(values,mask):return (values*mask).sum()/mask.sum().clamp_min(1)
    target=batch['policy'];logp=F.log_softmax(logits,dim=-1)
    ce=-(target*logp).sum(-1).mean()
    entropy=-(target*target.clamp_min(1e-30).log()).sum(-1).mean()
    value=mean_mask(-(batch['state_wdl']*F.log_softmax(state,-1)).sum(-1),batch['state_valid'])
    act=mean_mask(-(batch['action_wdl']*F.log_softmax(action,-1)).sum(-1),batch['action_valid'])
    result=mean_mask(-(batch['outcome']*F.log_softmax(outcome,-1)).sum(-1),batch['outcome_valid'])
    # Right-censored log-normal duration. Missing observations contribute zero.
    y=batch['duration'].log1p();mu=duration[:,0];log_sigma=duration[:,1].clamp(-3,3)
    z=(y-mu)*torch.exp(-log_sigma)
    observed=.5*z.square()+log_sigma
    censored=-torch.special.log_ndtr(-z)
    dur=mean_mask(torch.where(batch['censored'],censored,observed),batch['duration_valid'])
    total=ce+value+(.5*act if action_values else 0)+.25*result+.05*dur
    return total,dict(policy_ce=ce,policy_entropy=entropy,policy_kl=ce-entropy,state_ce=value,action_ce=act,outcome_ce=result,duration_nll=dur)
