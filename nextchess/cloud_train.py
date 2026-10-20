"""Bounded single-node DDP trainer. Running this does not stop Pod billing.

The pilot trainer remains unchanged. Checkpoints use a separate resume contract.
"""
import argparse
import fcntl
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time
import numpy as np
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import functional as F
from .cloud_data import ParallelStream
from .model import Config, Net
from .transport import atomic_json, digest

NAMES = ('policy_ce', 'policy_entropy', 'state_ce', 'action_ce', 'outcome_ce', 'duration_nll')


def distributed():
    return dist.is_available() and dist.is_initialized()


def reduce_sum(value):
    if distributed(): dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def any_rank(value, device):
    flag = torch.tensor(int(value), device=device)
    if distributed(): dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def counts(batch):
    valid = batch['sample_valid']
    n = valid.sum()
    return torch.stack((n, n, (batch['state_valid'] & valid).sum(),
                        (batch['action_valid'] & valid[:, None]).sum(),
                        (batch['outcome_valid'] & valid).sum(),
                        (batch['duration_valid'] & valid).sum())).float()


def objective_sums(out, batch):
    logits, state, action, duration, outcome = out
    valid = batch['sample_valid']
    policy = batch['policy']
    ce = -(policy * F.log_softmax(logits, -1)).sum(-1)
    entropy = -(policy * policy.clamp_min(1e-30).log()).sum(-1)
    state_loss = -(batch['state_wdl'] * F.log_softmax(state, -1)).sum(-1)
    action_loss = -(batch['action_wdl'] * F.log_softmax(action, -1)).sum(-1)
    outcome_loss = -(batch['outcome'] * F.log_softmax(outcome, -1)).sum(-1)
    y = batch['duration'].log1p()
    mu, log_sigma = duration[:, 0], duration[:, 1].clamp(-3, 3)
    z = (y-mu)*torch.exp(-log_sigma)
    duration_loss = torch.where(batch['censored'], -torch.special.log_ndtr(-z), .5*z.square()+log_sigma)
    return torch.stack(((ce*valid).sum(), (entropy*valid).sum(),
                        (state_loss*(batch['state_valid'] & valid)).sum(),
                        (action_loss*(batch['action_valid'] & valid[:, None])).sum(),
                        (outcome_loss*(batch['outcome_valid'] & valid)).sum(),
                        (duration_loss*(batch['duration_valid'] & valid)).sum()))


def rng_state(device):
    return dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
                python=random.getstate(), numpy=np.random.get_state())


def restore_rng(state, device):
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None: torch.cuda.set_rng_state(state['cuda'].cpu(), device)
    random.setstate(state['python']); np.random.set_state(state['numpy'])


def update(model, optimizer, scaler, batch, *, precision='bf16', retries=2, action_values=True):
    """Reject a bad attempt before optimizer state or caller counters advance."""
    device = batch['features'].device
    world = dist.get_world_size() if distributed() else 1
    denominators = reduce_sum(counts(batch)).clamp_min(1)
    weights = batch['features'].new_tensor([1, 0, 1, .5 if action_values else 0, .25, .05])
    before = rng_state(device)
    for attempt in range(retries+1):
        restore_rng(before, device)
        optimizer.zero_grad(set_to_none=True)
        # BF16/FP16 retry ultimately falls back to FP32 for this same batch.
        mode = precision if attempt == 0 or (precision == 'fp16' and attempt == 1) else 'fp32'
        ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16 if mode == 'bf16' else torch.float16) \
              if mode != 'fp32' and device.type == 'cuda' else nullcontext()
        with ctx:
            out = model(batch['features'], batch['moves'], batch['legal'])
        sums = objective_sums(out, batch)
        loss = (sums * weights / denominators).sum() * world
        # Every rank executes backward, including on a bad loss; the ensuing
        # all-reduced finite decision rejects the entire global update.
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=False)
        bad = not bool((torch.isfinite(norm) & torch.isfinite(sums).all()).item())
        if not any_rank(bad, device):
            scaler.step(optimizer); scaler.update()
            metrics = reduce_sum(sums.detach()) / denominators
            return metrics, float(norm), attempt
        if scaler.is_enabled():
            scaler.update(new_scale=max(scaler.get_scale() * .5, 1.))
    optimizer.zero_grad(set_to_none=True)
    restore_rng(before, device)
    raise FloatingPointError('same global batch failed bounded numerical recovery; no update committed')


def to_device(batch, device):
    result = {}
    for key, value in batch.items():
        if key == 'record_ids': continue
        tensor = torch.from_numpy(value)
        if device.type == 'cuda': tensor = tensor.pin_memory()
        result[key] = tensor.to(device, non_blocking=device.type == 'cuda')
    return result


@torch.no_grad()
def update_ema(averages, parameters, decay):
    # One foreach launch for the same lerp equation, rather than one per tensor.
    torch._foreach_lerp_(averages, parameters, 1-decay)


def source_hash():
    names = ('cloud_train.py', 'cloud_data.py', 'fast_raw.py', 'transport.py', 'raw.py', 'model.py')
    return hashlib.sha256(b''.join((Path(__file__).parent/n).read_bytes() for n in names)).hexdigest()


def save_torch(path, value):
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        torch.save(value, f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', default='data/corpus.sqlite'); p.add_argument('--transport')
    p.add_argument('--out', required=True); p.add_argument('--arm', choices=['standard', 'adaptive', 'wide'], default='wide')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--precision', choices=['bf16', 'fp16', 'fp32'], default='bf16')
    p.add_argument('--batch', type=int, default=1024); p.add_argument('--workers', type=int, default=2)
    p.add_argument('--prefetch', type=int, default=2); p.add_argument('--buffer', type=int, default=4096)
    p.add_argument('--epochs', type=int, default=3); p.add_argument('--positions', type=int, default=0)
    p.add_argument('--max-seconds', type=float, default=600, help='trainer deadline only; external Pod controller is required')
    p.add_argument('--save-seconds', type=float, default=300); p.add_argument('--seed', type=int, default=731)
    p.add_argument('--resume', action='store_true'); p.add_argument('--init-checkpoint')
    p.add_argument('--compile', action='store_true'); p.add_argument('--tiny', action='store_true')
    p.add_argument('--attention', choices=['manual','sdpa'], default='manual')
    p.add_argument('--stop-after-steps', type=int, default=0, help='bounded recovery testing')
    p.add_argument('--backup-config', help='S3 configuration JSON; credentials come from environment/profile')
    a = p.parse_args()
    if min(a.batch, a.epochs, a.max_seconds, a.save_seconds) <= 0 or a.positions < 0:
        p.error('budgets and batch sizes must be positive')
    if a.resume and a.init_checkpoint: p.error('resume and initialization are distinct operations')
    rank, world, local_rank = int(os.getenv('RANK', 0)), int(os.getenv('WORLD_SIZE', 1)), int(os.getenv('LOCAL_RANK', 0))
    if world > 1:
        if a.device == 'cuda': torch.cuda.set_device(local_rank)
        dist.init_process_group('nccl' if a.device == 'cuda' else 'gloo', timeout=timedelta(seconds=180))
    device = torch.device(f'cuda:{local_rank}' if a.device == 'cuda' else 'cpu')
    if device.type == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA required; no silent CPU fallback')
    torch.set_num_threads(2 if device.type == 'cpu' else 1)
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    c = Config(width=384, heads=12, ffn=576) if a.arm == 'wide' else Config()
    if a.arm == 'standard': c.adaptive = c.action_values = False
    if a.tiny: c = Config(width=32, heads=4, ffn=48, blocks=2)
    net = Net(c).to(device)
    net.set_attention(a.attention)
    initialization = dict(kind='random', seed=a.seed, prior_exposures=0)
    if a.init_checkpoint:
        ck = torch.load(a.init_checkpoint, map_location='cpu', weights_only=False)
        if ck['config'] != asdict(c): raise ValueError('initialization architecture differs')
        net.load_state_dict(ck['model'])
        initialization = dict(kind='pilot_weights_fresh_optimizer', sha256=digest(a.init_checkpoint),
                              prior_exposures=ck['seen'], source_manifest_sha=ck['manifest_sha'])
    ema = deepcopy(net).eval()
    for param in ema.parameters(): param.requires_grad_(False)
    ema_parameters, net_parameters = list(ema.parameters()), list(net.parameters())
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=.01, fused=device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and a.precision == 'fp16')
    model = torch.compile(net) if a.compile else net
    if world > 1:
        model = DDP(model, device_ids=[local_rank] if device.type == 'cuda' else None, gradient_as_bucket_view=True)
    out = Path(a.out).resolve()
    if rank == 0: out.mkdir(parents=True, exist_ok=True)
    if world > 1: dist.barrier()
    lock = None
    lock_error = [None]
    if rank == 0:
        try:
            lock = (out/'.training.lock').open('a')
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError: lock_error[0] = 'another trainer owns this output directory'
    if world > 1: dist.broadcast_object_list(lock_error, src=0)
    if lock_error[0]: raise RuntimeError(lock_error[0])
    reader = ParallelStream(a.manifest, batch=a.batch, seed=a.seed, capacity=a.buffer, rank=rank, world=world,
                            workers=a.workers, prefetch=a.prefetch, transport=a.transport)
    target = a.positions or a.epochs*reader.positions
    contract = dict(schema='next-cloud-train-v1', config=asdict(c), manifest_sha=digest(a.manifest),
                    code_sha=source_hash(), batch=a.batch, world=world, precision=a.precision,
                    compiled=a.compile, attention=a.attention, seed=a.seed, target=target, buffer=a.buffer,
                    torch_version=torch.__version__, numpy_version=np.__version__)
    step = seen = rejected = 0
    migrations = []
    stopped = False
    def stop(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    checkpoint = out/'latest.pt'
    if a.resume:
        status = json.loads((out/'status.json').read_text())
        if digest(checkpoint) != status['sha256']: raise ValueError('checkpoint checksum differs')
        ck = torch.load(checkpoint, map_location=device, weights_only=False)
        if ck['contract'] != contract: raise ValueError('cloud resume contract changed')
        if ck['reader']['exposed'] != ck['seen']: raise ValueError('checkpoint coverage/cursor disagree')
        net.load_state_dict(ck['model']); ema.load_state_dict(ck['ema'])
        optimizer.load_state_dict(ck['optimizer']); scaler.load_state_dict(ck['scaler'])
        reader.load_state_dict(ck['reader']); restore_rng(ck['rank_rng'][rank], device)
        step, seen, rejected = ck['step'], ck['seen'], ck['rejected_attempts']
        initialization = ck['initialization']
        migrations = ck.get('migrations', [])
    elif checkpoint.exists():
        raise FileExistsError('checkpoint exists; explicit resume required')
    if initialization.get('source_manifest_sha', contract['manifest_sha']) != contract['manifest_sha']:
        raise ValueError('pilot initialization uses a different corpus')
    backup = None
    if a.backup_config and rank == 0:
        from .cloud_storage import Store
        backup = Store.from_config(a.backup_config)
    started = time.monotonic(); initial_seen = seen
    last_save = started
    def save(reason):
        local_rng = rng_state(device)
        rank_rng = [None]*world
        if world > 1: dist.all_gather_object(rank_rng, local_rng)
        else: rank_rng = [local_rng]
        failure = None
        if rank == 0:
            try:
                folder = out/'checkpoints'; folder.mkdir(exist_ok=True)
                path = folder/f'step-{step:09d}.pt'
                state = dict(config=asdict(c), model=net.state_dict(), ema=ema.state_dict(), optimizer=optimizer.state_dict(),
                             scaler=scaler.state_dict(), contract=contract, reader=reader.state_dict(), rank_rng=rank_rng,
                             step=step, seen=seen, target=target, manifest_sha=contract['manifest_sha'], code_sha=contract['code_sha'],
                             initialization=initialization, migrations=migrations,
                             rejected_attempts=rejected, replay_exposures=0)
                save_torch(path, state)
                link = out/'latest.tmp.pt'
                if link.exists(): link.unlink()
                os.link(path, link); os.replace(link, checkpoint)
                status = dict(schema=contract['schema'], reason=reason, complete=seen >= target, positions=seen,
                              target=target, step=step, completed_passes=seen//reader.positions,
                              partition_positions=reader.positions, replay_exposures=0, rejected_attempts=rejected,
                              world=world, precision=a.precision,
                              checkpoint=str(path), sha256=digest(path), elapsed_seconds=time.monotonic()-started,
                              model_parameters=sum(p.numel() for p in net.parameters()), initialization=initialization,
                              migrations=migrations)
                atomic_json(out/'status.json', status)
                if backup: backup.checkpoint(path, status)
            except Exception as error:
                failure = f'{type(error).__name__}: {error}'
        notice = [failure]
        if world > 1: dist.broadcast_object_list(notice, src=0)
        if notice[0]: raise RuntimeError(f'checkpoint publication failed: {notice[0]}')
    log = (out/'metrics.jsonl').open('a', buffering=1) if rank == 0 else None
    try:
        save('initial_or_resumed')
        while seen < target:
            exhausted = stopped or time.monotonic()-started >= a.max_seconds or (a.stop_after_steps and step >= a.stop_after_steps)
            if any_rank(exhausted, device): break
            # Explicit remaining-budget truncation occurs at the reference
            # scheduler before any data is accepted. Recreate from committed
            # state only for a non-epoch-aligned final requested chunk.
            if target-seen < a.batch:
                state = reader.state_dict(); reader.close()
                reader = ParallelStream(a.manifest, batch=target-seen, seed=a.seed, capacity=a.buffer,
                                        rank=rank, world=world, workers=a.workers, prefetch=1, transport=a.transport)
                reader.load_state_dict(state)
            step_started = time.monotonic()
            batch, count = reader.peek()
            read_finished = time.monotonic()
            tensor = to_device(batch, device)
            transfer_enqueued = time.monotonic()
            lr = 3e-4*min(1., (step+1)/1000)*(.1+.9*.5*(1+math.cos(math.pi*seen/target)))
            for group in optimizer.param_groups: group['lr'] = lr
            try:
                metrics, norm, attempts = update(model, optimizer, scaler, tensor, precision=a.precision, action_values=c.action_values)
            except FloatingPointError:
                save('numerical_failure_uncommitted_batch')
                raise
            update_finished = time.monotonic()
            reader.commit(); step += 1; seen += count; rejected += attempts
            decay = min(.999, (1+step)/(10+step))
            update_ema(ema_parameters, net_parameters, decay)
            if rank == 0:
                values = dict(zip(NAMES, metrics.cpu().tolist()))
                elapsed = time.monotonic()-started
                row = dict(step=step, positions=seen, lr=lr, grad_norm=norm, retry_attempts=attempts,
                           elapsed_seconds=elapsed, positions_s=(seen-initial_seen)/max(elapsed, 1e-9),
                           reader_wait_ms=1000*(read_finished-step_started),
                           transfer_enqueue_ms=1000*(transfer_enqueued-read_finished),
                           update_wall_ms=1000*(update_finished-transfer_enqueued),
                           **values, policy_kl=values['policy_ce']-values['policy_entropy'])
                log.write(json.dumps(row)+'\n')
                if step % 25 == 0: print(json.dumps(row), flush=True)
            save_now = time.monotonic()-last_save >= a.save_seconds or seen % reader.positions == 0
            if any_rank(save_now, device): save('periodic_or_pass_end'); last_save = time.monotonic()
        save('complete' if seen >= target else 'paused_at_deadline')
    finally:
        reader.close()
        if log: log.close()
        if lock: lock.close()
        if distributed(): dist.destroy_process_group()


if __name__ == '__main__': main()
