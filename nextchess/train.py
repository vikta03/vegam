"""Resumable single-GPU full-coverage training. No frozen model components."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import signal
import time
import numpy as np
import torch
from .model import Config,Net,losses
from .stream import Stream


def atomic_torch(value,path):
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',default='data/corpus.sqlite')
    p.add_argument('--out',required=True);p.add_argument('--arm',choices=['standard','adaptive','wide'],default='adaptive')
    p.add_argument('--positions',type=int,default=5_000_000);p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--microbatch',type=int,default=64);p.add_argument('--batch',type=int,default=1024)
    p.add_argument('--device',default='cuda');p.add_argument('--save-every',type=int,default=250)
    p.add_argument('--seed',type=int,default=731);p.add_argument('--resume',action='store_true')
    p.add_argument('--tiny',action='store_true',help='contract tests only, never production')
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; do not silently train on CPU')
    c=Config() if a.arm!='wide' else Config(width=384,heads=12,ffn=576)
    if a.arm=='standard':c.adaptive=False;c.action_values=False
    if a.tiny:c=Config(width=32,heads=4,ffn=48,blocks=2)
    stream=Stream(a.manifest,seed=a.seed)
    target=a.positions if a.positions>0 else a.epochs*stream.positions
    model=Net(c).to(a.device);ema=deepcopy(model).eval()
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',enabled=a.device=='cuda')
    step=0;seen=0;stop=False;started=time.monotonic()
    with Path(a.manifest).open('rb') as f:
        manifest_sha=__import__('hashlib').file_digest(f,'sha256').hexdigest()
    code_sha=__import__('hashlib').sha256(b''.join((Path(__file__).parent/name).read_bytes()
        for name in ('train.py','raw.py','stream.py','model.py'))).hexdigest()
    def stopping(*_):
        nonlocal stop;stop=True
    signal.signal(signal.SIGTERM,stopping);signal.signal(signal.SIGINT,stopping)
    checkpoint=out/'latest.pt'
    if a.resume:
        old=torch.load(checkpoint,map_location=a.device,weights_only=False)
        if (old['config']!=asdict(c) or old['manifest_sha']!=manifest_sha or old['batch']!=a.batch
            or old['target']!=target or old.get('microbatch',a.microbatch)!=a.microbatch
            or old.get('code_sha')!=code_sha):raise ValueError('resume contract changed')
        model.load_state_dict(old['model']);ema.load_state_dict(old['ema']);optimizer.load_state_dict(old['optimizer'])
        scaler.load_state_dict(old['scaler']);stream.load_state_dict(old['stream']);step=old['step'];seen=old['seen']
        torch.set_rng_state(old['torch_rng'].cpu())
        if a.device=='cuda':torch.cuda.set_rng_state_all([s.cpu() for s in old['cuda_rng']])
        random.setstate(old['python_rng']);np.random.set_state(old['numpy_rng'])
    elif checkpoint.exists():raise FileExistsError('checkpoint exists; use --resume')
    initial_seen=seen
    def save():
        atomic_torch(dict(config=asdict(c),model=model.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(),stream=stream.state_dict(),step=step,seen=seen,batch=a.batch,microbatch=a.microbatch,
            manifest_sha=manifest_sha,code_sha=code_sha,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if a.device=='cuda' else [],
            python_rng=random.getstate(),numpy_rng=np.random.get_state(),target=target),checkpoint)
        status=dict(step=step,positions=seen,target=target,epoch=stream.epoch,partition_positions=stream.positions,
                    complete=seen>=target,config=asdict(c),parameters=sum(x.numel() for x in model.parameters()),
                    checkpoint=str(checkpoint.resolve()),state='complete' if seen>=target else 'paused' if stop else 'running')
        tmp=out/'status.tmp';tmp.write_text(json.dumps(status,indent=2)+'\n');os.replace(tmp,out/'status.json')
    save()
    with (out/'metrics.jsonl').open('a',buffering=1) as log:
        while seen<target and not stop:
            batch=stream.batch(min(a.batch,target-seen))
            if batch is None:continue
            count=len(batch['features']);batch.pop('record_ids');optimizer.zero_grad(set_to_none=True)
            progress=min(1.,seen/max(target,1))
            lr=3e-4*min(1.,(step+1)/1000)*(.1+.9*.5*(1+math.cos(math.pi*progress)))
            for group in optimizer.param_groups:group['lr']=lr
            stats={}
            # Normalize objectives over the full effective batch, not the number
            # of labels in an individual microbatch.
            denoms={k:max(1,int(batch[k].sum())) for k in ['state_valid','outcome_valid','action_valid','duration_valid']}
            for start in range(0,count,a.microbatch):
                end=min(count,start+a.microbatch)
                micro={k:torch.from_numpy(v[start:end]).to(a.device) for k,v in batch.items()}
                with torch.autocast(device_type=a.device,dtype=torch.float16,enabled=a.device=='cuda'):
                    prediction=model(micro['features'],micro['moves'],micro['legal'])
                _,parts=losses(prediction,micro,c.action_values)
                weights={'policy_ce':(end-start)/count,'state_ce':float(micro['state_valid'].sum())/denoms['state_valid'],
                         'action_ce':.5*float(micro['action_valid'].sum())/denoms['action_valid'] if c.action_values else 0.,
                         'outcome_ce':.25*float(micro['outcome_valid'].sum())/denoms['outcome_valid'],
                         'duration_nll':.05*float(micro['duration_valid'].sum())/denoms['duration_valid']}
                loss=sum(parts[k]*v for k,v in weights.items())
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss; latest atomic checkpoint is recoverable')
                scaler.scale(loss).backward()
                masks={'state_ce':'state_valid','action_ce':'action_valid','outcome_ce':'outcome_valid','duration_nll':'duration_valid'}
                for key,value in parts.items():
                    weight=float(micro[masks[key]].sum())/denoms[masks[key]] if key in masks else (end-start)/count
                    stats[key]=stats.get(key,0)+float(value.detach())*weight
            scaler.unscale_(optimizer);norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            scaler.step(optimizer);scaler.update();step+=1;seen+=count
            with torch.no_grad():
                decay=min(.999,(1+step)/(10+step))
                for avg,cur in zip(ema.parameters(),model.parameters()):avg.lerp_(cur,1-decay)
            seconds=time.monotonic()-started;rate=(seen-initial_seen)/max(seconds,1e-9)
            row=dict(step=step,positions=seen,epoch=stream.epoch,lr=lr,positions_s=rate,eta_s=(target-seen)/max(rate,1e-9),grad_norm=float(norm),**stats)
            log.write(json.dumps(row)+'\n')
            if step%10==0:print(json.dumps(row),flush=True)
            if step%a.save_every==0:save()
    save();print(json.dumps(dict(finished=seen>=target,positions=seen,checkpoint=str(checkpoint))),flush=True)


if __name__=='__main__':main()
