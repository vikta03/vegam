"""Held-out objectives and export latency, independent of training selection."""
import argparse
import ctypes
import json
from pathlib import Path
import time
import numpy as np
import onnxruntime as ort
import torch
from .model import losses
from .stream import Stream

def run(model,manifest,positions=32768,provider='cpu',out=None):
    torch.set_num_threads(2)
    if provider=='cuda':
        ctypes.CDLL('libcudnn.so.9',mode=ctypes.RTLD_GLOBAL)
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(model),sess_options=options,providers=['CUDAExecutionProvider' if provider=='cuda' else 'CPUExecutionProvider'])
    if provider=='cuda' and 'CUDAExecutionProvider' not in session.get_providers():raise RuntimeError('CUDA provider failed')
    stream=Stream(manifest,split='valid',seed=917);sums={};denoms={};seen=0;top=0;started=time.monotonic()
    masks={'state_ce':'state_valid','action_ce':'action_valid','outcome_ce':'outcome_valid','duration_nll':'duration_valid'}
    while seen<positions:
        batch=stream.batch(min(32,positions-seen));batch.pop('record_ids');seen+=len(batch['features'])
        inputs={k:batch[k] for k in ('features','moves','legal')}
        outputs=session.run(None,inputs);tensors=tuple(torch.from_numpy(v) for v in outputs)
        _,parts=losses(tensors,{k:torch.from_numpy(v) for k,v in batch.items()})
        for k,value in parts.items():
            n=int(batch[masks[k]].sum()) if k in masks else len(batch['features'])
            sums[k]=sums.get(k,0)+float(value)*n;denoms[k]=denoms.get(k,0)+n
        top+=int((outputs[0].argmax(1)==batch['policy'].argmax(1)).sum())
    metrics={k:sums[k]/max(1,denoms[k]) for k in sums}
    timing={}
    for n in (1,2,4,8,16,32):
        inputs={k:np.repeat(batch[k][:1],n,axis=0) for k in ('features','moves','legal')}
        for _ in range(5):session.run(None,inputs)
        values=[]
        for _ in range(30):
            start=time.perf_counter();session.run(None,inputs);values.append((time.perf_counter()-start)*1000)
        timing[n]={'median_ms':float(np.median(values)),'p95_ms':float(np.percentile(values,95)),'positions_s':1000*n/np.median(values)}
    receipt={'model':str(Path(model).resolve()),'positions':seen,'split':'valid','seed':917,'provider':provider,
        'policy_top1_teacher':top/seen,'valid_labels':denoms,'losses':metrics,'batch_latency':timing,'seconds':time.monotonic()-started}
    if out:Path(out).write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--manifest',default='data/corpus.sqlite')
    p.add_argument('--positions',type=int,default=32768);p.add_argument('--provider',choices=['cpu','cuda'],default='cpu');p.add_argument('--out',required=True);a=p.parse_args()
    print(json.dumps(run(a.model,a.manifest,a.positions,a.provider,a.out)))
