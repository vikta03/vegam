"""Atomic ONNX export of an immutable EMA snapshot; FP32 contract at I/O."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import torch
from .model import Config,Net


def export(checkpoint,output,half=False):
    torch.set_num_threads(2)
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=Net(Config(**ck['config']));model.load_state_dict(ck['ema']);model.eval()
    if half:model.half()
    x=torch.zeros(1,64,120);moves=torch.zeros(1,256,4,dtype=torch.int64);mask=torch.ones(1,256,dtype=torch.bool)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);tmp=output.with_suffix('.tmp.onnx')
    names=['policy_logits','wdl_logits','action_wdl_logits','duration_params','outcome_logits']
    torch.onnx.export(model,(x,moves,mask),str(tmp),input_names=['features','moves','legal'],output_names=names,
        dynamic_axes={key:{0:'batch'} for key in ['features','moves','legal',*names]},opset_version=18,dynamo=False)
    import onnx
    m=onnx.load(str(tmp));onnx.checker.check_model(m)
    metadata=dict(schema='nextchess-v1',channels=120,max_moves=256,config=ck['config'],positions=ck['seen'],step=ck['step'],
                  manifest_sha=ck['manifest_sha'],precision='float16' if half else 'float32',trained_from_random=True,
                  sha256=hashlib.sha256(tmp.read_bytes()).hexdigest(),strength='unrated')
    if 'initialization' in ck:
        metadata.update(initialization=ck['initialization'],migrations=ck.get('migrations',[]),
                        prior_training_exposures=ck['initialization'].get('prior_exposures',0),
                        coverage_note='positions counts accepted cloud exposures; prior pilot exposures are separate, not unique positions')
    onnx.helper.set_model_props(m,{k:json.dumps(v) for k,v in metadata.items() if k!='sha256'})
    onnx.save(m,str(tmp));metadata['sha256']=hashlib.sha256(tmp.read_bytes()).hexdigest();os.replace(tmp,output)
    output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('output');p.add_argument('--half',action='store_true')
    a=p.parse_args();print(json.dumps(export(a.checkpoint,a.output,a.half)))
