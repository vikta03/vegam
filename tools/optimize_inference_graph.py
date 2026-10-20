"""Remove redundant batch-shape plumbing from the preserved ONNX evaluator.

Only Reshape shape operands change: batch is inferred from tensor element count,
and all non-batch dimensions come from the model's checked configuration. No
learned initializer or arithmetic operator is changed.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from nextchess.transport import digest,atomic_json


def optimize(source,output):
    if output.exists():raise FileExistsError(output)
    model=onnx.load(str(source))
    props={x.key:json.loads(x.value) for x in model.metadata_props}
    assert props['schema']=='nextchess-v1'
    config=props['config'];width=config['width'];heads=config['heads'];rank=config['rank']
    assert width%heads==0
    nodes={n.name:n for n in model.graph.node}
    replacements=[]
    for block in range(config['blocks']):
        # The exporter creates these in QKV, adaptive-gate, attended-output order.
        shapes=[[-1,64,3,heads,width//heads],[-1,heads,1,rank],[-1,64,width]]
        assert config['adaptive']
        for suffix,shape in zip(('Reshape','Reshape_1','Reshape_2'),shapes):
            name=f'/blocks.{block}/{suffix}'
            node=nodes[name];assert node.op_type=='Reshape' and len(node.input)==2
            constant=f'next_static_shape_{block}_{suffix}'
            model.graph.initializer.append(numpy_helper.from_array(np.array(shape,np.int64),constant))
            replacements.append(dict(node=name,old_shape=node.input[1],new_shape=shape))
            node.input[1]=constant
    # Remove dead Shape/Gather/Concat plumbing, but keep every learned tensor.
    needed={out.name for out in model.graph.output};retained=[]
    for node in reversed(model.graph.node):
        if any(out in needed for out in node.output):
            retained.append(node);needed.update(node.input)
    removed=len(model.graph.node)-len(retained)
    del model.graph.node[:];model.graph.node.extend(reversed(retained))
    props['inference_optimization']=dict(kind='inferred-batch-reshapes',parent_model_sha256=digest(source),
                                        removed_nodes=removed,replacements=replacements)
    helper.set_model_props(model,{k:json.dumps(v) for k,v in props.items()})
    onnx.checker.check_model(model)
    output.parent.mkdir(parents=True,exist_ok=True)
    onnx.save(model,str(output))
    props['sha256']=digest(output)
    atomic_json(output.with_suffix('.json'),props)
    return dict(model=str(output),sha256=props['sha256'],removed_nodes=removed)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();print(json.dumps(optimize(a.source,a.output)))
