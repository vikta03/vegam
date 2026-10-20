"""Wrap a serialized TensorRT engine as an ONNX Runtime EPContext model.

The wrapper has the same inputs and outputs as the source ONNX model and one
com.microsoft EPContext node whose ep_cache_context names the engine file,
relative to the wrapper's own directory.
"""
import argparse
from pathlib import Path

import onnx
from onnx import helper


def wrap(source, engine, output, architecture='86'):
    source, engine, output = Path(source), Path(engine), Path(output)
    if engine.parent.resolve() != output.parent.resolve():
        raise ValueError('engine must sit next to the wrapper; ep_cache_context is relative')
    model = onnx.load(str(source), load_external_data=False)
    node = helper.make_node(
        'EPContext', [i.name for i in model.graph.input], [o.name for o in model.graph.output],
        name='next_trt_engine', domain='com.microsoft',
        main_context=1, embed_mode=0, ep_cache_context=engine.name,
        hardware_architecture=architecture, source='TensorrtExecutionProvider')
    graph = helper.make_graph([node], 'next_trt', list(model.graph.input), list(model.graph.output))
    wrapped = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18),
                                                      helper.make_opsetid('com.microsoft', 1)])
    wrapped.ir_version = model.ir_version
    for prop in model.metadata_props:
        wrapped.metadata_props.append(prop)
    onnx.save(wrapped, str(output))
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source'); p.add_argument('engine'); p.add_argument('output')
    p.add_argument('--architecture', default='86')
    a = p.parse_args()
    print(wrap(a.source, a.engine, a.output, a.architecture))
