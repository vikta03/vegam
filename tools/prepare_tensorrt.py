#!/usr/bin/env python3
"""Explicit GPU job: compile this model for the local GPU using TensorRT 10.16.1.

Strong types, FP32 residual/LN/GELU/heads, TF32 disabled, batches 1/8/32.
The portable source graph is distinct from the CPU/CUDA fallback graph.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package", type=Path, required=True)
    a = p.parse_args()
    root = a.package.resolve()
    models = json.loads((root / "models.json").read_text())
    for name in ("model.onnx", "model.trt-source.onnx"):
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != models[name]["sha256"]:
            p.error(f"Wrong model: {name}")
    if any((root / n).exists() for n in ("model.trt.engine", "model.trt.onnx")):
        p.error("A TensorRT plan already exists; use a fresh package to rebuild safely")
    import tensorrt as trt
    if not trt.__version__.startswith("10.16.1"):
        p.error("This recipe requires TensorRT 10.16.1")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse((root / "model.trt-source.onnx").read_bytes()):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    config.builder_optimization_level = 3
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1024 * 1024 * 1024)
    profile = builder.create_optimization_profile()
    for name, tail in [("features", (64, 120)), ("moves", (256, 4)), ("legal", (256,))]:
        profile.set_shape(name, (1, *tail), (8, *tail), (32, *tail))
    config.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT did not produce an engine")
    with tempfile.TemporaryDirectory(prefix=".trt-build-", dir=root) as temporary:
        stage = Path(temporary)
        engine = stage / "model.trt.engine"
        engine.write_bytes(bytes(plan))
        # Query the architecture through the engine's public inspector metadata.
        # Wrapper field is informational; deserialization remains the compatibility check.
        gpu = subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,compute_cap,driver_version",
            "--format=csv,noheader"], text=True, timeout=10).strip().splitlines()[0]
        architecture = gpu.split(",")[1].strip().replace(".", "")
        from wrap_trt_engine import wrap
        wrap(root / "model.trt-source.onnx", engine, stage / "model.trt.onnx", architecture)
        # Require successful local deserialization before making the wrapper discoverable.
        runtime = trt.Runtime(logger)
        loaded = runtime.deserialize_cuda_engine(engine.read_bytes())
        if loaded is None:
            raise RuntimeError("The freshly built TensorRT engine cannot be loaded")
        engine.replace(root / engine.name)
        (stage / "model.trt.onnx").replace(root / "model.trt.onnx")
        (root / "tensorrt-build.json").write_text(json.dumps({
            "gpu": gpu, "tensorrt": trt.__version__, "tf32": False,
            "source_sha256": models["model.trt-source.onnx"]["sha256"],
            "engine_sha256": hashlib.sha256((root / engine.name).read_bytes()).hexdigest(),
            "batch_min_opt_max": [1, 8, 32], "builder_level": 3,
        }, indent=2) + "\n")
    print("TensorRT prepared. The next engine startup reports the selected provider.")


if __name__ == "__main__":
    main()
