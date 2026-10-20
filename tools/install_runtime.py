#!/usr/bin/env python3
"""Install pinned libraries into a relocatable, package-local runtime.

Run explicitly: this downloads large vendor packages for --gpu.
It does not run an engine, compile a TensorRT plan, or change system packages.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    if sys.version_info < (3, 11):
        parser.error("Python 3.11 or newer is required")
    if args.gpu and sys.version_info >= (3, 14):
        parser.error("Pinned TensorRT 10.16 bindings require Python 3.11–3.13; use one of those for GPU setup")
    root = args.destination.resolve()
    root.mkdir(parents=True, exist_ok=True)
    mode = "gpu" if args.gpu else "cpu"
    requirements = Path(__file__).resolve().parents[1] / f"requirements-{mode}.txt"
    target = root / "_runtime"
    if target.exists():
        parser.error("_runtime already exists; use a fresh destination to install or change providers")
    target.mkdir()
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-compile",
                        "--target", str(target), "-r", str(requirements)], check=True)
        lib = root / "lib"
        lib.mkdir(exist_ok=True)
        files = sorted(p for p in target.rglob("*.so*") if p.is_file())
        links = {}
        for src in files:
            if not src.name.startswith("lib"):
                continue
            dst = lib / src.name
            if dst.exists() or dst.is_symlink():
                raise RuntimeError(f"Refusing to replace library {dst.name}")
            dst.symlink_to(os.path.relpath(src, lib))
            links[src.name] = str(src.relative_to(root))
        # The C++ linker uses libonnxruntime.so; the wheel uses a versioned file.
        ort = sorted(lib.glob("libonnxruntime.so.*"))
        if not ort:
            raise RuntimeError("The downloaded wheel contains no ONNX Runtime shared library")
        if not (lib / "libonnxruntime.so").exists():
            (lib / "libonnxruntime.so").symlink_to(ort[0].name)
        if not (lib / "libonnxruntime.so.1").exists():
            (lib / "libonnxruntime.so.1").symlink_to(ort[0].name)
        (root / "runtime.json").write_text(json.dumps({
            "mode": mode, "requirements": requirements.read_text(),
            "libraries": links, "python": sys.version.split()[0],
        }, indent=2) + "\n")
        print(f"Installed {mode} runtime in {root}. Keep _runtime and lib together.")
    except BaseException:
        print(f"Incomplete installation retained at {target}; use a fresh destination to retry.", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
