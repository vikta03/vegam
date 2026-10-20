#!/usr/bin/env python3
"""Build the frozen engine and assemble a release without changing search code."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ort-libdir", type=Path, default=ROOT / "runtime/lib")
    p.add_argument("--build-dir", type=Path, default=ROOT / "build")
    p.add_argument("--package", type=Path, default=ROOT / "dist/vegam-linux-x86_64")
    p.add_argument("--model", type=Path, default=ROOT / "models/model.onnx")
    p.add_argument("--jobs", type=int, default=2)
    a = p.parse_args()
    if not 1 <= a.jobs <= 16:
        p.error("--jobs must be between 1 and 16")
    for path in (a.ort_libdir / "libonnxruntime.so", a.model):
        if not path.is_file():
            p.error(f"Missing {path}; install the runtime and fetch the model first")
    model_hash = hashlib.sha256(a.model.read_bytes()).hexdigest()
    expected = json.loads((ROOT / "models.json").read_text())["model.onnx"]["sha256"]
    if model_hash != expected:
        p.error("Model checksum differs from the released epoch-3 network")
    cmd = ["meson", "setup", str(a.build_dir.resolve()), str(ROOT / "engine"),
           "--buildtype=release", "--wrap-mode=nodownload", "--prefix=/usr/local",
           "-Dbuild_backends=false", "-Ddefault_backend=next-ort",
           "-Dgtest=false", "-Dnative_arch=false", "-Db_lto=false",
           "-Dnext_ort_include=" + str(ROOT / "third_party/onnxruntime/include"),
           "-Dnext_ort_libdir=" + str(a.ort_libdir.resolve())]
    if (a.build_dir / "build.ninja").exists():
        cmd.insert(2, "--reconfigure")
    subprocess.run(cmd, check=True)
    subprocess.run(["ninja", "-C", str(a.build_dir), f"-j{a.jobs}"], check=True)
    package = a.package.resolve()
    package.mkdir(parents=True, exist_ok=True)
    if (package / "next-chess").exists():
        p.error("Package already contains a binary; choose a fresh --package directory")
    # Meson's install step removes its temporary absolute dependency RPATH.
    # DESTDIR keeps the install entirely inside this temporary directory.
    with tempfile.TemporaryDirectory(prefix="vegam-install-") as temporary:
        subprocess.run(["meson", "install", "-C", str(a.build_dir),
                        "--no-rebuild", "--destdir", temporary], check=True)
        installed = Path(temporary) / "usr/local/bin/lc0"
        dynamic = subprocess.check_output(["readelf", "-d", str(installed)], text=True)
        paths = re.findall(r"\((?:RUNPATH|RPATH)\).*?\[([^\]]*)\]", dynamic)
        if paths != ["$ORIGIN/lib"]:
            raise RuntimeError(f"Non-relocatable executable library paths: {paths}")
        shutil.copy2(installed, package / "next-chess")
    shutil.copy2(a.model, package / "model.onnx")
    for name in ("vegamt-tf3-q", "lc0.config", "COPYING", "LICENSE-MODEL.md",
                 "THIRD_PARTY.md", "requirements-cpu.txt", "requirements-gpu.txt", "models.json"):
        shutil.copy2(ROOT / name, package / name)
    shutil.copytree(ROOT / "third_party/licenses", package / "third_party/licenses")
    (package / "third_party/onnxruntime").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "third_party/onnxruntime/LICENSE",
                 package / "third_party/onnxruntime/LICENSE")
    (package / "tools").mkdir(exist_ok=True)
    for name in ("install_runtime.py", "prepare_tensorrt.py", "wrap_trt_engine.py", "fetch_models.py"):
        shutil.copy2(ROOT / "tools" / name, package / "tools" / name)
    (package / "vegamt-tf3-q").chmod(0o755)
    (package / "SOURCE.txt").write_text(
        "Corresponding source: https://github.com/vikta03/vegam/tree/v1.0.0\n"
        "The source archive accompanies this binary in the same release.\n")
    (package / "README.txt").write_text(
        "VEGAM-TF3-Q by Remmy Victor (@vikta03)\n\n"
        "Install libraries once (large download for GPU):\n"
        "  python3 tools/install_runtime.py --destination . --gpu\n"
        "Or omit --gpu for CPU only.\n\n"
        "Add ./vegamt-tf3-q to a chess GUI using the UCI protocol.\n"
        "TensorRT acceleration must be prepared for this machine:\n"
        "  python3 tools/fetch_models.py --destination . --trt\n"
        "  PYTHONPATH=./_runtime LD_LIBRARY_PATH=./lib python3 tools/prepare_tensorrt.py --package .\n"
        "Preparation uses the GPU and can take minutes. It is never automatic.\n"
        "Full instructions: https://github.com/vikta03/vegam\n")
    files = {str(f.relative_to(package)): hashlib.sha256(f.read_bytes()).hexdigest()
             for f in sorted(package.rglob("*")) if f.is_file()}
    (package / "manifest.json").write_text(json.dumps({
        "name": "VEGAM-TF3-Q", "release": "v1.0.0", "files": files,
        "default_search": "classic", "runtime_included": False,
    }, indent=2) + "\n")
    print(package)


if __name__ == "__main__":
    main()
