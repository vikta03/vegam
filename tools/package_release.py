#!/usr/bin/env python3
"""Package checked source/binary/media assets; never upload or run an engine."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def add_file(archive, src, name):
    info = archive.gettarinfo(str(src), arcname=name)
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    with src.open("rb") as data:
        archive.addfile(info, data)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--verification", type=Path, required=True)
    a = p.parse_args()
    proof = json.loads(a.verification.read_text())
    if proof.get("passed") is not True:
        p.error("Passing release verification is required")
    binary_hash = hashlib.sha256((a.package / "next-chess").read_bytes()).hexdigest()
    if proof.get("binary_sha256") != binary_hash:
        p.error("Verification does not identify this binary")
    a.assets.mkdir(parents=True, exist_ok=True)
    with tarfile.open(a.assets / "vegam-linux-x86_64.tar.gz", "w:gz") as tar:
        for src in sorted(a.package.rglob("*")):
            rel = src.relative_to(a.package)
            if rel.parts[0] in {"lib", "_runtime"} or "trt-cache" in str(rel):
                continue
            if src.is_file() and src.name not in {"model.trt.engine", "model.trt.onnx", "runtime.json", "tensorrt-build.json"}:
                add_file(tar, src, "vegam-linux-x86_64/" + str(rel))
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).split(b"\0")
    with tarfile.open(a.assets / "vegam-v1.0.0-source.tar.gz", "w:gz") as tar:
        for raw in paths:
            if raw:
                rel = raw.decode()
                add_file(tar, ROOT / rel, "vegam-v1.0.0/" + rel)
    shutil.copy2(a.verification, a.assets / "verification.json")
    selected = [x for x in sorted(a.assets.iterdir()) if x.is_file() and x.name != "SHA256SUMS"]
    (a.assets / "SHA256SUMS").write_text("".join(
        hashlib.sha256(x.read_bytes()).hexdigest() + "  " + x.name + "\n" for x in selected))
    print("Release archives and SHA256SUMS created; nothing uploaded.")


if __name__ == "__main__":
    main()
