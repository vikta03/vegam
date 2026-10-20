#!/usr/bin/env python3
"""Download release models and verify their pinned SHA-256 before installation."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--destination", type=Path, default=Path("models"))
    p.add_argument("--trt", action="store_true", help="Also fetch the mixed-precision TensorRT source graph")
    p.add_argument("--ema", action="store_true", help="Also fetch the weights-only PyTorch checkpoint")
    a = p.parse_args()
    manifest = json.loads((Path(__file__).resolve().parents[1] / "models.json").read_text())
    names = ["model.onnx"]
    if a.trt:
        names.append("model.trt-source.onnx")
    if a.ema:
        names.append("vegam-epoch3-ema.pt")
    if any(name not in manifest for name in names):
        p.error("This source snapshot has not completed release preparation; use a tagged release.")
    a.destination.mkdir(parents=True, exist_ok=True)
    for name in names:
        item = manifest[name]
        dst = a.destination / name
        if dst.exists():
            if hashlib.sha256(dst.read_bytes()).hexdigest() != item["sha256"]:
                raise RuntimeError(f"{name} exists but differs; refusing to overwrite it")
            print(f"Verified {name}")
            continue
        temp = dst.with_suffix(dst.suffix + ".download")
        try:
            with urllib.request.urlopen(item["url"], timeout=60) as src, temp.open("wb") as out:
                while block := src.read(1024 * 1024):
                    out.write(block)
            if hashlib.sha256(temp.read_bytes()).hexdigest() != item["sha256"]:
                raise RuntimeError(f"Checksum failed: {name}")
            temp.replace(dst)
            print(f"Downloaded and verified {name}")
        finally:
            temp.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
