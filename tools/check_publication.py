#!/usr/bin/env python3
"""Static publication gates. Never prints matched secret contents."""
import ast
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE = {".git", ".runtime", "runtime", "build", "dist", "models", "output", "__pycache__"}


def files():
    for p in ROOT.rglob("*"):
        rel = p.relative_to(ROOT)
        if rel.parts[0] in EXCLUDE or any(x.startswith(".venv") for x in rel.parts):
            continue
        if str(rel) in {"book/data/build.json", "book/data/validation.json"}:
            continue
        if p.is_file():
            yield p


def main():
    issues = []
    selected = list(files())
    for p in selected:
        rel = str(p.relative_to(ROOT))
        if p.is_symlink():
            issues.append((rel, "unexpected source symlink"))
        if p.stat().st_size > 40 * 1024 * 1024:
            issues.append((rel, "large file belongs in release assets"))
        if p.suffix.lower() not in {".py", ".md", ".json", ".txt", ".cc", ".h", ".sh", ".cff", ".in", ".toml", ".yml"}:
            continue
        try:
            text = p.read_text()
        except UnicodeDecodeError:
            issues.append((rel, "unexpected binary in text file"))
            continue
        if p.suffix == ".py":
            try:
                ast.parse(text)
            except SyntaxError as error:
                issues.append((rel, f"syntax error line {error.lineno}"))
        # Upstream source examples are not local workstation paths.
        if not rel.startswith("engine/") and rel != "tools/check_publication.py":
            for pattern, label in [
                (r"/home/(?:vikta|ubuntu)/|/root/(?:\.ssh|\.aws)/", "private workstation path"),
                (r"(?im)^co-authored-by:.*(?:claude|chatgpt|codex)", "AI co-author trailer"),
                (r"(?i)(?:author|prepared by|prepared for).{0,50}(?:claude|chatgpt|codex)", "AI author byline"),
            ]:
                if re.search(pattern, text):
                    issues.append((rel, label))
        if rel == "tools/check_publication.py":
            continue
        for pattern, label in [
            (r"gh[pousr]_[A-Za-z0-9]{25,}", "GitHub credential"),
            (r"-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----", "private key"),
            (r"\bAKIA[A-Z0-9]{16}\b", "AWS access-key identifier"),
            (r"\bsk-(?:proj-)?[A-Za-z0-9_-]{30,}", "API credential"),
        ]:
            if re.search(pattern, text):
                issues.append((rel, label))
    imports = json.loads((ROOT / "provenance/import-manifest.json").read_text())["files"]
    for item in imports:
        rel = item["path"]
        if rel.startswith("engine/src/") or rel in {
            "nextchess/model.py", "nextchess/raw.py", "nextchess/fast_raw.py",
            "nextchess/cloud_data.py", "nextchess/cloud_train.py"}:
            p = ROOT / rel
            if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != item["sha256"]:
                issues.append((rel, "frozen engine/model source differs"))
    # Check the public entry pages, not upstream README links or historical identifiers.
    pages = [ROOT / "README.md", ROOT / "MODEL_CARD.md", ROOT / "LICENSE.md",
             ROOT / "LICENSE-MODEL.md", ROOT / "book/README.md", ROOT / "research/README.md",
             *sorted((ROOT / "docs").glob("*.md"))]
    for p in pages:
        for link in re.findall(r"\]\(([^)]+)\)|src=\"([^\"]+)\"", p.read_text()):
            target = next(x for x in link if x).split("#")[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            if not (p.parent / target).exists():
                issues.append((str(p.relative_to(ROOT)), "missing local link: " + target))
    result = {"passed": not issues, "files_checked": len(selected), "issues": issues}
    print(json.dumps(result, indent=2))
    sys.exit(bool(issues))


if __name__ == "__main__":
    main()
