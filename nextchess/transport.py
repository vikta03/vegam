"""Immutable corpus identity with separately verified, relocatable archive paths."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3


def digest(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def verify(manifest, root, output):
    """Hash every relocated archive once. Never modify the canonical SQLite."""
    manifest, root = Path(manifest).resolve(), Path(root).resolve()
    sha = digest(manifest)
    db = sqlite3.connect(f'file:{manifest}?mode=ro', uri=True)
    entries = {}
    names = set()
    try:
        for aid, original, size, expected in db.execute('SELECT id,path,bytes,sha256 FROM archives ORDER BY id'):
            name = Path(original).name
            if name in names:
                raise ValueError('archive basenames collide; explicit transport layout required')
            names.add(name)
            path = root / name
            before = path.stat()
            if before.st_size != size or digest(path) != expected:
                raise ValueError(f'archive integrity mismatch: {name}')
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError(f'archive changed during verification: {name}')
            entries[str(aid)] = dict(path=str(path), bytes=size, sha256=expected, mtime_ns=after.st_mtime_ns)
    finally:
        db.close()
    if digest(manifest) != sha:
        raise ValueError('manifest changed during transport verification')
    result = dict(schema='next-transport-v1', manifest_sha=sha, archives=entries)
    atomic_json(output, result)
    return result


def load(manifest, receipt):
    if receipt is None:
        return None
    value = json.loads(Path(receipt).read_text())
    if value.get('schema') != 'next-transport-v1' or value.get('manifest_sha') != digest(manifest):
        raise ValueError('transport receipt does not match immutable corpus')
    return value['archives']


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True); p.add_argument('--root', required=True)
    p.add_argument('--out', required=True); a = p.parse_args()
    result = verify(a.manifest, a.root, a.out)
    print(json.dumps(dict(verified_archives=len(result['archives']), manifest_sha=result['manifest_sha'])))
