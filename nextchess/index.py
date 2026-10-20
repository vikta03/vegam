"""CRC-verified, content-deduplicated tar member index; never extracts games."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import struct
import tarfile
import time

SIZES = {5: 8308, 6: 8356}


def inspect_archive(path):
    before = path.stat()
    with path.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    rows, errors = [], []
    with tarfile.open(path) as tar, path.open('rb') as raw:
        for member in tar:
            if not member.isfile() or not member.name.endswith('.gz'):
                continue
            try:
                if not 18 <= member.size <= 16 * 1024 * 1024:
                    raise ValueError('compressed game outside 18B..16MiB bound')
                raw.seek(member.offset_data)
                compressed = raw.read(member.size)
                # Gzip CRC and length are verified here; an index is never a
                # statement that a footer alone constitutes integrity proof.
                declared = struct.unpack('<I', compressed[-4:])[0]
                if declared > 64 * 1024 * 1024:
                    raise ValueError('uncompressed game exceeds64MiB bound')
                with gzip.GzipFile(fileobj=__import__('io').BytesIO(compressed)) as gf:
                    game = gf.read(64 * 1024 * 1024 + 1)
                if len(game) > 64 * 1024 * 1024:
                    raise ValueError('decompression bound exceeded')
                version, fmt = struct.unpack_from('<II', game)
                size = SIZES.get(version)
                if not size or fmt not in (1, 2) or not game or len(game) % size:
                    raise ValueError(f'unsupported or malformed version={version},format={fmt},bytes={len(game)}')
                # Every record must use this layout, not just the first record.
                import numpy as np
                headers = np.ndarray((len(game)//size, 2), '<u4', game, strides=(size,4))
                if not ((headers[:,0] == version).all() and (headers[:,1] == fmt).all()):
                    raise ValueError('mixed record layouts within game')
                rows.append((member.name,member.offset_data,member.size,version,fmt,
                             len(game)//size,hashlib.sha256(game).digest()))
            except (ValueError, OSError, EOFError, struct.error) as error:
                errors.append((member.name,str(error)))
    after = path.stat()
    if (before.st_size,before.st_mtime_ns) != (after.st_size,after.st_mtime_ns):
        raise RuntimeError(f'archive changed while indexing: {path}')
    return str(path.resolve()),before.st_size,before.st_mtime_ns,digest,rows,errors


def build(root, output, workers=4, limit=0):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'immutable manifest already exists: {output}')
    output.parent.mkdir(parents=True,exist_ok=True)
    partial = output.with_suffix('.building.sqlite')
    db = sqlite3.connect(partial)
    db.executescript('''
    CREATE TABLE IF NOT EXISTS archives(id INTEGER PRIMARY KEY,path TEXT UNIQUE,
      bytes INTEGER,mtime_ns INTEGER,sha256 TEXT,source TEXT,date TEXT);
    CREATE TABLE IF NOT EXISTS games(id INTEGER PRIMARY KEY,archive INTEGER,name TEXT,
      offset INTEGER,bytes INTEGER,version INTEGER,format INTEGER,positions INTEGER,
      sha256 BLOB UNIQUE,split TEXT,order_key TEXT,stratum TEXT);
    CREATE TABLE IF NOT EXISTS duplicates(archive INTEGER,name TEXT,sha256 BLOB);
    CREATE TABLE IF NOT EXISTS rejected(archive INTEGER,name TEXT,reason TEXT);
    CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT);
    ''')
    files = sorted(Path(root).glob('*.tar'))
    if limit: files = files[:limit]
    snapshot = [(str(p.resolve()),p.stat().st_size,p.stat().st_mtime_ns) for p in files]
    stamp = hashlib.sha256(json.dumps(snapshot).encode()).hexdigest()
    previous = db.execute("SELECT value FROM meta WHERE key='snapshot'").fetchone()
    if previous and previous[0] != stamp:
        raise RuntimeError('dataset snapshot changed; use a new output manifest')
    db.execute("INSERT OR IGNORE INTO meta VALUES('snapshot',?)",(stamp,));db.commit()
    done = {r[0] for r in db.execute('SELECT path FROM archives')}
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for path,size,mtime,sha,rows,errors in pool.map(inspect_archive,[p for p in files if str(p.resolve()) not in done],buffersize=workers):
            name = Path(path).name
            match = re.search(r'(.*?)(\d{8})-\d{4}\.tar$',name)
            source,date = match.groups() if match else (name,'unknown')
            aid = db.execute('INSERT INTO archives(path,bytes,mtime_ns,sha256,source,date) VALUES(?,?,?,?,?,?)',
                             (path,size,mtime,sha,source,date)).lastrowid
            for member,offset,csize,ver,fmt,count,gamehash in rows:
                order = hashlib.sha256(b'next-split-v1'+gamehash).hexdigest()
                cur = db.execute('INSERT OR IGNORE INTO games(archive,name,offset,bytes,version,format,positions,sha256,split,order_key,stratum) VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                                 (aid,member,offset,csize,ver,fmt,count,gamehash,'pending',order,source+date))
                if cur.rowcount == 0:db.execute('INSERT INTO duplicates VALUES(?,?,?)',(aid,member,gamehash))
            db.executemany('INSERT INTO rejected VALUES(?,?,?)',[(aid,n,e) for n,e in errors])
            db.commit()
            print(json.dumps(dict(archive=name,games=len(rows),rejected=len(errors),seconds=round(time.monotonic()-started))),flush=True)
    # Rank within source/date strata. Splits never depend on processing order.
    db.executescript('''
    CREATE INDEX IF NOT EXISTS game_strata ON games(stratum,order_key);
    CREATE TEMP TABLE assignments AS
      SELECT id,ROW_NUMBER() OVER(PARTITION BY stratum ORDER BY order_key) AS rn,
      COUNT(*) OVER(PARTITION BY stratum) AS n FROM games;
    CREATE INDEX assignment_ids ON assignments(id);
    UPDATE games SET split=(SELECT CASE WHEN rn<=CAST(n*0.98 AS INTEGER) THEN 'train'
      WHEN rn<=CAST(n*0.99 AS INTEGER) THEN 'valid' ELSE 'test' END
      FROM assignments WHERE assignments.id=games.id);
    CREATE INDEX IF NOT EXISTS games_split ON games(split,id);
    ''')
    summary = dict(snapshot=stamp,archives=len(files),
      games=db.execute('SELECT count(*) FROM games').fetchone()[0],
      positions=db.execute('SELECT sum(positions) FROM games').fetchone()[0],
      duplicates=db.execute('SELECT count(*) FROM duplicates').fetchone()[0],
      rejected=db.execute('SELECT count(*) FROM rejected').fetchone()[0],
      splits=list(db.execute('SELECT split,count(*),sum(positions) FROM games GROUP BY split')),
      integrity='gzip CRC and full decompressed game SHA256 verified; semantic records checked on consumption')
    db.execute("INSERT OR REPLACE INTO meta VALUES('summary',?)",(json.dumps(summary),))
    db.commit();db.close()
    os.replace(partial,output)
    output.with_suffix('.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--out',default='data/corpus.sqlite');p.add_argument('--workers',type=int,default=4)
    p.add_argument('--limit',type=int,default=0);a=p.parse_args();build(a.root,a.out,a.workers,a.limit)
