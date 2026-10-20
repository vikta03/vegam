"""Parallel decoding with commit-only coverage and deterministic DDP sharding."""
from collections import OrderedDict, deque
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import multiprocessing as mp
from pathlib import Path
import sqlite3
import numpy as np
from .raw import records, load_game
from .fast_raw import decode
from .transport import load as load_transport


class References:
    def __init__(self, manifest, split='train', seed=731, capacity=4096):
        if capacity < 1:
            raise ValueError('shuffle capacity must be positive')
        db = sqlite3.connect(f'file:{Path(manifest).resolve()}?mode=ro', uri=True)
        try:
            self.snapshot = db.execute("SELECT value FROM meta WHERE key='snapshot'").fetchone()[0]
            pairs = np.array(db.execute('SELECT id,positions FROM games WHERE split=? ORDER BY id', (split,)).fetchall(), dtype=np.int64)
        finally:
            db.close()
        if not len(pairs):
            raise ValueError('empty partition')
        self.ids, self.counts = pairs[:, 0], pairs[:, 1]
        self.positions = int(self.counts.sum())
        self.split, self.seed, self.capacity = split, seed, capacity
        self.epoch = self.cursor = self.exposed = 0
        self.buffer = []
        self.rng = np.random.default_rng(seed)
        self._order()

    def _order(self):
        # Same permutation and swap-remove shuffle as the pilot reader.
        self.order = np.random.default_rng(self.seed + self.epoch).permutation(len(self.ids))

    def state_dict(self):
        # References are immutable integer tuples; only the containing list and
        # RNG dictionaries need independent ownership for queued snapshots.
        return dict(snapshot=self.snapshot, split=self.split, seed=self.seed,
                    capacity=self.capacity, epoch=self.epoch, cursor=self.cursor,
                    buffer=list(self.buffer), exposed=self.exposed,
                    rng=deepcopy(self.rng.bit_generator.state))

    def load_state_dict(self, state):
        for name in ['snapshot', 'split', 'seed', 'capacity']:
            if state[name] != getattr(self, name):
                raise ValueError(f'reader {name} differs')
        for name in ['epoch', 'cursor', 'exposed']:
            setattr(self, name, state[name])
        self.buffer = list(state['buffer'])
        self.rng.bit_generator.state = deepcopy(state['rng'])
        self._order()

    def take(self, size):
        if size < 1:
            raise ValueError('batch size must be positive')
        refs = []
        while len(refs) < size:
            while len(self.buffer) < self.capacity and self.cursor < len(self.order):
                at = self.order[self.cursor]
                self.cursor += 1
                self.buffer.extend((int(self.ids[at]), j) for j in range(int(self.counts[at])))
            if not self.buffer:
                break
            pick = int(self.rng.integers(len(self.buffer)))
            refs.append(self.buffer[pick])
            self.buffer[pick] = self.buffer[-1]; self.buffer.pop()
        if not refs:
            self.epoch += 1; self.cursor = 0; self._order()
            return self.take(size)
        self.exposed += len(refs)
        return np.asarray(refs, np.int64)


class Games:
    def __init__(self, manifest, transport=None, cache_games=128):
        self.db = sqlite3.connect(f'file:{Path(manifest).resolve()}?mode=ro', uri=True)
        self.transport = transport
        self.cache = OrderedDict()
        self.limit = cache_games

    def get(self, gid):
        if gid in self.cache:
            self.cache.move_to_end(gid)
            return self.cache[gid]
        row = self.db.execute('SELECT a.id,a.path,a.bytes,a.mtime_ns,a.sha256,g.offset,g.bytes,g.sha256 '
                              'FROM games g JOIN archives a ON a.id=g.archive WHERE g.id=?', (int(gid),)).fetchone()
        if row is None:
            raise ValueError('unknown game id')
        aid, path, size, mtime, sha, offset, length, game_sha = row
        if self.transport is not None:
            entry = self.transport[str(aid)]
            if entry['sha256'] != sha or entry['bytes'] != size:
                raise ValueError('transport archive identity differs')
            path, mtime = entry['path'], entry['mtime_ns']
        stat = Path(path).stat()
        if (stat.st_size, stat.st_mtime_ns) != (size, mtime):
            raise ValueError('archive changed after verification')
        arr = records(load_game(path, offset, length, game_sha))
        self.cache[gid] = arr
        while len(self.cache) > self.limit:
            self.cache.popitem(last=False)
        return arr

    def decode(self, refs):
        groups = {}
        for dest, (gid, frame) in enumerate(refs):
            groups.setdefault(int(gid), []).append((dest, int(frame)))
        by_version = {}
        for gid, items in groups.items():
            game = self.get(gid)
            dst, frames = map(np.array, zip(*items))
            by_version.setdefault(int(game['version'][0]), []).append((game, dst, frames))
        result = None
        for items in by_version.values():
            arr = np.concatenate([np.asarray(game[frames]) for game, _, frames in items])
            ctx = dict(
                initial_rooks=np.concatenate([np.broadcast_to(g.initial_rooks, (len(f), 4)) for g, _, f in items]),
                initial_stm=np.concatenate([np.full(len(f), g.initial_stm) for g, _, f in items]),
                frame_ids=np.concatenate([g.frame_ids[f] for g, _, f in items]),
                total_frames=np.concatenate([np.full(len(f), g.total_frames) for g, _, f in items]))
            dst = np.concatenate([d for _, d, _ in items])
            part = decode(arr, ctx)
            if result is None:
                result = {k: np.empty((len(refs), *v.shape[1:]), v.dtype) for k, v in part.items()}
            for k in part:
                result[k][dst] = part[k]
        result['record_ids'] = refs.copy()
        return result


_games = None


def _initialize(manifest, transport, cache):
    global _games
    _games = Games(manifest, transport, cache)


def _decode(refs):
    return _games.decode(refs)


class ParallelStream:
    """Prefetch may run ahead; only commit() advances checkpoint coverage."""
    def __init__(self, manifest, *, split='train', seed=731, capacity=4096,
                 batch=1024, rank=0, world=1, workers=2, prefetch=2, transport=None):
        if not 0 <= rank < world or workers < 0 or prefetch < 1:
            raise ValueError('invalid worker/rank configuration')
        self.source = References(manifest, split, seed, capacity)
        self.positions = self.source.positions
        self.batch, self.rank, self.world = batch, rank, world
        self.committed = self.source.state_dict()
        mapping = load_transport(manifest, transport)
        self.pool = ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn'),
                                       initializer=_initialize, initargs=(str(manifest), mapping, 128)) if workers else None
        self.games = None if workers else Games(manifest, mapping)
        self.prefetch = prefetch if workers else 1
        self.queue = deque()
        self.pending = None

    def _fill(self):
        while len(self.queue) < self.prefetch:
            refs = self.source.take(self.batch)
            local = refs[self.rank::self.world]
            # Empty ranks still run one zero-weight item so all collectives and
            # parameter gradients participate in an uneven final batch.
            empty = not len(local)
            if empty:
                local = refs[:1]
            job = self.pool.submit(_decode, local) if self.pool else self.games.decode(local)
            self.queue.append((job, self.source.state_dict(), len(refs), empty))

    def peek(self):
        if self.pending is None:
            self._fill()
            job, state, count, empty = self.queue.popleft()
            batch = job.result() if self.pool else job
            batch['sample_valid'] = np.full(len(batch['features']), not empty, bool)
            self.pending = (batch, state, count)
        return self.pending[0], self.pending[2]

    def commit(self):
        if self.pending is None:
            raise RuntimeError('no batch to commit')
        self.committed = self.pending[1]
        self.pending = None

    def state_dict(self):
        state = dict(self.committed)
        state['buffer'] = list(self.committed['buffer'])
        state['rng'] = deepcopy(self.committed['rng'])
        return state

    def load_state_dict(self, state):
        if self.queue or self.pending is not None:
            raise RuntimeError('restore before requesting any batches')
        self.source.load_state_dict(state)
        self.committed = self.source.state_dict()

    def close(self):
        if self.pool:
            self.pool.shutdown(wait=True, cancel_futures=True)
        else:
            self.games.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
