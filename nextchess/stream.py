"""Bounded, without-replacement frame shuffling with exact restart state."""
from collections import OrderedDict
import hashlib
import sqlite3
from pathlib import Path
import numpy as np
from .raw import records,decode,load_game


class Stream:
    def __init__(self,manifest,split='train',seed=731,buffer_size=4096,cache_games=128):
        self.path=Path(manifest).resolve()
        self.db=sqlite3.connect(f'file:{self.path}?mode=ro',uri=True)
        self.snapshot=self.db.execute("SELECT value FROM meta WHERE key='snapshot'").fetchone()[0]
        self.ids=np.fromiter((r[0] for r in self.db.execute('SELECT id FROM games WHERE split=? ORDER BY id',(split,))),np.int64)
        if not len(self.ids):raise ValueError(f'empty split {split}')
        self.positions=self.db.execute('SELECT sum(positions) FROM games WHERE split=?',(split,)).fetchone()[0]
        self.split=split;self.seed=seed;self.capacity=buffer_size;self.cache_limit=cache_games
        self.cache=OrderedDict();self.epoch=0;self.cursor=0;self.buffer=[];self.exposed=0;self.rng=np.random.default_rng(seed)
        self._order()

    def _order(self):
        self.order=np.random.default_rng(self.seed+self.epoch).permutation(self.ids)

    def game(self,gid):
        if gid in self.cache:
            self.cache.move_to_end(gid);return self.cache[gid]
        row=self.db.execute('SELECT a.path,g.offset,g.bytes,g.sha256,a.bytes,a.mtime_ns FROM games g JOIN archives a ON g.archive=a.id WHERE g.id=?',(int(gid),)).fetchone()
        stat=Path(row[0]).stat()
        if (stat.st_size,stat.st_mtime_ns)!=(row[4],row[5]):raise ValueError('archive changed after indexing')
        arr=records(load_game(*row[:4]));self.cache[gid]=arr
        while len(self.cache)>self.cache_limit:self.cache.popitem(last=False)
        return arr

    def state_dict(self):
        return dict(snapshot=self.snapshot,split=self.split,seed=self.seed,capacity=self.capacity,
                    epoch=self.epoch,cursor=self.cursor,buffer=list(self.buffer),exposed=self.exposed,rng=self.rng.bit_generator.state)

    def load_state_dict(self,state):
        for k in ('snapshot','split','seed','capacity'):
            if state[k]!=getattr(self,k):raise ValueError(f'resume {k} differs')
        for k in ('epoch','cursor','exposed'):setattr(self,k,state[k])
        self.buffer=list(state['buffer']);self.rng.bit_generator.state=state['rng'];self._order()

    def batch(self,size):
        # A batch never crosses an epoch boundary. No drop_last and no hidden
        # padding/repetition of the last training examples.
        refs=[]
        while len(refs)<size:
            while len(self.buffer)<self.capacity and self.cursor<len(self.order):
                gid=int(self.order[self.cursor]);self.cursor+=1
                self.buffer.extend((gid,j) for j in range(len(self.game(gid))))
            if not self.buffer:break
            pick=int(self.rng.integers(len(self.buffer)));refs.append(self.buffer[pick])
            self.buffer[pick]=self.buffer[-1];self.buffer.pop()
        if not refs:
            self.epoch+=1;self.cursor=0;self._order();return None
        groups={}
        for dst,(gid,frame) in enumerate(refs):groups.setdefault(gid,[]).append((dst,frame))
        result=None
        for gid,entries in groups.items():
            dst,frame=map(np.array,zip(*entries));part=decode(self.game(gid)[frame])
            if result is None:result={k:np.empty((len(refs),*v.shape[1:]),v.dtype) for k,v in part.items()}
            for key in result:result[key][dst]=part[key]
        self.exposed+=len(refs)
        result['record_ids']=np.asarray(refs,np.int64)
        return result
