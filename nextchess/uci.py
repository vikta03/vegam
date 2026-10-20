"""Synchronous, deadline-bounded UCI driver for reproducible engine receipts."""
from collections import deque
import os
import selectors
import subprocess
import time
import tempfile
import chess

class Engine:
    def __init__(self,command,env=None,timeout=30):
        self.timeout=timeout;self.lines=deque(maxlen=256);self.pending=b''
        self.stderr=tempfile.TemporaryFile()
        self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,env=env)
        self.selector=selectors.DefaultSelector();self.selector.register(self.process.stdout,selectors.EVENT_READ)
        self.send('uci');self.until('uciok');self.send('isready');self.until('readyok')
    def send(self,line):
        self.process.stdin.write((line+'\n').encode());self.process.stdin.flush()
    def until(self,prefix,timeout=None):
        end=time.monotonic()+(self.timeout if timeout is None else timeout);collected=[]
        while True:
            while b'\n' in self.pending:
                raw,self.pending=self.pending.split(b'\n',1);line=raw.decode(errors='replace').strip()
                self.lines.append(line);collected.append(line)
                if line.startswith(prefix):return collected
                if line.startswith('error '):raise RuntimeError(line)
            left=end-time.monotonic()
            if left<=0:raise TimeoutError((prefix,list(self.lines)[-10:]))
            for key,_ in self.selector.select(min(left,.25)):
                chunk=os.read(key.fd,65536)
                if not chunk:
                    self.stderr.seek(0);errors=self.stderr.read().decode(errors='replace')[-4000:]
                    raise RuntimeError(('engine exited',self.process.poll(),list(self.lines)[-10:],errors))
                self.pending+=chunk
    def position(self,board,new_game=True):
        if new_game:self.send('ucinewgame')
        self.send('setoption name UCI_Chess960 value '+str(board.chess960).lower())
        root=board.root()
        command='position fen '+root.shredder_fen(en_passant='fen')
        if board.move_stack:command+=' moves '+' '.join(m.uci() for m in board.move_stack)
        self.send(command)
    def go(self,board,nodes=None,movetime=None,new_game=True,clock=None):
        self.position(board,new_game)
        limit=f'nodes {nodes}' if nodes is not None else f'movetime {movetime}' if movetime is not None else clock
        start=time.monotonic();self.send('go '+limit);lines=self.until('bestmove')
        move=board.parse_uci(lines[-1].split()[1]);assert move in board.legal_moves
        return {'move':move,'pv':[move],'seconds':time.monotonic()-start,'lines':lines}
    def close(self):
        if self.process.poll() is None:
            try:self.send('quit');self.process.wait(timeout=2)
            except (BrokenPipeError,subprocess.TimeoutExpired):self.process.kill();self.process.wait()
        self.selector.close();self.stderr.close()
    def __enter__(self):return self
    def __exit__(self,*_):self.close()
