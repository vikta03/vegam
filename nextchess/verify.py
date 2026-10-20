"""Cross-language feature/export and native UCI regression checks."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time
import chess
import chess.engine
import numpy as np
import torch
import onnxruntime as ort
from .model import Config,Net
from .raw import decode,records,load_game
from .uci import Engine

PROMOTIONS={0:None,1:chess.KNIGHT,2:chess.QUEEN,3:chess.ROOK,4:chess.BISHOP}

def board_from_features(x):
    b=chess.Board(None,chess960=True)
    for c in range(12):
        for sq in np.flatnonzero(x[:,c]):b.set_piece_at(int(sq),chess.Piece(c%6+1,c<6))
    b.turn=chess.WHITE;b.halfmove_clock=round(float(x[0,109])*100)
    b.castling_rights=sum(chess.BB_SQUARES[int(sq)] for sq in np.flatnonzero(x[:,104:108].sum(1)))
    ep=np.flatnonzero(x[:,110]);b.ep_square=int(ep[0]) if len(ep) else None
    if x[0,108]:b=b.mirror()
    return b

def action_tuple(b,m):
    flip=0 if b.turn else 56
    castle=1 if b.is_kingside_castling(m) else 2 if b.is_queenside_castling(m) else 0
    return (m.from_square^flip,m.to_square^flip,{v:k for k,v in PROMOTIONS.items()}[m.promotion],castle)

def read_dump(path):
    with path.open('rb') as f:
        x=np.fromfile(f,np.float32,64*120).reshape(64,120)
        moves=np.fromfile(f,np.int64,256*4).reshape(256,4)
        mask=np.fromfile(f,np.bool_,256)
    return x,moves[mask]

def export_parity(checkpoint,model,manifest):
    torch.set_num_threads(2);ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    net=Net(Config(**ck['config']));net.load_state_dict(ck['ema']);net.eval()
    session=ort.InferenceSession(str(model),providers=['CPUExecutionProvider'])
    db=sqlite3.connect(f'file:{manifest}?mode=ro',uri=True);row=db.execute(
        'SELECT a.path,g.offset,g.bytes,g.sha256 FROM games g JOIN archives a ON a.id=g.archive WHERE g.split="valid" ORDER BY g.order_key LIMIT 1').fetchone()
    data=decode(records(load_game(*row))[:17]);inputs={k:data[k] for k in ('features','moves','legal')}
    with torch.no_grad():want=net(*(torch.from_numpy(inputs[k]) for k in inputs))
    got=session.run(None,inputs);errors=[]
    for a,b in zip(want,got):
        a=a.numpy();np.testing.assert_allclose(a,b,atol=3e-5,rtol=3e-5);errors.append(float(np.max(np.abs(a-b))))
    return dict(batch=17,max_absolute_errors=errors)

def native_checks(binary,model,manifest,all960=False):
    db=sqlite3.connect(f'file:{manifest}?mode=ro',uri=True)
    summary=dict(history_positions=0,layouts960=0,special_castling=0,cache_history_variants=0,native_value_max_error=0.)
    opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
    direct=ort.InferenceSession(str(model),sess_options=opts,providers=['CPUExecutionProvider'])
    with tempfile.TemporaryDirectory(prefix='next-contract-') as directory:
        env={**os.environ,'NEXT_PROVIDER':'cpu','NEXT_DUMP_INPUTS':directory}
        command=[str(Path(binary).resolve()),'--weights='+str(Path(model).resolve()),'--threads=1','--minibatch-size=1','--nncache=0','--move-overhead=0']
        with Engine(command,env=env,timeout=15) as engine:
            def probe(board,expected=None):
                prior=set(Path(directory).glob('*.bin'))
                try:info=engine.go(board,nodes=1)
                except Exception as e:raise RuntimeError(('native contract failed',board.fen(),summary)) from e
                made=set(Path(directory).glob('*.bin'))-prior
                if not made:raise AssertionError('native root not evaluated')
                x,m=read_dump(min(made,key=lambda p:int(p.stem)))
                path=min(made,key=lambda p:int(p.stem))
                values=np.fromfile(path.with_suffix('.values'),np.float32)
                moves=np.zeros((1,256,4),np.int64);moves[0,:len(m)]=m
                legal=np.arange(256)[None,:]<len(m)
                outputs=direct.run(None,{'features':x[None,:,:],'moves':moves,'legal':legal})
                def softmax(t):
                    exp=np.exp(t-t.max(-1,keepdims=True));return exp/exp.sum(-1,keepdims=True)
                state=softmax(outputs[1])[0];actions=softmax(outputs[2])[0,:len(m)]
                want=np.concatenate(([state[0]-state[2],state[1],0.],softmax(outputs[0])[0,:len(m)],actions[:,0]-actions[:,2],actions[:,1]))
                np.testing.assert_allclose(values,want,atol=3e-5,rtol=3e-5)
                summary['native_value_max_error']=max(summary['native_value_max_error'],float(abs(values-want).max()))
                actual=set(map(tuple,m));legal={action_tuple(board,move) for move in board.legal_moves}
                if actual!=legal:raise AssertionError((board.fen(),actual^legal))
                if expected is not None:
                    if not np.array_equal(x,expected):
                        diff=np.argwhere(x!=expected)
                        raise AssertionError((board.fen(),diff[:20].tolist(),int(len(diff))))
                else:
                    reconstructed=board_from_features(x)
                    assert reconstructed.board_fen()==board.board_fen()
                    assert reconstructed.turn==board.turn
                    assert reconstructed.castling_rights==board.castling_rights
                    assert reconstructed.halfmove_clock==board.halfmove_clock
                assert info['pv'][0] in board.legal_moves
                for path in made:path.unlink()
                return info
            rows=db.execute('SELECT a.path,g.offset,g.bytes,g.sha256 FROM games g JOIN archives a ON a.id=g.archive ORDER BY g.order_key LIMIT 12')
            for row in rows:
                arr=records(load_game(*row))
                if int(arr['version'][0])!=6:continue
                batch=decode(arr);board=board_from_features(batch['features'][0])
                for i in range(min(24,len(arr))):
                    if i in (0,1,7,15,23):probe(board,batch['features'][i]);summary['history_positions']+=1
                    ids=np.flatnonzero(arr['policy'][i]>=0);j=int(np.flatnonzero(ids==arr['played_idx'][i])[0])
                    a=batch['moves'][i,j];flip=0 if board.turn else 56
                    move=chess.Move(int(a[0])^flip,int(a[1])^flip,PROMOTIONS[int(a[2])])
                    assert move in board.legal_moves;board.push(move)
            if all960:
                for index in range(960):
                    board=chess.Board.from_chess960_pos(index);probe(board);summary['layouts960']+=1
                    # Both directions, stationary king/rook and crossing paths.
                    for sq,piece in list(board.piece_map().items()):
                        if piece.color==chess.BLACK or piece.piece_type not in (chess.KING,chess.ROOK):board.remove_piece_at(sq)
                    board.castling_rights &= chess.BB_RANK_1
                    free=next(f for f in range(8) if not board.piece_at(f) or board.piece_at(f).piece_type!=chess.ROOK)
                    board.set_piece_at(56+free,chess.Piece(chess.KING,chess.BLACK))
                    assert board.is_valid();probe(board);summary['special_castling']+=1
            board=chess.Board('3rr1k1/1pp2ppp/pbn3q1/3Pp3/1PQ1P3/P1N1BN1b/3R1PP1/2R3K1 w - - 4 25')
            safe=set()
            for move in board.legal_moves:
                child=board.copy();child.push(move);loses=False
                for reply in child.legal_moves:
                    child.push(reply);mate=child.is_checkmate();child.pop()
                    if mate:loses=True;break
                if not loses:safe.add(move)
            assert len(safe)==8
            move=engine.go(board,nodes=1)['move']
            assert move in safe,move
            summary['mate25_safe_moves']=len(safe)
            repeated=chess.Board()
            for move in ('g1f3','g8f6','f3g1','f6g8'):repeated.push_uci(move)
            for board in (repeated,chess.Board(repeated.fen()),chess.Board(),repeated):
                probe(board);summary['cache_history_variants']+=1
            mate=chess.Board('7k/8/5KQ1/8/8/8/8/8 w - - 0 1')
            reply=engine.go(mate,nodes=1)['move']
            mate.push(reply);assert mate.is_checkmate();summary['mate_in_one']=True
            start=time.monotonic()
            engine.position(chess.Board());engine.send('go infinite')
            time.sleep(.05);engine.send('stop');engine.until('bestmove',timeout=2)
            summary['stop_seconds']=time.monotonic()-start
            assert summary['stop_seconds']<2
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binary',default='build/lc0/lc0');p.add_argument('--model',default='build/smoke.onnx')
    p.add_argument('--checkpoint',default='runs/smoke/latest.pt');p.add_argument('--manifest',default='data/corpus.sqlite')
    p.add_argument('--all960',action='store_true');p.add_argument('--out',default='runs/verification.json');a=p.parse_args()
    receipt={'export':export_parity(a.checkpoint,a.model,a.manifest),'native':native_checks(a.binary,a.model,a.manifest,a.all960)}
    Path(a.out).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
