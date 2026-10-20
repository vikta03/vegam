"""Versioned raw records. All coordinates are side-to-move, a1=0."""
import gzip
import hashlib
from pathlib import Path
import re
import struct
import numpy as np

CHANNELS=120
MAX_MOVES=256
ROOT=Path(__file__).resolve().parents[1]
text=(ROOT/'vendor/lc0/src/neural/encoder.cc').read_text()
MOVE_STRINGS=re.findall(r'"([a-h][1-8][a-h][1-8][qrb]?)"',text.split('const char* kMoveStrs[] = {',1)[1].split('};',1)[0])
assert len(MOVE_STRINGS)==1858
def square(s):return ord(s[0])-97+8*(int(s[1])-1)
MOVE_BASE=np.array([(square(s[:2]),square(s[2:4]),{'q':2,'r':3,'b':4}.get(s[4:],0),0) for s in MOVE_STRINGS],dtype=np.int64)


def dtype(version):
    fields=[('version','<u4'),('format','<u4'),('policy','<f4',(1858,)),('planes','u1',(104,8)),
            ('rights','u1',(4,)),('stm','u1'),('clock','u1'),('flags','u1'),('old_result','i1')]
    for name in ('root_q','best_q','root_d','best_d','root_m','best_m','plies_left'):fields.append((name,'<f4'))
    if version==6:
        for name in ('result_q','result_d','played_q','played_d','played_m','orig_q','orig_d','orig_m'):fields.append((name,'<f4'))
        fields += [('visits','<u4'),('played_idx','<u2'),('best_idx','<u2'),('policy_kld','<f4'),('q_st','<f4')]
    elif version!=5:raise ValueError(f'unsupported record version {version}')
    out=np.dtype(fields);assert out.itemsize=={5:8308,6:8356}[version];return out


class GameRecords(np.ndarray):
    """Sliced frames retain rule identities established by their game prefix."""
    def __array_finalize__(self,obj):
        if obj is not None:
            self.initial_rooks=getattr(obj,'initial_rooks',None)
            self.initial_stm=getattr(obj,'initial_stm',None)
            self.frame_ids=getattr(obj,'frame_ids',None)
            self.total_frames=getattr(obj,'total_frames',None)

    def __getitem__(self,key):
        result=super().__getitem__(key)
        if isinstance(key,str) and isinstance(result,np.ndarray):return np.asarray(result)
        if isinstance(result,GameRecords) and result.dtype.names and result.dtype==self.dtype and getattr(self,'frame_ids',None) is not None:
            result.frame_ids=self.frame_ids[key]
        return result


def records(raw):
    version,fmt=struct.unpack_from('<II',raw)
    dt=dtype(version)
    if len(raw)%dt.itemsize or fmt not in (1,2):raise ValueError('malformed/unsupported record layout')
    arr=np.frombuffer(raw,dt).view(GameRecords)
    arr.frame_ids=np.arange(len(arr));arr.total_frames=len(arr)
    if not ((arr['version']==version).all() and (arr['format']==fmt).all()):raise ValueError('mixed layouts')
    arr.initial_stm=int(arr['stm'][0]);arr.initial_rooks=np.zeros(4,np.int64)
    if fmt==1:
        first=np.unpackbits(arr['planes'][0],axis=1,bitorder='big')
        for side,(kp,rp,rank) in enumerate(((5,3,0),(11,9,56))):
            kings=np.flatnonzero(first[kp]);rooks=np.flatnonzero(first[rp,rank:rank+8])
            if len(kings)!=1:raise ValueError('initial game position lacks one king per side')
            king=int(kings[0])%8
            for castle in range(2):
                j=2*side+castle
                if not arr['rights'][0,j]:continue
                candidates=rooks[rooks<king] if castle==0 else rooks[rooks>king]
                if len(candidates)!=1:raise ValueError('ambiguous rook identity in game prefix')
                arr.initial_rooks[j]=int(candidates[0])
    return arr


def wdl(q,d):
    v=np.stack(((1-d+q)/2,d,(1-d-q)/2),axis=-1).astype(np.float32)
    mask=np.isfinite(v).all(-1)&(v>=-1e-5).all(-1)&(v<=1+1e-5).all(-1)
    v=np.where(mask[...,None],np.maximum(v,0),np.array([0,1,0],np.float32))
    v/=v.sum(-1,keepdims=True)
    return v,mask


def decode(arr):
    n=len(arr);version=int(arr['version'][0]);fmt=int(arr['format'][0])
    # LC0 reverses bits inside each byte on disk. MSB-first unpack restores a1..h8.
    planes=np.unpackbits(arr['planes'],axis=2,bitorder='big').astype(np.float32)
    x=np.zeros((n,64,CHANNELS),np.float32);x[:,:,:104]=planes.transpose(0,2,1)
    if not np.isfinite(arr['policy']).all():raise ValueError('nonfinite policy')
    legal=arr['policy']>=0
    if np.any((arr['policy']<0)&(arr['policy']!=-1)):raise ValueError('unknown illegal sentinel')
    counts=legal.sum(1)
    if np.any((counts<1)|(counts>MAX_MOVES)):raise ValueError('invalid legal move count')
    mass=np.maximum(arr['policy'],0).sum(1)
    if np.any(abs(mass-1)>1e-4):raise ValueError('policy mass is not one')
    initial_rooks=getattr(arr,'initial_rooks',None)
    if fmt==1 and initial_rooks is None:raise ValueError('classical frames require their game-prefix castling context')
    for j,sq in enumerate((0,7,56,63)):
        rights=arr['rights'][:,j]
        if fmt==1:
            if np.any(rights>1):raise ValueError('nonclassical rights in format1')
            # Format1 also occurs in Chess960 games. Recover identities from
            # the initial position, then swap own/opponent when STM changes.
            files=np.where(arr['stm']==arr.initial_stm,initial_rooks[j],initial_rooks[j^2])
            squares=files+(0 if j<2 else 56)
            rook_plane=3 if j<2 else 9
            if np.any((rights!=0)&(planes[np.arange(n),rook_plane,squares]!=1)):
                raise ValueError('ambiguous classical castling identity')
            x[np.arange(n),squares,104+j]=rights
        else:
            for f in range(8):x[:,(0 if j<2 else 56)+f,104+j]=(rights>>f)&1
    if np.any(arr['stm']>1):raise ValueError('invalid stm')
    x[:,:,108]=arr['stm'][:,None];x[:,:,109]=arr['clock'][:,None]/100
    # Recover the previous opponent double push using the retained history.
    old=planes[:,19,:];cur=planes[:,6,:]
    for f in range(8):
        double=(old[:,48+f]==1)&(cur[:,48+f]==0)&(old[:,32+f]==0)&(cur[:,32+f]==1)
        x[:,40+f,110]=double
    for h in range(8):x[:,:,111+h]=(planes[:,h*13:h*13+12,:].sum((1,2))>0)[:,None]
    x[:,:,119]=1
    moves=np.zeros((n,MAX_MOVES,4),np.int64);mask=np.zeros((n,MAX_MOVES),bool)
    target=np.zeros((n,MAX_MOVES),np.float32);action=np.zeros((n,MAX_MOVES,3),np.float32)
    action[:,:,1]=1;action_mask=np.zeros((n,MAX_MOVES),bool)
    best_wdl,best_valid=wdl(arr['best_q'],arr['best_d'])
    if version==6:
        outcome,outcome_valid=wdl(arr['result_q'],arr['result_d'])
        played_wdl,played_valid=wdl(arr['played_q'],arr['played_d'])
    else:
        q=arr['old_result'].astype(np.float32)
        outcome,outcome_valid=wdl(q,(q==0).astype(np.float32))
    for i in range(n):
        ids=np.flatnonzero(legal[i]);m=MOVE_BASE[ids].copy()
        pawn=planes[i,0,m[:,0]]!=0
        m[(m[:,2]==0)&pawn&(m[:,1]>=56),2]=1 # Knight promotion uses base index in LC0.
        castle=(planes[i,5,m[:,0]]!=0)&(planes[i,3,m[:,1]]!=0)
        m[castle,3]=np.where(m[castle,1]>m[castle,0],1,2)
        count=len(ids);moves[i,:count]=m;mask[i,:count]=True;target[i,:count]=arr['policy'][i,ids]/mass[i]
        if version==6:
            for field,vals,valid in (('played_idx',played_wdl,played_valid),('best_idx',best_wdl,best_valid)):
                where=np.flatnonzero(ids==arr[field][i])
                if len(where)!=1:raise ValueError(f'{field} is not a legal policy index')
                if valid[i]:action[i,where[0]]=vals[i];action_mask[i,where[0]]=True
    censored=(arr['flags']&0x30)!=0
    # The writer's plies_left includes its final best_m estimate. That is not
    # an observed survival bound. Use the retained game suffix instead: an
    # adjudicated game survived at least to its last recorded decision, while
    # a completed game's last recorded decision precedes its final move.
    if arr.frame_ids is None:raise ValueError('duration requires game position context')
    duration=(arr.total_frames-arr.frame_ids-censored.astype(np.int64)).astype(np.float32)
    duration_valid=duration>=0
    # UNDECIDED games reaching the cap are encoded as (0,1) by the writer.
    # This sentinel is not an observed draw outcome.
    outcome_valid &= (arr['flags']&0x10)==0
    return dict(features=x,moves=moves,legal=mask,policy=target,state_wdl=best_wdl,
                state_valid=best_valid,outcome=outcome,outcome_valid=outcome_valid,
                action_wdl=action,action_valid=action_mask,duration=np.where(duration_valid,duration,0),
                duration_valid=duration_valid,censored=censored)


def load_game(path,offset,length,expected_hash):
    with open(path,'rb') as f:f.seek(offset);compressed=f.read(length)
    raw=gzip.decompress(compressed)
    if hashlib.sha256(raw).digest()!=expected_hash:raise ValueError('game changed after manifest creation')
    return raw
