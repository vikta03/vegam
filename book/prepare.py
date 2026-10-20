"""Read saved measurements; derive arithmetic and figures. No inference or training."""
import os
os.environ['MPLBACKEND']='Agg'
os.environ['MPLCONFIGDIR']='/tmp/vegam-book-mpl'
os.environ['OMP_NUM_THREADS']='2'
import sys, json, math, hashlib
from pathlib import Path
import numpy as np
import torch
import chess, chess.pgn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyArrowPatch
from matplotlib.font_manager import FontProperties
from visual_style import board, use_style, ensure_piece_assets, polish_chart, label_bars, SEQUENTIAL, DIVERGING
ROOT=Path(__file__).resolve().parent
WS=ROOT.parent
NEXT=WS
ATLAS=ROOT/'inputs'
RUN=ATLAS
F=ROOT/'figures';D=ROOT/'data'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','figure.facecolor':'white','savefig.facecolor':'white'})
INK='#173042'; TEAL='#087E8B'; GOLD='#BC812A'; RED='#AF514C'; LIGHT='#E4EBE8'
use_style(); ensure_piece_assets()
r=json.loads((RUN/'result.json').read_text()); t=np.load(RUN/'trace.npz'); inp=np.load(ATLAS/'inputs.npz')
positions=json.loads((ATLAS/'positions.json').read_text()); record=next(p for p in positions if p['id']==r['position_id'])
b=chess.Board(record['start_fen']);history=[b.copy()]
for m in record['moves']:
    move=chess.Move.from_uci(m);assert move in b.legal_moves;b.push(move);history.append(b.copy())
assert b.fen()==r['fen'];assert len(record['moves'])==160
with (D/'catalan.pgn').open() as f:game=chess.pgn.read_game(f)
assert not game.errors
allboards=[game.board()]; sanctions=[]
for node in game.mainline():
    sanctions.append(node.san());allboards.append(node.board())
assert allboards[-1].is_checkmate(); assert len(sanctions)==201
weights_path=WS/'models/vegam-epoch3-ema.pt'
assert hashlib.sha256(weights_path.read_bytes()).hexdigest()==json.loads((WS/'models.json').read_text())['vegam-epoch3-ema.pt']['sha256']
torch.set_num_threads(2)
ck=torch.load(weights_path,map_location='cpu',weights_only=True)
w={k:v.detach().half().float().numpy() for k,v in ck['ema'].items()};params=sum(v.numel() for v in ck['ema'].values());del ck
# Reconstruct two front-end coordinates directly from saved input and weights.
x=inp['features'];emb=x@w['input.weight'].T+w['input.bias']+w['position'][0]
assert np.max(abs(emb-t['embedded']))<3e-6
nonzero={chess.square_name(s):[[int(i),float(x[s,i])] for i in np.flatnonzero(x[s])] for s in [36,54,55,47]}
projection=[]
for j in range(3):
    contributions=x[54]*w['input.weight'][j]
    projection.append({'dim':j,'terms':[[int(i),float(x[54,i]),float(w['input.weight'][j,i]),float(contributions[i])] for i in np.flatnonzero(x[54])], 'sum':float(contributions.sum()),'bias':float(w['input.bias'][j]),'position':float(w['position'][0,54,j]),'embedded':float(t['embedded'][54,j])})
# Complete hand-computable 3-square, one-head lesson; these are teaching data.
Z=np.array([[1.,0.],[0.,1.],[1.,1.]])
Wq=np.array([[1.,0.],[1.,1.]]);Wk=np.array([[1.,1.],[0.,1.]]);Wv=np.array([[1.,2.],[-1.,1.]])
Q=Z@Wq;K=Z@Wk;V=Z@Wv
C=Q@K.T/math.sqrt(2)
S=np.array([[0,.2,-.1],[.1,0,.2],[-.2,.1,0.]])
left=np.array([1.,.5,-.5]);right=np.array([.2,-.4,.6]);G=.5
adaptive=G*np.outer(left,right);L=C+S+adaptive
A=np.exp(L-L.max(-1,keepdims=True));A/=A.sum(-1,keepdims=True);Y=A@V
Wo=np.array([[.5,0.],[.25,1.]])
U=Y@Wo;R=Z+U
lnmean=R[0].mean();lnvar=((R[0]-lnmean)**2).mean();ln=(R[0]-lnmean)/math.sqrt(lnvar+1e-5)
W1=np.array([[1.,-1.,.5],[.5,1.,-1.]]);b1=np.array([.1,0.,-.1]);W2=np.array([[.5,0.],[0.,.5],[.25,-.25]])
gelu=lambda x:.5*x*(1+np.vectorize(math.erf)(x/math.sqrt(2)))
ffpre=ln@W1+b1;ffact=gelu(ffpre);ffupdate=ffact@W2;final=R[0]+ffupdate
p=np.array([.2,.5,.3]);target=np.array([.7,.2,.1]);ce=-sum(target*np.log(p));grad=p-target;logits=np.log(p);newlogits=logits-.1*grad;newp=np.exp(newlogits);newp/=newp.sum()
# Exact local chain rule through toy attention and finite-difference verification.
row=0;dY=np.array([1.,0.]);dA=V@dY;dL=A[row]*(dA-A[row]@dA);dQ=dL@K/math.sqrt(2);dK=dL[:,None]*Q[row][None,:]/math.sqrt(2)
toy={'Z':Z.tolist(),'Wq':Wq.tolist(),'Wk':Wk.tolist(),'Wv':Wv.tolist(),'Q':Q.tolist(),'K':K.tolist(),'V':V.tolist(),'C':C.tolist(),'S':S.tolist(),'adaptive':adaptive.tolist(),'L':L.tolist(),'A':A.tolist(),'Y':Y.tolist(),'Wo':Wo.tolist(),'U':U.tolist(),'R':R.tolist(),'ln':ln.tolist(),'ffpre':ffpre.tolist(),'ffact':ffact.tolist(),'ffupdate':ffupdate.tolist(),'final':final.tolist(),'training':{'p':p.tolist(),'target':target.tolist(),'ce':ce,'grad':grad.tolist(),'newp':newp.tolist(),'newce':float(-sum(target*np.log(newp)))},'chain':{'dA':dA.tolist(),'dL':dL.tolist(),'dQ':dQ.tolist(),'dK':dK.tolist()}}
checks=[]
for k in range(2):
    eps=1e-5;q=Q[row].copy();q[k]+=eps
    def f(q):
        l=q@K.T/math.sqrt(2)+S[row]+adaptive[row];a=np.exp(l-l.max());a/=a.sum();return a@V[:,0]
    qm=Q[row].copy();qm[k]-=eps
    num=(f(q)-f(qm))/(2*eps);checks.append(abs(num-dQ[k]));assert abs(num-dQ[k])<1e-8
post=b.copy();post.push_san('Bg7');rook_moves=[m for m in post.legal_moves if m.from_square==chess.H7]
assert {m.uci() for m in rook_moves}=={'h7h8','h7g7','h7h6'}
for san,reply in [('Rh8','Bxh8'),('Rxg7','hxg7'),('Rxh6','Bxh6')]:
    bb=post.copy();bb.push_san(san);bb.push_san(reply)
projmsg=(t['block.0.attention'][5,54,:,None]*t['block.0.value'][5])@w['blocks.0.out.weight'][:,5*32:6*32].T
components={k:t['block.0.'+k][5,54].tolist() for k in ['content_scores','static_bias','adaptive_bias','attention']}
components['projected_message_norm']=np.linalg.norm(projmsg,axis=1).tolist()
facts={'position':record,'model_parameters':params,'nonzero':nonzero,'projection_g7':projection,'toy':toy,'real_components':components,'real_q_g7':t['block.0.query'][5,54].tolist(),'real_k_h7':t['block.0.key'][5,55].tolist(),'real_dot':float(t['block.0.query'][5,54]@t['block.0.key'][5,55]),'decoder_first':{k:t[k][13,:6].tolist() for k in ['decoder.source','decoder.destination','decoder.interaction','decoder.combined','move_hidden']},'history_fens':[bb.fen() for bb in history[-8:][::-1]],'figure_manifest':{},'validation':{'input_projection_max_error':float(abs(emb-t['embedded']).max()),'toy_attention_rowsum_max_error':float(abs(A.sum(-1)-1).max()),'toy_gradient_error':max(checks),'legal_pgn_plies':len(sanctions),'rook_exits_verified':3}}
# Chess artwork is rendered from explicit, rule-accurate SVG paths.
def save(name,fig,kind='MEASURED'):
    fig.savefig(F/(name+'.png'),dpi=300,bbox_inches='tight',pad_inches=.14);plt.close(fig);facts['figure_manifest'][name]={'kind':kind}
def panels(name,boards,titles,arrows=None):
    fig,axs=plt.subplots(1,len(boards),figsize=(7.3,3.85 if len(boards)==2 else 3.5),squeeze=False)
    for i,(bo,title) in enumerate(zip(boards,titles)):board(axs[0,i],bo,title,arrows=arrows[i] if arrows else ())
    fig.tight_layout();save(name,fig,'RULES / GAME')
panels('position',[b,post],['Before 81.Bg7  /  White to move','After 81.Bg7  /  Black to move'],[[('e5','g7',TEAL)],[('g7','h8',TEAL),('g7','h6',TEAL),('h6','g7',GOLD)]])
for san,reply,key in [('Rh8','Bxh8','exit_h8'),('Rxg7','hxg7','exit_g7'),('Rxh6','Bxh6','exit_h6')]:
    bb=post.copy();bb.push_san(san);bb2=bb.copy();bb2.push_san(reply);panels(key,[bb,bb2],[f'81...{san}',f'82.{reply}'],[[('h7',bb.peek().to_square,RED)],[(bb2.peek().from_square,bb2.peek().to_square,TEAL)]])
panels('game_arc',[allboards[134],allboards[160]],['Before 68.Kd1','Before 81.Bg7'],None)
panels('conversion',[allboards[170],allboards[187]],['After 85.Kb4: king enters','After 94.a8=Q: promotion'],None)
fig,axs=plt.subplots(1,2,figsize=(7.3,3.5));board(axs[0],b,'g7 is empty; h6 holds a pawn',arrows=[('e5','g7',TEAL)])
nz=np.flatnonzero(x[54]);axs[1].barh(np.arange(len(nz)),x[54,nz],color=TEAL);axs[1].set_yticks(np.arange(len(nz)),[f'channel {i}' for i in nz]);axs[1].invert_yaxis();axs[1].set_xlim(0,1.15);axs[1].set_title('Exact nonzero g7 inputs');axs[1].set_xlabel('Feature value');polish_chart(axs[1]);label_bars(axs[1],axs[1].containers[0],fmt='%.2g');fig.tight_layout();save('board_input',fig)
fig,axs=plt.subplots(1,2,figsize=(7.3,3.5));board(axs[0],b,'Same board, four square rows');im=axs[1].imshow(x[[36,54,55,47]],aspect='auto',interpolation='nearest',cmap=SEQUENTIAL,vmin=0,vmax=1);axs[1].set_yticks(range(4),['e5 bishop','g7 empty','h7 rook','h6 pawn']);axs[1].set_xticks([0,13,52,91,104,119]);axs[1].set_xlabel('Input channel (0-119)');axs[1].set_title('Four rows from 64 x 120');fig.tight_layout();save('feature_rows',fig)
fig,axs=plt.subplots(1,3,figsize=(7.4,2.75))
for ax,k,ttl in zip(axs,['embedded','block.0.residual','block.7.residual'],['After input projection','After block 1','After block 8']):
    ax.plot(t[k][54],lw=.6,color=TEAL);ax.set_title(ttl,fontsize=10);ax.set_xlabel('g7 coordinate (0-383)');ax.set_ylim(-28,12)
axs[0].set_ylabel('Signed activation');fig.tight_layout();save('g7_vectors',fig)
fig,ax=plt.subplots(figsize=(7.2,3.35));
for sq,col in [(36,TEAL),(54,GOLD),(55,RED),(47,'#646BB1')]:
    ax.plot(range(9),[np.linalg.norm(t['embedded'][sq])]+[np.linalg.norm(t[f'block.{i}.residual'][sq]) for i in range(8)],marker='o',ms=4,label=chess.square_name(sq),color=col)
ax.set_xticks(range(9),['Input']+list(map(str,range(1,9))));ax.set_xlabel('Transformer block completed');ax.set_ylabel('Residual L2 norm');ax.legend(ncol=4);ax.set_title('Representations change; norm is not confidence');fig.tight_layout();save('norms',fig)
for name,key,ttl in [('content','content_scores','Content score into g7'),('static','static_bias','Fixed bias into g7'),('adaptive','adaptive_bias','Adaptive bias into g7')]:
    fig,axs=plt.subplots(1,2,figsize=(7.3,3.45));board(axs[0],b,'Block 1, head 6');vals=t['block.0.'+key][5,54];vm=max(abs(vals.min()),abs(vals.max()));board(axs[1],b,ttl,vals,'RdBu_r',-vm,vm);fig.tight_layout();save(name,fig)
fig,axs=plt.subplots(1,2,figsize=(7.3,3.5));board(axs[0],b,'Attention into g7',t['block.0.attention'][5,54],vmin=0);board(axs[1],b,'Projected message magnitude',np.linalg.norm(projmsg,axis=1),vmin=0);fig.tight_layout();save('attention_message',fig)
fig,ax=plt.subplots(figsize=(7.2,3.55));im=ax.imshow(t['block.0.attention'][5],cmap=SEQUENTIAL,aspect='auto');ax.set_xticks([0,7,28,54,63],['a1','h1','e4','g7','h8']);ax.set_yticks([0,7,28,54,63],['a1','h1','e4','g7','h8']);ax.set_xlabel('Supplying square j');ax.set_ylabel('Receiving square i');ax.axhline(54,color=GOLD,lw=1);ax.set_title('64 x 64 attention matrix: block 1, head 6');fig.colorbar(im,ax=ax,label='Row-normalized weight');fig.tight_layout();save('matrix64',fig)
fig,ax=plt.subplots(figsize=(7.2,3.2));im=ax.imshow(t['block.0.gates'],vmin=-1,vmax=1,cmap=DIVERGING,aspect='auto');ax.set_yticks(range(12),range(1,13));ax.set_xticks(range(0,16,2),range(0,16,2));ax.set_ylabel('Head (1-12)');ax.set_xlabel('Rank component (0-15)');ax.set_title('Block 1: signed gates for 16 relation patterns');fig.colorbar(im,ax=ax,label='Gate coefficient');fig.tight_layout();save('gates',fig)
fig,axs=plt.subplots(1,2,figsize=(7.3,3.5));board(axs[0],b,'MLP block 1, unit 360',t['block.0.ff_act'][:,360],vmin=-.17);board(axs[1],b,'That unit is high on many squares');fig.tight_layout();save('neuron',fig)
fig,axs=plt.subplots(1,2,figsize=(7.2,2.75));xx=np.linspace(-3,3,301);axs[0].plot(xx,gelu(xx),color=TEAL,label='GELU');axs[0].plot(xx,np.maximum(xx,0),ls='--',color=GOLD,label='ReLU');axs[0].axhline(0,color='gray',lw=.4);axs[0].legend();axs[0].set_xlabel('Pre-activation');axs[0].set_ylabel('Output');axs[0].set_title('Smooth feature gating');axs[1].plot(xx,np.tanh(xx),color=TEAL);axs[1].set_title('Signed adaptive gate: tanh');axs[1].set_xlabel('Gate pre-activation');axs[1].set_ylabel('Gate output');fig.tight_layout();save('nonlinear',fig,'MATHEMATICS')
fig,axs=plt.subplots(1,2,figsize=(7.3,3.2));board(axs[0],b,'A move is an ordered pair',arrows=[('e5','g7',TEAL),('e5','f4',GOLD)]);names=['source','destination','interaction','context','promotion','castle'];bg=[];bf=[]
for name in names:
    ar=t['decoder.'+name];bg.append(np.linalg.norm(ar[13] if len(ar)>1 else ar[0]));bf.append(np.linalg.norm(ar[11] if len(ar)>1 else ar[0]))
y=np.arange(6);axs[1].barh(y-.17,bg,.32,color=TEAL,label='Bg7');axs[1].barh(y+.17,bf,.32,color=GOLD,label='Bf4');axs[1].set_yticks(y,names);axs[1].invert_yaxis();axs[1].set_xlabel('Vector L2 norm (not contribution)');axs[1].legend();axs[1].set_title('Inputs to the move decoder');fig.tight_layout();save('decoder_terms',fig)
fig,axs=plt.subplots(1,2,figsize=(7.3,3.2));moves=r['baseline']['moves'][:5];axs[0].barh([a['san'] for a in moves],[100*a['policy'] for a in moves],color=[TEAL,GOLD,LIGHT,LIGHT,LIGHT]);axs[0].invert_yaxis();axs[0].set_xlabel('Raw policy probability (%)');axs[0].set_xlim(0,100);axs[0].set_title('Raw move preference');polish_chart(axs[0]);label_bars(axs[0],axs[0].containers[0],fmt='%.2f%%');dist=[r['baseline']['state_wdl'],moves[0]['action_wdl'],moves[1]['action_wdl'],moves[2]['action_wdl']];lefts=np.zeros(4)
for i,(col,label) in enumerate([(TEAL,'Win'),(GOLD,'Draw'),(RED,'Loss')]):
    vals=np.array(dist)[:,i]*100;axs[1].barh(range(4),vals,left=lefts,color=col,label=label);lefts+=vals
axs[1].set_yticks(range(4),['State','Bg7 action','Bf4 action','Kc3 action']);axs[1].invert_yaxis();axs[1].set_xlabel('Predicted W / D / L (%)');axs[1].legend(loc='upper center',bbox_to_anchor=(.5,-.2),ncol=3,fontsize=8);axs[1].set_xlim(0,100);axs[1].set_title('Different prediction targets');fig.tight_layout();save('outputs',fig)
fig,ax=plt.subplots(figsize=(7.2,3.4));read=r['readout'];ss=sorted(read,key=lambda a:abs(a['gap_contribution']),reverse=True)[:12];ax.barh([str(a['unit']) for a in ss],[a['gap_contribution'] for a in ss],color=[TEAL if a['gap_contribution']>=0 else RED for a in ss]);ax.invert_yaxis();ax.axvline(0,color=INK,lw=.5);ax.set_xlabel('Contribution to logit(Bg7) - logit(Bf4)');ax.set_ylabel('Move-decoder unit (0-based)');ax.set_title('What adds to the Bg7 - Bf4 logit gap');polish_chart(ax);label_bars(ax,ax.containers[0],fmt='%+.3f',fontsize=8.5);ax.set_xlim(-.14,.32);fig.tight_layout();save('readout',fig)
heads=np.array([a['bg7_policy'] for a in r['interventions'][:96]]).reshape(8,12);delta=(heads-r['baseline_focus']['bg7_policy'])*100
fig,ax=plt.subplots(figsize=(7.2,3.4));im=ax.imshow(delta,cmap=DIVERGING,vmin=-30,vmax=30,aspect='auto');ax.set_xticks(range(12),range(1,13));ax.set_yticks(range(8),range(1,9));ax.set_xlabel('Head removed');ax.set_ylabel('Block');ax.set_title('Remove one head: how much does Bg7 change?');ax.add_patch(Rectangle((4.5,-.5),1,1,fill=False,edgecolor=GOLD,lw=2.2));ax.text(5,0,f'{delta[0,5]:+.1f}',ha='center',va='center',fontsize=8.5,color='white');fig.colorbar(im,ax=ax,label='Bg7 probability change (percentage points)');fig.tight_layout();save('head_ablations',fig)
selected=[('Baseline',r['baseline_focus'])]+[(a['label'],a) for a in r['interventions'] if a['label'] in ['Block 1, head 6','Selected head, mean over receiving squares','Block 1, adaptive bias off','decoder.context off','Move-decoder unit 34 off']]
fig,ax=plt.subplots(figsize=(7.2,3.3));labels=['Baseline','B1 H6 removed','B1 H6 mean-replaced','B1 adaptive off','Decoder unit 34 off','Decoder context off'];vals=[selected[0][1]['bg7_policy']]
for label in ['Block 1, head 6','Selected head, mean over receiving squares','Block 1, adaptive bias off','Move-decoder unit 34 off','decoder.context off']:vals.append(next(a['bg7_policy'] for a in r['interventions'] if a['label']==label))
ax.barh(labels,np.array(vals)*100,color=[INK,RED,GOLD,TEAL,GOLD,RED]);ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xlabel('Bg7 probability (%)');ax.set_title('Bg7 under six interventions');polish_chart(ax);label_bars(ax,ax.containers[0],fmt='%.2f%%');ax.axvline(100*vals[0],ls=':',lw=1.3,color=INK,alpha=.65);fig.tight_layout();save('interventions',fig)
fig,axs=plt.subplots(1,2,figsize=(7.2,3));axs[0].bar(['Bg7','Bf4','Kc3'],[72.12,19.89,4.15],color=[TEAL,GOLD,RED]);axs[0].set_ylim(0,100);axs[0].set_ylabel('Native prior (%)');axs[1].bar(['Bg7','Bf4','Kc3'],np.array([977,43,3])/1023*100,color=[TEAL,GOLD,RED]);axs[1].set_ylim(0,100);axs[1].set_ylabel('Child visits (%)');axs[0].set_title('Before search allocation');axs[1].set_title('After 1,024 root visits');
for aa in axs:polish_chart(aa,'y');label_bars(aa,aa.containers[0],fmt='%.2f%%');aa.set_ylim(0,108)
fig.tight_layout();save('search',fig)
fig,axs=plt.subplots(1,3,figsize=(7.4,2.5));
for ax,mat,ttl in zip(axs,[C,L,A],['Content scores','After both biases','Row softmax']):
    im=ax.imshow(mat,cmap=SEQUENTIAL,aspect='auto');ax.set_xticks(range(3),['g7','h7','h6']);ax.set_yticks(range(3),['g7','h7','h6']);ax.set_title(ttl,fontsize=10)
    for ii in range(3):
        for jj in range(3):
            rgb=im.cmap(im.norm(mat[ii,jj]))[:3];lum=float(np.dot(rgb,[.2126,.7152,.0722]))
            ax.text(jj,ii,f'{mat[ii,jj]:.3f}',ha='center',va='center',fontsize=9,color='white' if lum<.48 else INK)
fig.tight_layout();save('toy_attention',fig,'ILLUSTRATIVE')
fig,ax=plt.subplots(figsize=(7.2,2.8));xx=np.arange(3);ax.bar(xx-.24,p,.24,label='Before',color=GOLD);ax.bar(xx,newp,.24,label='One update',color=TEAL);ax.bar(xx+.24,target,.24,label='Target',color=INK);ax.set_xticks(xx,['Bg7 (toy)','Bf4 (toy)','Kc3 (toy)']);ax.set_ylabel('Probability');ax.set_ylim(0,.8);ax.legend(ncol=3);ax.set_title('One computed learning step  /  illustrative');fig.tight_layout();save('toy_training',fig,'ILLUSTRATIVE')
# A readable layer/token heatmap, same scales across examples.
fig,axs=plt.subplots(1,2,figsize=(7.3,3.45));board(axs[0],b,'Before 81.Bg7');im=axs[1].imshow(np.stack([t[f'block.{i}.residual'][54,:32] for i in range(8)]),aspect='auto',cmap=DIVERGING,vmin=-6,vmax=6);axs[1].set_yticks(range(8),range(1,9));axs[1].set_xlabel('First 32 of 384 coordinates');axs[1].set_ylabel('Block completed');axs[1].set_title('g7 across the network');fig.colorbar(im,ax=axs[1],fraction=.045);fig.tight_layout();save('g7_layers',fig)
# Extra game positions with saved measures, not new inference.
other=[]
for ply in [12,28,48,76,86,126,144,158,160,162,166,170,184,186]:
    pp=next((q for q in positions if q['game_id']==record['game_id'] and q['ply']==ply),None)
    if pp and (ATLAS/'analyses'/pp['id']/'summary.json').exists():
        ss=json.loads((ATLAS/'analyses'/pp['id']/'summary.json').read_text());other.append({'record':pp,'summary':ss})
facts['other_positions']=other
for k,ply,ttl in [('early_ne5',12,'Before 7.Ne5'),('before_g5',82,'Before 42.g5: opening a route'),('late_a7',184,'Before 93.a7: promotion approaches')]:
    bo=allboards[ply];fig,ax=plt.subplots(figsize=(4.1,3.8));board(ax,bo,ttl);fig.tight_layout();save(k,fig,'RULES / GAME')
# Full legal game and legal history; a compact exact record accompanies the PDF.
(D/'catalan.pgn').write_text((D/'catalan.pgn').read_text())
(D/'bg7-record.json').write_text(json.dumps(record,indent=2))
(D/'measurements.json').write_text(json.dumps(r,indent=2))
(D/'facts.json').write_text(json.dumps(facts,indent=2))
print(json.dumps({'params':params,'nonzero':nonzero,'projection':projection,'toy':toy,'real_dot':facts['real_dot'],'decoder_first':facts['decoder_first'],'figures':len(facts['figure_manifest']),'validation':facts['validation']},indent=2))
