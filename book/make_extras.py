"""Additional diagrams, typesetting math and source identity audit. CPU only."""
import os
os.environ['MPLBACKEND']='Agg';os.environ['MPLCONFIGDIR']='/tmp/vegam-book-mpl'
from pathlib import Path
import json,hashlib,math,re
import numpy as np
import chess,chess.pgn
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Rectangle
from matplotlib.font_manager import FontProperties
from visual_style import board, use_style, ensure_piece_assets
ROOT=Path(__file__).resolve().parent;WS=ROOT.parent;NEXT=WS;FIXED=WS/'engine';F=ROOT/'figures';D=ROOT/'data'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
INK='#173042';TEAL='#087E8B';GOLD='#BC812A';LIGHT='#E4EBE8'
use_style();ensure_piece_assets()
facts=json.loads((D/'facts.json').read_text())
font=FontProperties(family='DejaVu Sans')
with (D/'catalan.pgn').open() as f:g=chess.pgn.read_game(f)
b=g.board();boards=[b.copy()];sans=[]
for m in g.mainline_moves():sans.append(b.san(m));b.push(m);boards.append(b.copy())
# Use the same crisp piece artwork and coordinate system in every chapter.
fig,ax=plt.subplots(figsize=(4.2,4.0));b=boards[86]
board(ax,b,'Before 44.h6',arrows=[('g5','g6',GOLD),('h5','h6',TEAL)])
fig.tight_layout();fig.savefig(F/'before_h6.png',dpi=300,bbox_inches='tight',pad_inches=.14);plt.close(fig)
facts['figure_manifest']['before_h6']={'kind':'RULES / GAME'}

# The network and search get distinct regions. Only enabled outputs feed search.
fig,ax=plt.subplots(figsize=(7.3,8.3));ax.set_xlim(0,10);ax.set_ylim(0,12);ax.axis('off')
def card(x,y,w,h,title,body='',color=TEAL,fill='#F1F5F2',fs=11):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.035,rounding_size=.09',facecolor=fill,edgecolor='#BCCECE',lw=.8))
 ax.plot([x+.13,x+.13],[y+.16,y+h-.16],color=color,lw=3.2,solid_capstyle='round')
 ax.text(x+w/2,y+h-.18,title,ha='center',va='top',fontsize=fs,fontweight='bold',color=INK)
 if body:ax.text(x+w/2,y+h-.57,body,ha='center',va='top',fontsize=10,color='#405966',linespacing=1.35)
def ar(x,y,x2,y2,label='',color=TEAL):
 ax.add_patch(FancyArrowPatch((x,y),(x2,y2),arrowstyle='-|>',mutation_scale=14,lw=1.7,color=color,shrinkA=3,shrinkB=3))
 if label:ax.text((x+x2)/2+.12,(y+y2)/2,label,ha='left',va='center',fontsize=9,color=color)
ax.text(.15,11.8,'ONE NEURAL EVALUATION',fontsize=10,fontweight='bold',color=TEAL)
card(1.7,10.45,6.6,1.02,'1  Encode the position','64 squares x 120 explicit features')
card(1.7,8.94,6.6,1.02,'2  Project + add square embeddings','64 square representations x 384 coordinates')
ar(5,10.42,5,9.99)
card(1.7,7.16,6.6,1.3,'3  Transform, eight times','Normalize -> attention -> residual add\nNormalize -> MLP -> residual add')
ar(5,8.9,5,8.49)
card(1.7,5.77,6.6,.9,'4  Final LayerNorm','64 x 384, still indexed by square')
ar(5,7.12,5,6.7)
card(.15,3.88,4.25,1.25,'Global route','Mean over 64 rows\nLinear + GELU -> 128 context')
card(5.35,3.88,4.5,1.25,'Move route','Source + destination + product\n+ context + promotion + castle')
ar(3.2,5.72,2.28,5.18);ar(6.8,5.72,7.6,5.18);ar(4.44,4.48,5.31,4.48,'')
card(.15,2.03,4.25,1.17,'State WDL','Outcome + duration also decoded;\ntheir daily-search uses are off.',color=GOLD,fill='#F8F1E5')
card(5.35,2.03,4.5,1.17,'Policy + action WDL','GELU -> Linear -> GELU -> Linear\nOne policy logit + W/D/L per move',color=GOLD,fill='#F8F1E5')
ar(2.28,3.83,2.28,3.25);ar(7.6,3.83,7.6,3.25)
card(.75,.06,8.5,1.3,'SEARCH: MCTS + chess rules','Policy guides exploration; state values back up.\nAction WDL advises edges that have no completed visits.',color=GOLD,fill='#FBF4E7',fs=12)
ar(2.28,1.99,2.28,1.4,'state WDL',GOLD);ar(7.6,1.99,7.6,1.4,'priors + action',GOLD)
fig.tight_layout();fig.savefig(F/'architecture.png',dpi=300,bbox_inches='tight',pad_inches=.14);plt.close(fig);facts['figure_manifest']['architecture']={'kind':'CODE'}

# A routing stream and a message stream: values bypass compatibility scoring.
fig,ax=plt.subplots(figsize=(7.3,4.0));ax.set_xlim(0,10);ax.set_ylim(0,5.8);ax.axis('off')
def routebox(x,y,w,h,label,color=TEAL):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.025,rounding_size=.08',facecolor='#F1F5F2',edgecolor=color,lw=.85))
 ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=10.5,color=INK,linespacing=1.4)
ax.text(.02,5.5,'ROUTING: produce one score for each supplier',fontsize=10,fontweight='bold',color=GOLD)
routebox(.05,3.8,2,.85,'Receiver: g7\n384 numbers');routebox(.05,2.42,2,.85,'Supplier: h6\n384 numbers')
routebox(2.6,3.8,1.5,.85,'Q projection\n32 numbers',GOLD);routebox(2.6,2.42,1.5,.85,'K projection\n32 numbers',GOLD)
routebox(4.62,3.03,2.4,1.12,'Dot product / sqrt(32)\n+ fixed bias\n+ adaptive bias',GOLD)
routebox(7.62,3.03,2.3,1.12,'Softmax across\nall 64 suppliers',GOLD)
ar(2.09,4.22,2.57,4.22,color=GOLD);ar(2.09,2.84,2.57,2.84,color=GOLD)
ar(4.13,4.22,4.59,3.82,color=GOLD);ar(4.13,2.84,4.59,3.34,color=GOLD);ar(7.05,3.59,7.59,3.59,color=GOLD)
ax.text(2.5,1.77,'MESSAGE: mix the transformed supplier content',fontsize=10,fontweight='bold',color=TEAL)
routebox(2.6,.35,1.5,.85,'V projection\n32 numbers');routebox(6.35,.22,3.57,1.1,'Multiply V by its weight;\nsum over 64 suppliers\n-> 32-number head output')
ar(1.03,2.38,1.03,.8);ar(1.03,.8,2.57,.8);ar(4.14,.8,6.32,.8);ar(8.76,2.99,8.76,1.35,color=GOLD)
fig.tight_layout();fig.savefig(F/'qkv_routes.png',dpi=300,bbox_inches='tight',pad_inches=.14);plt.close(fig);facts['figure_manifest']['qkv_routes']={'kind':'CODE / SCHEMATIC'}
# source catalog with hashes and named line ranges
# Publication sources; original snapshot hashes remain in source-index.json.
paths={'model':NEXT/'nextchess/model.py','backend':FIXED/'src/neural/next_backend.cc',
       'encoder':FIXED/'src/neural/encoder.cc','training':NEXT/'nextchess/cloud_train.py',
       'search':FIXED/'src/search/classic/search.cc','game':D/'catalan.pgn'}
source_index={key:{'path':str(p.relative_to(WS)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for key,p in paths.items()}
(D/'publication-source-index.json').write_text(json.dumps(source_index,indent=2))
# Full prefix for a self-contained reader reproduction.
prefix=' '.join(f'{i//2+1}. {sans[i]} {sans[i+1]}' for i in range(0,160,2));(D/'prefix.txt').write_text(prefix)
# Equation images are typeset from formulas, not image generation.
eqs={
'input_projection':[r'H^{(0)}=XW_{in}^{T}+b_{in}+E',r'H^{(0)}_{sj}=\sum_{c=0}^{119}X_{sc}W_{in,jc}+b_{in,j}+E_{sj}'],
'layernorm':[r'm=\frac{1}{d}\sum_{r=1}^{d}x_r,\quad v=\frac{1}{d}\sum_{r=1}^{d}(x_r-m)^2',r'\operatorname{LN}(x)_r=\gamma_r\frac{x_r-m}{\sqrt{v+\epsilon}}+\beta_r'],
'qkv':[r'Q=ZW_Q,\qquad K=ZW_K,\qquad V=ZW_V'],
'bilinear':[r'q_i k_j^T=z_iW_QW_K^Tz_j^T=z_iMz_j^T'],
'scale_variance':[r'\operatorname{Var}\!\left(\sum_{r=1}^{d}q_rk_r\right)=d',r'\operatorname{Var}\!\left(\frac{q\cdot k}{\sqrt{d}}\right)=1\quad\mathrm{(under\ stated\ assumptions)}'],
'softmax':[r'p_j=\frac{e^{l_j}}{\sum_k e^{l_k}},\qquad \frac{p_i}{p_j}=e^{l_i-l_j}'],
'gates':[r'u=\operatorname{vec}(\operatorname{Compress}(Z))',r'g=\tanh\!\left(\operatorname{GELU}(uW_1+b_1)W_2+b_2\right)'],
'adaptive':[r'D_{hij}=\sum_{r=1}^{16}L_{hir}\,g_{hr}\,R_{hjr}',r'D_h=L_h\operatorname{diag}(g_h)R_h^T'],
 'transport':[r'Y_i=\sum_{j=1}^{64}A_{ij}V_j,\qquad Y=AV'],
'residual':[r'X^{\prime}=X+\operatorname{AttentionUpdate}(\operatorname{LN}_1(X))',r'X_{next}=X^{\prime}+\operatorname{GELU}(\operatorname{LN}_2(X^{\prime})W_1+b_1)W_2+b_2'],
'wdl':[r'W=\frac{1-D+q}{2},\quad L=\frac{1-D-q}{2},\quad q=W-L'],
'crossentropy':[r'\mathcal{L}_{policy}=-\sum_i y_i\log p_i',r'H(y)=-\sum_i y_i\log y_i,\quad D_{KL}(y\Vert p)=\mathcal{L}_{policy}-H(y)'],
'loss_total':[r'\mathcal{L}=\mathcal{L}_{policy}+\mathcal{L}_{state}+0.5\mathcal{L}_{action}',r'\phantom{\mathcal{L}=}+0.25\mathcal{L}_{outcome}+0.05\mathcal{L}_{duration}'],
'duration':[r'y=\log(1+t),\quad z=(y-\mu)e^{-\log\sigma}',r'\ell_{observed}=\frac{z^2}{2}+\log\sigma,\quad\ell_{censored}=-\log\Phi(-z)'],
'ce_gradient':[r'\frac{\partial\mathcal{L}}{\partial l_k}=p_k-y_k,\qquad w_{new}=w-\eta\nabla_w\mathcal{L}'],
'attention_gradient':[r'\frac{\partial a_i}{\partial l_j}=a_i(\delta_{ij}-a_j)',r'\frac{\partial J}{\partial l_j}=a_j\left(v_{j,0}-\sum_k a_kv_{k,0}\right)'],
'adam':[r'm_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2',r'\hat m_t=\frac{m_t}{1-\beta_1^t},\quad\hat v_t=\frac{v_t}{1-\beta_2^t}',r'w_t=(1-\eta\lambda)w_{t-1}-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}'],
'decoder':[r'a_m=s_m+d_m+\frac{s_m\odot d_m}{8}+g+p_m+k_m',r'h_m=\operatorname{GELU}(\operatorname{GELU}(a_m)W_J+b_J)',r'[l_m,u_{m,W},u_{m,D},u_{m,L}]=h_mW_{out}+b_{out}'],
'readout':[r'l_{Bg7}-l_{Bf4}=\sum_{r=0}^{63}w_r\left(h_{Bg7,r}-h_{Bf4,r}\right)'],
'puct':[r'\mathrm{selection}\ \approx\ Q+cP\frac{\sqrt{N_{parent}}}{1+N_{child}}'],
'coldq':[r'Q_{cold}=0.75\,FPU+0.25\,(q_{action}+s_{draw}d_{action})'],
'score_fallback':[r'cp_{fallback}=45\tan(1.56728071628\,WL)'],
'sae':[r'f=\operatorname{ReLU}(xW_{enc}+b_{enc}),\quad\hat x=fW_{dec}+b_{dec}',r'\mathcal{L}_{SAE}=\Vert x-\hat x\Vert_2^2+\lambda\Vert f\Vert_1']}
# Some equations above use column convention for standalone per-vector maps;
# source explains transpose distinction; equation caption will make it explicit.
for name,lines in eqs.items():
 fig=plt.figure(figsize=(7, .42*len(lines)+.14));fig.patch.set_alpha(0)
 for i,line in enumerate(lines):fig.text(.025,1-(i+.7)/(len(lines)+.35),'$'+line+'$',fontsize=15,color=INK,va='center')
 fig.savefig(F/('eq_'+name+'.png'),dpi=260,bbox_inches='tight',pad_inches=.12,transparent=True);plt.close(fig)
(D/'equations.json').write_text(json.dumps(eqs,indent=2));(D/'facts.json').write_text(json.dumps(facts,indent=2));print('extras',len(eqs),'equations',len(source_index),'source records')
