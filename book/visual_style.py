"""Print-first scientific figures with rule-accurate chess pieces and coordinates."""
from pathlib import Path
import io, json, subprocess
from functools import lru_cache
import numpy as np
import chess, chess.svg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'figures' / 'pieces'
INK = '#173042'
TEAL = '#087E8B'
GOLD = '#BC812A'
RED = '#AF514C'
LIGHT = '#E4EBE8'
PAPER = '#FCFBF7'
SEQUENTIAL = LinearSegmentedColormap.from_list('vegam-sequential', ['#F7F4EA','#D4E4DD','#86BCBB','#317E93','#183B60'])
DIVERGING = LinearSegmentedColormap.from_list('vegam-diverging', ['#245C82','#86B5C9','#FAF7EF','#D4947C','#A53F49'])


def use_style():
    plt.rcParams.update({
        'font.family':'DejaVu Sans', 'font.size':11,
        'text.color':INK, 'axes.labelcolor':INK, 'axes.edgecolor':'#A7B7B9',
        'axes.titlecolor':INK, 'axes.titlesize':11.5, 'axes.titleweight':'bold',
        'axes.labelsize':10.5, 'axes.titlepad':11,
        'xtick.color':'#4B606B', 'ytick.color':'#4B606B',
        'xtick.labelsize':9, 'ytick.labelsize':9,
        'axes.spines.top':False, 'axes.spines.right':False,
        'axes.linewidth':.65, 'grid.color':'#D4DEDC', 'grid.linewidth':.6,
        'figure.facecolor':'white', 'axes.facecolor':'white', 'savefig.facecolor':'white',
        'legend.frameon':False, 'legend.fontsize':9,
        'lines.linewidth':1.7,
    })


def ensure_piece_assets():
    """Rasterize python-chess's explicit SVG paths once using installed librsvg."""
    ASSETS.mkdir(exist_ok=True)
    specs=[]
    for symbol in 'PNBRQKpnbrqk':
        for heat in (False,True):
            key=('white-' if symbol.isupper() else 'black-')+symbol.lower()+('-heat' if heat else '')
            path=ASSETS/(key+'.png')
            if path.exists():continue
            svg=chess.svg.piece(chess.Piece.from_symbol(symbol),size=256)
            if symbol.isupper():
                svg=svg.replace('"#fff"','"#FFFDF7"').replace('"#000"','"#142B3B"')
            else:
                svg=svg.replace('fill="#000"','fill="#142B3B"').replace('stroke="#000"',f'stroke="{"#F8FAF5" if heat else "#142B3B"}"').replace('"#fff"','"#FFFDF7"')
            (ASSETS/(key+'.svg')).write_text(svg)
            specs.append({'svg':svg,'path':str(path)})
    if specs:
        code='''import sys,json,gi,cairo
from gi.repository import GLib
gi.require_version('Rsvg','2.0')
from gi.repository import Rsvg
for item in json.load(sys.stdin):
 surface=cairo.ImageSurface(cairo.FORMAT_ARGB32,256,256)
 context=cairo.Context(surface)
 handle=Rsvg.Handle.new_from_data(item['svg'].encode())
 rect=Rsvg.Rectangle();rect.x=0;rect.y=0;rect.width=256;rect.height=256
 handle.render_document(context,rect)
 surface.write_to_png(item['path'])
'''
        subprocess.run(['/usr/bin/python3','-c',code],input=json.dumps(specs),text=True,check=True)


@lru_cache(maxsize=24)
def piece(symbol,heat=False):
    key=('white-' if symbol.isupper() else 'black-')+symbol.lower()+('-heat' if heat else '')
    return np.asarray(Image.open(ASSETS/(key+'.png')).convert('RGBA'))


def board(ax,bo,title='',values=None,cmap='YlGnBu',vmin=None,vmax=None,arrows=(),label_values=False,highlights=None):
    ax.set_aspect('equal');ax.set_xlim(-.18,8.18);ax.set_ylim(-.18,8.18)
    if values is None:
        colors=['#F3EEE0','#8EA99D']
        for s in range(64):
            f=s%8;rk=s//8
            ax.add_patch(Rectangle((f,rk),1,1,facecolor=colors[(f+rk)%2==0],edgecolor='none',zorder=0))
    else:
        v=np.asarray(values).reshape(8,8)
        cm=DIVERGING if cmap=='RdBu_r' else SEQUENTIAL if cmap=='YlGnBu' else cmap
        im=ax.imshow(v,origin='lower',extent=(0,8,0,8),cmap=cm,vmin=vmin,vmax=vmax,interpolation='nearest',zorder=0)
        cb=ax.figure.colorbar(im,ax=ax,fraction=.048,pad=.055,shrink=.91)
        cb.outline.set_visible(False);cb.ax.tick_params(labelsize=9,length=2)
        for n in range(1,8):
            ax.plot([n,n],[0,8],color='white',alpha=.25,lw=.45,zorder=1)
            ax.plot([0,8],[n,n],color='white',alpha=.25,lw=.45,zorder=1)
    ax.add_patch(Rectangle((0,0),8,8,facecolor='none',edgecolor='#526D70',lw=.8,zorder=2))
    marks={} if highlights is None else dict(highlights)
    for a,z,col in arrows:
        if isinstance(a,str):a=chess.parse_square(a)
        if isinstance(z,str):z=chess.parse_square(z)
        marks.setdefault(a,col);marks.setdefault(z,col)
    if values is not None:marks.setdefault(chess.G7,GOLD)
    for s,col in marks.items():
        if isinstance(s,str):s=chess.parse_square(s)
        ax.add_patch(Rectangle((s%8+.035,s//8+.035),.93,.93,fill=False,edgecolor=col,lw=1.8,zorder=4))
    for s,pc in bo.piece_map().items():
        if label_values:continue
        pad=.11 if values is not None else .04
        ax.imshow(piece(pc.symbol(),values is not None),extent=(s%8+pad,s%8+1-pad,s//8+pad,s//8+1-pad),origin='upper',zorder=5,interpolation='lanczos')
    if label_values:
        for s,vv in enumerate(values):ax.text(s%8+.5,s//8+.5,f'{vv:.2f}',ha='center',va='center',fontsize=8,color=INK,zorder=7)
    for i,(a,z,col) in enumerate(arrows):
        if isinstance(a,str):a=chess.parse_square(a)
        if isinstance(z,str):z=chess.parse_square(z)
        # Curving short opposing arrows keeps the bishop/pawn relationship legible.
        reverse=any((aa==z or aa==chess.square_name(z)) and (zz==a or zz==chess.square_name(a)) for aa,zz,_ in arrows)
        ax.add_patch(FancyArrowPatch((a%8+.5,a//8+.5),(z%8+.5,z//8+.5),arrowstyle='-|>',mutation_scale=15,lw=2.4,color=col,alpha=.96,shrinkA=8,shrinkB=10,connectionstyle='arc3,rad=.17' if reverse else 'arc3,rad=0',zorder=8))
    ax.set_xticks(np.arange(8)+.5,list('abcdefgh'));ax.set_yticks(np.arange(8)+.5,list('12345678'))
    ax.tick_params(length=0,labelsize=11,pad=3)
    for sp in ax.spines.values():sp.set_visible(False)
    ax.set_title(title,fontsize=11.5,pad=12,color=INK)
    # imshow resets the limits; restore the full, correctly oriented board.
    ax.set_xlim(-.12,8.12);ax.set_ylim(-.12,8.12)


def polish_chart(ax,axis='x'):
    ax.set_axisbelow(True)
    ax.grid(axis=axis,alpha=.7)
    ax.tick_params(length=0,pad=5)


def label_bars(ax,container,fmt='%.1f',padding=4,fontsize=9):
    ax.bar_label(container,fmt=fmt,padding=padding,fontsize=fontsize,color=INK)
