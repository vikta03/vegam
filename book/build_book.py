"""Typeset the source-verified VEGAM textbook with navigable PDF structure."""
import re,json,math,html,hashlib
from pathlib import Path
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT,TA_CENTER,TA_JUSTIFY
from reportlab.platypus import BaseDocTemplate,PageTemplate,Frame,Paragraph,Spacer,PageBreak,Table,TableStyle,Image,KeepTogether,Flowable,CondPageBreak
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.utils import ImageReader
from PIL import Image as PILImage
ROOT=Path(__file__).resolve().parent;WS=ROOT.parent;OUT=WS/'output/pdf/inside-vegam-from-chess-pieces-to-learned-intelligence.pdf';OUT.parent.mkdir(parents=True,exist_ok=True);FIG=ROOT/'figures';DATA=ROOT/'data'
facts=json.loads((DATA/'facts.json').read_text());measure=json.loads((DATA/'measurements.json').read_text());W,H=540,720;M=45;CW=W-2*M
import os
font_dirs = [Path(os.environ['VEGAM_FONT_DIR'])] if os.environ.get('VEGAM_FONT_DIR') else []
font_dirs += [Path('/usr/share/fonts/truetype'), Path('/usr/share/fonts')]
def font_file(name):
 for directory in font_dirs:
  matches=sorted(directory.rglob(name)) if directory.exists() else []
  if matches:return matches[0]
 raise FileNotFoundError('Install fonts-crosextra-caladea and fonts-dejavu-core, or set VEGAM_FONT_DIR: '+name)
for name,filename in [('Body','Caladea-Regular.ttf'),('Body-Bold','Caladea-Bold.ttf'),('Body-Italic','Caladea-Italic.ttf'),('Body-BoldItalic','Caladea-BoldItalic.ttf'),('Sans','DejaVuSans.ttf'),('Sans-Bold','DejaVuSans-Bold.ttf'),('Sans-Italic','DejaVuSans-Oblique.ttf'),('Mono','DejaVuSansMono.ttf')]:
 pdfmetrics.registerFont(TTFont(name,str(font_file(filename))))
pdfmetrics.registerFontFamily('Body',normal='Body',bold='Body-Bold',italic='Body-Italic',boldItalic='Body-BoldItalic');pdfmetrics.registerFontFamily('Sans',normal='Sans',bold='Sans-Bold',italic='Sans-Italic',boldItalic='Sans-Bold')
INK=colors.HexColor('#173042');TEAL=colors.HexColor('#087E8B');GOLD=colors.HexColor('#BE8A38');MUTED=colors.HexColor('#536774');LIGHT=colors.HexColor('#EDF3F1');PAPER=colors.HexColor('#FCFBF7')
styles={
 'body':ParagraphStyle('body',fontName='Body',fontSize=11.3,leading=15.1,textColor=INK,spaceAfter=6.5,allowWidows=0,allowOrphans=0,splitLongWords=1),
 'chapter':ParagraphStyle('chapter',fontName='Body-Bold',fontSize=26,leading=29.5,textColor=INK,spaceBefore=10,spaceAfter=16,keepWithNext=True),
 'section':ParagraphStyle('section',fontName='Sans-Bold',fontSize=11.5,leading=15.6,textColor=TEAL,spaceBefore=16,spaceAfter=8,keepWithNext=True),
 'answer':ParagraphStyle('answer',fontName='Body',fontSize=11.1,leading=14.4,textColor=INK,spaceAfter=6.5,allowWidows=0,allowOrphans=0),
 'reference':ParagraphStyle('reference',fontName='Body',fontSize=10.5,leading=14,textColor=INK,spaceAfter=5.5,allowWidows=0,allowOrphans=0),
 'caption':ParagraphStyle('caption',fontName='Sans',fontSize=8.15,leading=11.4,textColor=MUTED,spaceBefore=5,spaceAfter=12),
 'small':ParagraphStyle('small',fontName='Sans',fontSize=8.8,leading=12.5,textColor=INK,spaceAfter=7),
 'objective':ParagraphStyle('objective',fontName='Sans',fontSize=9.1,leading=13.3,textColor=INK,spaceBefore=14,spaceAfter=18,borderColor=TEAL,borderWidth=.6,borderPadding=10,backColor=LIGHT),
 'check':ParagraphStyle('check',fontName='Sans',fontSize=9.4,leading=13.7,textColor=INK,spaceBefore=8,spaceAfter=12,borderPadding=10,borderWidth=.6,borderColor=GOLD,backColor=colors.HexColor('#FBF5EA')),
 'cell':ParagraphStyle('cell',fontName='Sans',fontSize=8.6,leading=11.6,textColor=INK,splitLongWords=1),
 'headcell':ParagraphStyle('headcell',fontName='Sans-Bold',fontSize=8.35,leading=11,textColor=colors.white),
 'mono':ParagraphStyle('mono',fontName='Mono',fontSize=8.1,leading=11.5,textColor=INK,spaceAfter=8,splitLongWords=1),
}
styles['body'].bulletFontName='Body'
# Validate all intended references before rendering.
source_ids={'S1','S2','S3','S4','S5','S6','S7','S8','S9','T1','T2','T3','T4','G1','M1','M2','M3','A1'}|{f'R{i}' for i in range(1,9)}
def fmt(s,linkrefs=True):
 s=s.replace('—',' - ').replace('–','-').replace('‑','-').replace('−','-').replace('“','"').replace('”','"').replace('’',"'")
 s=html.escape(s)
 # Plain references resolve to source anchors, with grouped references retained.
 if linkrefs:
  def rr(m):
   txt=m.group(1)
   for sid in sorted(source_ids,key=len,reverse=True):txt=re.sub(r'\b'+sid+r'\b',lambda _:f'<link href="#source-{sid}" color="#087E8B">{sid}</link>',txt)
   return '['+txt+']'
  s=re.sub(r'\[([STMRGA][0-9][^\]]{0,80})\]',rr,s)
 s=re.sub(r'`([^`]+)`',r'<font name="Mono" size="8.3">\1</font>',s)
 s=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',s)
 s=re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)',r'<i>\1</i>',s)
 s=re.sub(r'https://[^\s<]+',lambda m:f'<link href="{m[0]}" color="#087E8B">Read the primary source</link>',s)
 return s

def para(s,style='body'):return Paragraph(fmt(s),styles[style])
def cells(rows):return [[Paragraph(fmt(str(c)),styles['headcell' if i==0 else 'cell']) for c in row] for i,row in enumerate(rows)]
def make_table(rows,widths=None):
 if widths is None:widths=[CW/len(rows[0])]*len(rows[0])
 tab=Table(cells(rows),colWidths=widths,repeatRows=1,hAlign='LEFT')
 tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),INK),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,LIGHT]),('LINEBELOW',(0,0),(-1,0),.6,TEAL),('LINEBELOW',(0,-1),(-1,-1),.4,colors.HexColor('#B2C2C1'))]))
 return tab

def get_table(key):
 if key=='channels':return make_table([['Channels','Meaning','Anchor value'],['0-103','8 frames x 13 planes; own/their pieces and repetition','Square-specific'],['104-107','Castling-right markers on rook squares','All zero'],['108','Absolute side to move: White 0, Black 1','0 everywhere'],['109','Reversible halfmove clock / 100','0.17 everywhere'],['110','Inferred double-push intermediate square','All zero'],['111-118','Presence of each history frame','Eight ones per square'],['119','Constant feature','1 everywhere']],[65,265,120])
 if key=='projection':
  q=facts['projection_g7'][0]
  return make_table([['Channel','Input','Weight into dim 0','Product']]+[[c,f'{x:.2f}',f'{w:+.8f}',f'{v:+.8f}'] for c,x,w,v in q['terms']],[60,65,165,160])
 if key=='toy_qkv':return make_table([['Object','Rows / values'],['Z (g7, h7, h6)','[1, 0]; [0, 1]; [1, 1]'],['W_Q','[1, 0]; [1, 1]'],['W_K','[1, 1]; [0, 1]'],['W_V','[1, 2]; [-1, 1]'],['Q = Z W_Q','[1, 0]; [1, 1]; [2, 1]'],['K = Z W_K','[1, 1]; [0, 1]; [1, 2]'],['V = Z W_V','[1, 2]; [-1, 1]; [0, 3]']],[140,310])
 if key=='targets':return make_table([['Head','Target / supervision','Role in daily search'],['Policy','Legal-move distribution from record','Move priors'],['State WDL','Teacher best_q, best_d converted to W/D/L','Leaf value'],['Action WDL','Sparse V6 played/best action labels','Cold-edge advice, Q on'],['Outcome','Recorded result when valid','Extra blending off'],['Duration','Observed remaining plies or censored bound','Moves-left utility off']],[85,230,135])
 if key=='revision':return make_table([['Object','Earlier narrative','Current verified state'],['Training','One completed pass','Three completed passes + separate pilot'],['Network','Square transformer with adaptive relations','Same mechanism; trained wide config 384 / 12 / 576'],['Execution','Earlier CUDA speed lineage','Packaged TensorRT model and corrected binary'],['Action values','Initially optional / off','Daily Q launcher enables 75/25 cold advice'],['Search','Lc0-derived search description','Daily classic; separate corrected DAG preview'],['Interpretability','Architecture narrative','64-position calibration + saved traces + anchor interventions']],[85,170,195])
 if key=='readout':
  rr=sorted(measure['readout'],key=lambda x:abs(x['gap_contribution']),reverse=True)[:8]
  return make_table([['Unit','Bg7 h','Bf4 h','Policy weight','Gap term']]+[[v['unit'],f"{v['bg7_activation']:.6f}",f"{v['bf4_activation']:.6f}",f"{v['weight']:+.6f}",f"{v['gap_contribution']:+.6f}"] for v in rr],[40,102,102,103,103])
 if key=='tensors':
  rows=[['Object','One-position shape','Meaning'],['Features','64 x 120','Explicit encoded facts'],['Moves / legal mask','256 x 4 / 256','Candidates and padding'],['Input weight (stored)','384 x 120','120 inputs to 384 outputs'],['Square embedding','1 x 64 x 384','Learned square identity'],['Residual stream','64 x 384','Evolving square state'],['QKV, combined','64 x 1152','3 x 12 x 32 per square'],['Q, K, V individually','12 x 64 x 32','Head / square / coordinate'],['Static bias','12 x 64 x 64','Head / receiver / supplier'],['Compressed board','64 x 8','Gate-generator input'],['Gate hidden / final','128 / 12 x 16','Nonlinear global coefficients'],['Left and right factors','12 x 64 x 16 each','Adaptive relation patterns'],['Adaptive / attention','12 x 64 x 64','Signed scores / weights'],['Head output','12 x 64 x 32','Weighted value mixtures'],['MLP pre / activation','64 x 576','Square-wise hidden channels'],['Final normalized state','64 x 384','Input to head readouts'],['Global context','128','Pooled board representation'],['Source / destination','64 x 64 each','Per-square endpoint features'],['Move combined / hidden','256 x 64 each','One row per padded move'],['Policy logits','256','One per move slot'],['State / outcome logits','3 each','Separate W/D/L heads'],['Action logits','256 x 3','One W/D/L triple per move'],['Duration parameters','2','mu and log_sigma']]
  return make_table(rows,[139,118,193])
 if key=='toy_full':
  toy=facts['toy'];keys=['Z','Q','K','V','C','S','adaptive','L','A','Y','Wo','U','R'];rows=[['Stage','Rows in g7 / h7 / h6 order (where applicable)']]
  for k in keys:
   ar=npas(toy[k]);rows.append([k,'; '.join('['+', '.join(f'{x:.6f}' for x in row)+']' for row in ar)])
  for k in ['ln','ffpre','ffact','ffupdate','final']:rows.append(['g7 '+k,'['+', '.join(f'{v:.6f}' for v in toy[k])+']'])
  return make_table(rows,[83,367])
 if key=='anchor_legal':
  return make_table([['Move','Logit','Policy %','Action W %','D %','L %']]+[[v['san'],f"{v['logit']:.4f}",f"{100*v['policy']:.3f}"]+[f'{100*x:.2f}' for x in v['action_wdl']] for v in measure['baseline']['moves']],[65,75,78,82,75,75])
 if key=='identities':
  p=measure['provenance'];return make_table([['Artifact','Full SHA-256'],['Checkpoint EMA source',p['checkpoint_sha256']],['Deployed model.onnx',p['model_sha256']],['Daily binary',p['binary_sha256']],['Model source',p['model_definition_sha256']],['TensorRT engine',p['native_runtime']['model.trt.engine']],['Daily config',p['native_runtime']['lc0.config']]],[135,315])
 if key=='history':return make_table([['Frame','FEN (physical White-board orientation)']]+[[i,fen] for i,fen in enumerate(facts['history_fens'])],[48,402])
 raise KeyError(key)
def npas(a):return a # Keep typesetting environment independent of numerical work.
class Cover(Flowable):
 def __init__(self):super().__init__();self.width=CW;self.height=605
 def draw(self):
  c=self.canv;c.setFillColor(INK);c.rect(-M,-100,W,H+200,fill=1,stroke=0)
  c.setFillColor(GOLD);c.setFont('Sans-Bold',9);c.drawString(0,582,'A SOURCE-VERIFIED TECHNICAL TEXTBOOK')
  c.setFillColor(colors.white);c.setFont('Body-Bold',48);c.drawString(0,518,'Inside VEGAM')
  c.setFont('Body',25);c.drawString(0,478,'From Chess Pieces')
  c.drawString(0,448,'to Learned Intelligence')
  c.setStrokeColor(GOLD);c.setLineWidth(1.1);c.line(0,420,CW,420)
  p=Paragraph('Follow one beautiful bishop move through 120 input features, eight transformer blocks, learned action values and search.',ParagraphStyle('coversub',fontName='Body',fontSize=14,leading=20,textColor=colors.HexColor('#E3ECEB')));p.wrapOn(c,CW-15,90);p.drawOn(c,0,365)
  # Two accurately rendered anchor boards act as the cover illustration.
  im=PILImage.open(FIG/'position.png');ih=CW*im.height/im.width;c.setFillColor(colors.white);c.roundRect(-8,90,CW+16,ih+16,5,fill=1,stroke=0);c.drawImage(str(FIG/'position.png'),0,98,width=CW,height=ih,mask='auto')
  c.setFillColor(colors.HexColor('#C5D5D4'));c.setFont('Sans',8.8);c.drawString(0,64,'ACTUAL WEIGHTS  /  WORKED MATHEMATICS  /  CAUSAL TESTS')
  c.setFont('Body',11);c.drawString(0,34,'Remmy Victor');c.drawRightString(CW,34,'10 October 2026')
class Book(BaseDocTemplate):
 def __init__(self,*args,**kw):
  super().__init__(*args,**kw);self.current_chapter='Inside VEGAM';self.page_records=[];self.figure_pages=[]
  frame=Frame(M,47,CW,H-98,id='normal',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
  self.addPageTemplates(PageTemplate('book',[frame],onPage=self.paper,onPageEnd=self.page_bg))
 def beforeDocument(self):self.current_chapter='Inside VEGAM';self.page_records=[];self.figure_pages=[]
 def paper(self,c,doc):
  if doc.page>1:
   c.saveState();c.setFillColor(PAPER);c.rect(0,0,W,H,fill=1,stroke=0);c.restoreState()
 def page_bg(self,c,doc):
  if doc.page==1:return
  c.saveState()
  c.setStrokeColor(colors.HexColor('#C6D3D0'));c.setLineWidth(.4);c.line(M,H-32,W-M,H-32)
  c.setFillColor(MUTED);c.setFont('Sans',7.3);c.drawString(M,H-24,'INSIDE VEGAM')
  title=self.current_chapter
  while pdfmetrics.stringWidth(title,'Sans',7.3)>CW-100:title=title[:-2]
  c.drawRightString(W-M,H-24,title)
  c.setFont('Sans',8);c.drawString(M,26,'Source-verified edition / 2026-10-10');c.drawRightString(W-M,26,str(doc.page));c.restoreState()
 def afterFlowable(self,flow):
  if hasattr(flow,'_bookmark'):
   key=flow._bookmark;title=flow.getPlainText();self.canv.bookmarkPage(key)
   level=flow._level;self.canv.addOutlineEntry(title,key,level,False);self.notify('TOCEntry',(level,title,self.page,key));self.page_records.append({'title':title,'page':self.page,'level':level,'key':key})
   if level==0:self.current_chapter=title
  if hasattr(flow,'_figkey'):self.figure_pages.append({'name':flow._figkey,'page':self.page})
# TOC target anchors are named in actual headings.
story=[Cover(),PageBreak(),para('How to use this book','chapter')]
story.append(para('Begin with the board in Chapter 1. Chapters 2-7 build the forward computation with a complete numerical attention example. Chapters 8-10 explain how training gives those operations predictive usefulness. Chapters 11-12 follow Bg7 into the actual output heads and search. Chapters 13-15 teach how to inspect and challenge a proposed internal explanation.'))
story.append(para('No prior transformer vocabulary is required. The mathematics is introduced where the engine uses it. Read the worked arithmetic slowly; the short checks are intended to make you predict a change before reading the answer. The appendices provide the tensor ledger, calculations, solutions, glossary and exact source map.'))
story.append(para('Every printed chessboard is reconstructed from legal positions. Teaching activations are labeled ILLUSTRATIVE. Real plots name their measurement family. CODE / MEASURED, MATHEMATICS and HYPOTHESIS mark different kinds of evidence. The distinction is part of the explanation, not fine print.','objective'))
story.append(para('Measurement convention','section'))
story.append(para('Publication: 10 October 2026. Source and measurement snapshot: 9 October 2026.','small'))
story.append(para('Unless another recording is named, detailed Bg7 arithmetic uses the saved CPU diagnostic with float16-rounded trained EMA weights and float32 arithmetic. Native TensorRT search and the earlier GPU calibration are separate receipts. Block/head numbers are one-based in prose; neuron/unit IDs and array indices are zero-based.'))
story.append(para('Navigation','section'))
story.append(para('The contents entries and PDF outline link to chapters and sections. Source tags such as [S1] and [M2] link to the source appendix. The book is complete without the interactive atlas. Its editable manuscript, numeric receipts and reproducible figure scripts remain in the companion folder.'))
story.append(para('The central promise','section'))
story.append(para('By the end, you can calculate where a query, key, bias, value message and policy preference come from. You can also distinguish a demonstrated numerical mechanism from an appealing chess interpretation that still needs a controlled experiment.'))
story.append(PageBreak());story.append(para('Contents','chapter'))
toc=TableOfContents();toc.levelStyles=[ParagraphStyle('toc0',fontName='Sans-Bold',fontSize=9.3,leading=12.5,spaceBefore=9,textColor=INK),ParagraphStyle('toc1',fontName='Sans',fontSize=8.4,leading=11,spaceBefore=1,leftIndent=13,textColor=MUTED)];toc.dotsMinLevel=0;story.append(toc);story.append(PageBreak())
figure_index=[];equation_index=[];headingcount=0;answer_mode=False
for file in sorted((ROOT/'manuscript').glob('*.md')):
 text=file.read_text();paras=re.split(r'\n\s*\n',text.strip())
 for block in paras:
  lines=block.splitlines()
  # Headings and objectives can share a Markdown paragraph.
  while lines and (lines[0].startswith('# ') or lines[0].startswith('## ') or lines[0].startswith('@objectives ')):
   line=lines.pop(0)
   if line.startswith('@objectives '):story.append(para('<b>Learning objectives.</b> '+line[12:],'objective') if False else Paragraph('<b>Learning objectives.</b> '+fmt(line[12:]),styles['objective']));continue
   chapter=line.startswith('# ');title=line[2:] if chapter else line[3:]
   if chapter:answer_mode=title.startswith('Appendix C.')
   if chapter and headingcount:story.append(PageBreak())
   p=para(title,'chapter' if chapter else 'section');p._bookmark='heading-'+str(headingcount);p._level=0 if chapter else 1;headingcount+=1;story.append(p)
  if not lines:continue
  block=' '.join(lines)
  if block=='@pagebreak':story.append(PageBreak());continue
  if block.startswith('@figure '):
   key,cap=block[8:].split('|',1);path=FIG/(key+'.png');im=PILImage.open(path);ww=CW;hh=ww*im.height/im.width
   max_height=425 if key=='architecture' else 330
   if hh>max_height:ww*=max_height/hh;hh=max_height
   num=len(figure_index)+1;img=Image(str(path),ww,hh);img.hAlign='CENTER';img._figkey=key
   caption=Paragraph(f'<b>Figure {num}.</b> '+fmt(cap),styles['caption'])
   group=[img,caption];img.spaceBefore=5
   if story and isinstance(story[-1],Paragraph) and getattr(story[-1],'_level',None)==1:group.insert(0,story.pop())
   story.append(KeepTogether(group));figure_index.append({'number':num,'key':key,'caption':cap,'path':str(path)});continue
  if block.startswith('@equation '):
   key=block[10:].strip();path=FIG/('eq_'+key+'.png');im=PILImage.open(path);# font at 15 pt before downscale; cap maximum width
   ww=min(CW-16,im.width/260*72);hh=ww*im.height/im.width;img=Image(str(path),ww,hh);img.hAlign='LEFT';img._figkey='eq_'+key
   eqn=len(equation_index)+1;story.append(KeepTogether([Spacer(1,4),img,Paragraph(f'({eqn})',styles['caption'])]));equation_index.append({'number':eqn,'key':key});continue
  if block.startswith('@table '):
   tab=get_table(block[7:].strip());tab.spaceBefore=6
   if story and isinstance(story[-1],Paragraph) and getattr(story[-1],'_level',None)==1:
    section=story.pop();section.keepWithNext=False;story += [CondPageBreak(115),section]
   story += [tab,Spacer(1,12)];continue
  if block.startswith('@check '):
   n,q=block[7:].split('|',1);p=Paragraph(f'<b>Pause and predict {n}.</b> '+fmt(q)+f' <font color="#6B7780">Answer: Appendix C.</font>',styles['check']);story.append(KeepTogether([Spacer(1,6),p]));continue
  if block.startswith('@gameprefix'):
   prefix=(DATA/'prefix.txt').read_text();story.append(para(prefix,'small'));continue
  # Source anchors on corresponding defining paragraphs, including A1 inside source appendix.
  anchor=''
  m=re.match(r'\*\*([STMRGA][0-9]+)(?:\s[-.]|\.)',block)
  if m and m[1] in source_ids:anchor=f'<a name="source-{m[1]}"/>'
  body_style='reference' if m and m[1].startswith('R') else 'answer' if answer_mode else 'body'
  story.append(Paragraph(anchor+fmt(block),styles[body_style]))
# Bibliographic metadata + build.
doc=Book(str(OUT),pagesize=(W,H),leftMargin=M,rightMargin=M,topMargin=51,bottomMargin=47,title='Inside VEGAM: From Chess Pieces to Learned Intelligence',author='Remmy Victor',subject='Source-verified illustrated textbook of the VEGAM neural chess engine',pageCompression=1)
doc.multiBuild(story,maxPasses=5)
metadata={'pdf':str(OUT),'figures':figure_index,'equations':equation_index,'headings':doc.page_records,'figure_pages':doc.figure_pages,'manuscript_words':sum(len(p.read_text().split()) for p in (ROOT/'manuscript').glob('*.md')),'pdf_sha256':hashlib.sha256(OUT.read_bytes()).hexdigest()}
(DATA/'build.json').write_text(json.dumps(metadata,indent=2));print(json.dumps({'output':str(OUT),'pages':doc.page,'figures':len(figure_index),'equations':len(equation_index),'words':metadata['manuscript_words'],'bytes':OUT.stat().st_size}))
