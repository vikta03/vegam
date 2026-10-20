"""Cheap structural, numerical and artifact validation; no engine execution."""
import re,json,hashlib,collections,math
from pathlib import Path
from pypdf import PdfReader
import pdfplumber
ROOT=Path(__file__).resolve().parent;WS=ROOT.parent;data=ROOT/'data';build=json.loads((data/'build.json').read_text());pdf=Path(build['pdf']);reader=PdfReader(pdf)
facts=json.loads((data/'facts.json').read_text());m=json.loads((data/'measurements.json').read_text());source=json.loads((data/'publication-source-index.json').read_text());issues=[]
for key,v in source.items():
 p=WS/v['path']
 if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=v['sha256']:issues.append('Changed source '+key)
for v in build['figures']:
 if not Path(v['path']).exists():issues.append('Missing figure '+v['key'])
assert facts['validation']['legal_pgn_plies']==201
assert facts['validation']['rook_exits_verified']==3
assert facts['validation']['input_projection_max_error']<1e-6
assert facts['validation']['toy_gradient_error']<1e-8
assert abs(sum(v['gap_contribution'] for v in m['readout'])-m['readout_actual_gap'])<1e-6
assert len(m['interventions'])==116
assert all(v['best']=='Bg7' for v in m['interventions'][:96])
assert abs(sum(v['policy'] for v in m['baseline']['moves'])-1)<1e-9
assert len(m['baseline']['moves'])==22
pages=[];bounds=[]
with pdfplumber.open(pdf) as doc:
 for n,p in enumerate(doc.pages,1):
  text=p.extract_text() or '';words=len(text.split());pages.append({'page':n,'words':words,'text_prefix':text[:100]})
  if not words:issues.append(f'Blank text page {n}')
  for ch in p.chars:
   if ch['x0']<-1 or ch['x1']>541 or ch['top']<-1 or ch['bottom']>721:bounds.append({'page':n,'text':ch['text'],'box':[ch['x0'],ch['top'],ch['x1'],ch['bottom']]})
alltext='\n'.join(p.extract_text() or '' for p in reader.pages)
for marker in ['@figure','@table','@equation','@objectives','@pagebreak','TODO','PLACEHOLDER','@check']:
 if marker in alltext:issues.append('Unprocessed marker '+marker)
for i in range(1,25):
 if f'Pause and predict {i}.' not in alltext:issues.append('Missing exercise '+str(i))
# Numerics printed by the manuscript, recalculated independently.
prob=m['baseline_focus']['bg7_policy']/m['baseline_focus']['bf4_policy']
report={'status':'passed' if not issues and not bounds else 'review_required','pages':len(reader.pages),'figures':len(build['figures']),'equations':len(build['equations']),'words':build['manuscript_words'],'issues':issues,'out_of_page_text':bounds,'short_pages':[p for p in pages if p['words']<100],'page_word_counts':pages,'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),'readout_residual':sum(v['gap_contribution'] for v in m['readout'])-m['readout_actual_gap'],'policy_ratio':prob,'source_files_verified':len(source),'figure_files_verified':len(build['figures'])}
(data/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['page_word_counts','short_pages']},indent=2));print('SHORT PAGES',report['short_pages'])
