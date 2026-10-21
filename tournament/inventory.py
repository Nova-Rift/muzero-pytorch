"""Publish safe model sizes and inference budgets, never checkpoint paths."""
import argparse,hashlib,json,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tournament.controllers import load_model
p=argparse.ArgumentParser();p.add_argument('--catalog',type=Path,required=True);p.add_argument('--site',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise ValueError('Preserve previous inventory')
torch.set_num_threads(1);rows=[]
for e in json.loads(a.catalog.read_text())['models']:
 if hashlib.sha256(Path(e['checkpoint']).read_bytes()).hexdigest()!=e['sha256']:raise ValueError('Changed checkpoint: '+e['id'])
 m=load_model(e,a.site,'cpu');family=e['family']
 row={k:v for k,v in e.items() if k!='checkpoint'}
 row['parameters']=sum(p.numel() for p in m.parameters());row['buffer_elements']=sum(b.numel() for b in m.buffers())
 row['inference']={'simulations':e['config']['simulations'],'root_candidates':e['root_candidates']} if family=='independent_muzero' else ({'simulations':8,'horizon':3} if family in ('lana','legacy_muzero') else {'mode':'policy-only; unofficial Dreamer4'})
 rows.append(row)
a.output.write_text(json.dumps({'models':rows,'limits':'Historical training budgets/agent counts differ; matching parameter counts do not imply equal training compute.'},indent=2)+'\n')
print(json.dumps([{'id':r['id'],'parameters':r['parameters']} for r in rows],indent=2))
