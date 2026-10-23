"""Exploratory frozen value diagnostics on recovered candidate histories; no learning."""
import argparse,json,sys
from pathlib import Path
import numpy as np,torch
p=argparse.ArgumentParser();p.add_argument('--model-source',type=Path,required=True);p.add_argument('--catalog',type=Path,required=True);p.add_argument('--traces',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu');p.add_argument('--self-test',action='store_true');a=p.parse_args();sys.path.insert(0,str(a.model_source));from arena.history import DeviceHistory
from tournament.controllers import load_model
from model import scalar
torch.set_num_threads(2)
if a.self_test:
 h=DeviceHistory(2,64,'cpu');obs=np.zeros((2,64),dtype=np.float32);obs[:,0]=[.1,.2];x=h.observe(obs);assert x.shape==(2,4159);assert torch.allclose(x[:,4032],torch.tensor([.1,.2]));h.acted([9,18]);x=h.observe(obs);assert torch.allclose(x[:,-1],torch.tensor([9/153,18/153]));print('History alignment check passed');sys.exit(0)
entries=[e for e in json.loads(a.catalog.read_text())['models'] if e['id'] in ['M1024','M2048']];assert len(entries)==2
models=[load_model(e,'',a.device) for e in entries];out=[]
with torch.no_grad():
 for file in sorted(a.traces.glob('*-traces/batch-001.jsonl')):
  rows=[json.loads(x) for x in file.read_text().splitlines()];n=len(rows[0]['seeds']);history=DeviceHistory(n,64,a.device);samples=[[] for _ in range(n)];indices=np.arange(n);seats=indices%2
  for before,after in zip(rows,rows[1:]):
   obs=np.asarray(before['observation'],dtype=np.float32)[indices,seats];x=history.observe(obs);predictions=[]
   for model in models:
    _,_,v=model.initial(x);predictions.append(scalar(v).cpu().tolist())
   for i in range(n):
    if not before['finished'][i]:samples[i].append([predictions[0][i],predictions[1][i]])
   actions=np.asarray(after['actions'])[indices,seats];history.acted(actions)
  for i,seed in enumerate(rows[0]['seeds']):
   final=next((row for row in rows if row['finished'][i]),rows[-1]);finished=bool(final['finished'][i]);win=int(finished and final['winner'][i]==int(seats[i]));values=np.asarray(samples[i]);assert values.size and np.isfinite(values).all()
   out.append(dict(trace=file.parent.name,seed=seed,candidate_seat=int(seats[i]),finished=finished,terminal_win=win if finished else None,decisions=len(values),first_values=values[0].tolist(),last_pretransition_values=values[-1].tolist(),mean_values=values.mean(0).tolist()))
report=dict(iteration=15,model_order=[e['id'] for e in entries],games=len(out),new_games=0,scope='Exploratory cross-checkpoint values on saved on-policy histories. Cutoffs have unknown terminal outcomes, excluded from terminal calibration. Sparse selected trajectories; no causal or promotion inference.',records=out)
a.output.write_text(json.dumps(report,indent=2));print('Saved value diagnostics for',len(out),'recovered games')
