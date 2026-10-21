"""Freeze named website models, new F candidate, and matched untrained baselines."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tournament.controllers import legacy_types
from model import MuZero,Config

def main():
 p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--lab',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);entries=[];types=legacy_types(a.site)
 def add(id,family,file,**kw):
  entries.append(dict(id=id,family=family,checkpoint=str(file.resolve()),sha256=hashlib.sha256(file.read_bytes()).hexdigest(),**kw))
 for family,seed,prefix in [('lana',20261022,'lana'),('legacy_muzero',20261023,'muzero'),('dreamer4_unofficial',20261024,'dreamer4')]:
  torch.manual_seed(seed);factory,_=types[family];m=factory('cpu') if family=='lana' else factory()
  if family=='lana':
   expected=json.loads((a.site/'public/lana/manifest.json').read_text())['initial_weights_sha256']
   assert hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in m.state_dict().values())).hexdigest()==expected
  path=a.output/(prefix+'-untrained.npz');np.savez_compressed(path,**{k:v.detach().numpy() for k,v in m.state_dict().items()})
  add(prefix+'-untrained',family,path,initialization_seed=seed,initialization='reconstructed from training seed; Lana digest checked against browser manifest',trained=False)
  for epoch in (100,1000):add(prefix+'-'+str(epoch),family,a.site/'.training/arena-baselines-001'/f'{prefix}-{epoch}.npz',trained=True,training_label=f'{epoch} epochs (legacy counter)')
 old=a.lab/'.local/runpod/muzero-independent-002/artifacts/muzero-speed-10000-002';meta=json.loads((old/'manifest.json').read_text())
 torch.manual_seed(meta['arguments']['seed']);m=MuZero(Config(**meta['config']));path=a.output/'independent8-untrained.npz';np.savez_compressed(path,**{k:v.detach().numpy() for k,v in m.state_dict().items()})
 add('independent8-untrained','independent_muzero',path,config=meta['config'],root_candidates=0,trained=False,initialization='reconstructed from training seed',initialization_seed=meta['arguments']['seed'])
 add('independent-10000','independent_muzero',old/'weights.npz',config=meta['config'],root_candidates=0,trained=True,training_label='10000 episodes; legacy cadence during training')
 run=a.lab/'.local/runpod/muzero-selfplay-001/artifacts/selfplay-E';meta=json.loads((run/'manifest.json').read_text())
 for name,file in [('selfplay32-untrained',run/'initial.npz'),('selfplay-E1024',run/'checkpoint-1024.npz'),('selfplay-F1024',a.lab/'.local/runpod/muzero-selfplay-F/artifacts/selfplay-F/checkpoint-1024.npz')]:
  add(name,'independent_muzero',file,config=meta['config'],root_candidates=8,trained='untrained' not in name,initialization='saved initial weights shared by E/F' if 'untrained' in name else None)
 manifest=a.site/'public/arena-models/manifest.json';browser=json.loads(manifest.read_text())['models']
 # Trained baseline and independent weights must be the exact deployed exports' source.
 for entry in entries:
  key=entry['id']+'-'+('policy' if entry['family']=='dreamer4_unofficial' else 'initial')
  if entry['id']=='independent-10000':key='muzero-independent-10000-initial'
  if entry['id']=='selfplay-E1024':key='muzero-selfplay-1024-initial'
  if key in browser:assert entry['sha256']==browser[key]['checkpoint_sha256']
 (a.output/'catalog.json').write_text(json.dumps(dict(models=entries,scope='nine homepage choices, F1024 and matching untrained inference configurations; intermediate checkpoints excluded'),indent=2)+'\n')
 print(json.dumps([dict(id=e['id'],family=e['family'],trained=e['trained']) for e in entries],indent=2))
if __name__=='__main__':main()
