"""Verify recovered real training states against authenticated remote manifests.

Checks recovery integrity and saved fields, not split/resume numerical equality.
"""
import argparse,gc,hashlib,json
from pathlib import Path
import numpy as np
import torch
FIELDS={'arguments','environment','source_files','torch_version','initial','model','optimizer','totals','replay','numpy_rng','torch_rng','cuda_rng'}
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()
def finite(value):
 if isinstance(value,torch.Tensor) and (value.is_floating_point() or value.is_complex()):
  if not torch.isfinite(value).all():raise ValueError('Nonfinite saved tensor')
 elif isinstance(value,dict):
  for x in value.values():finite(x)
 elif isinstance(value,(list,tuple)):
  for x in value:finite(x)
def verify(directory,remote,source,expected_games):
 directory=Path(directory);torch.set_num_threads(2)
 for name,item in remote.items():
  path=(directory/name).resolve()
  if not path.is_relative_to(directory.resolve()) or path.stat().st_size!=item['bytes'] or digest(path)!=item['sha256']:raise ValueError('Recovered file mismatch: '+name)
 manifest=json.loads((directory/'manifest.json').read_text());report=json.loads((directory/'report.json').read_text());rows=[]
 for count in expected_games:
  name=f'full-state/games-{count}-updates-{count}.pt';state=torch.load(directory/name,map_location='cpu',weights_only=True)
  assert set(state)==FIELDS
  assert state['source_files']==source['files'] and state['environment']==manifest['environment']
  assert state['torch_version']==manifest['torch'] and state['arguments']==manifest['arguments']
  assert state['totals']['games']==count and state['totals']['optimizer_updates']==count
  assert len(state['replay'])==3 and state['optimizer']['state'] and len(state['cuda_rng'])==1
  finite(state)
  with np.load(directory/'initial.npz',allow_pickle=False) as z:
   assert set(z.files)==set(state['initial'])
   for k in z.files:assert np.array_equal(z[k],state['initial'][k].numpy())
  if count==max(expected_games):
   with np.load(directory/'weights.npz',allow_pickle=False) as z:
    assert set(z.files)==set(state['model'])
    for k in z.files:assert np.array_equal(z[k],state['model'][k].numpy())
  rows.append({'games':count,'optimizer_updates':count,'bytes':remote[name]['bytes'],'sha256':remote[name]['sha256'],'replay_blocks':3,'safe_load':True,'source_bound':True,'finite_tensors':True});del state;gc.collect()
 assert report['requested_run_completed'] and report['games']==max(expected_games)
 return {'verified':True,'states':rows,'files_verified':len(remote),'scope':'Actual recovery integrity and state fields; no GPU split-resume equality or strength claim'}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--remote-manifest',type=Path,required=True);p.add_argument('--source-manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 result=verify(a.directory,json.loads(a.remote_manifest.read_text()),json.loads(a.source_manifest.read_text()),[512,640]);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
