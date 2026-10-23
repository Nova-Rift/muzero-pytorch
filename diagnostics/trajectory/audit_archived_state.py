import json,hashlib,pathlib,glob,torch,numpy as np
import argparse
p=argparse.ArgumentParser();p.add_argument('--research-root',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();root=a.research_root;out=a.output;out.mkdir(parents=True,exist_ok=True)
archive=root/'.local/research-history'; evidence=[]
for n in [8,9,10,11,15]:
 records=[(json.loads(p.read_text()),p) for p in (archive/'records').glob(f'iteration-{n}-*.json')]; data,record=max(records,key=lambda x:x[0]['archived_at'])
 seen=set()
 for f in data['files']:
  name=pathlib.Path(f['path']).name
  wanted=(name in ['report.md','resume-audit.json','full-state-audit.json'] and '/loop-' not in f['path']) or (n==8 and '/artifacts/arena-loop-001-M8/' in f['path'] and name in ['resume-state.pt','manifest.json','checkpoint-2048.npz'])
  if not wanted or f['sha256'] in seen:continue
  seen.add(f['sha256']); obj=archive/'objects'/f['sha256']; h=hashlib.sha256()
  with obj.open('rb') as r:
   for b in iter(lambda:r.read(8*1024*1024),b''):h.update(b)
  assert h.hexdigest()==f['sha256'];evidence.append(dict(f,record=str(record),hash_verified=True))
state_entry=next(f for f in evidence if f['path'].endswith('/resume-state.pt'))
s=torch.load(archive/'objects'/state_entry['sha256'],map_location='cpu',weights_only=True)
torch.set_num_threads(1)
tensors=0;elements=0
def inspect(x):
 global tensors,elements
 if isinstance(x,torch.Tensor):
  tensors+=1;elements+=x.numel()
  if x.is_floating_point():assert torch.isfinite(x).all()
 elif isinstance(x,dict):
  for v in x.values():inspect(v)
 elif isinstance(x,(list,tuple)):
  for v in x:inspect(v)
inspect(s)
cp=next(f for f in evidence if f['path'].endswith('/checkpoint-2048.npz')); z=np.load(archive/'objects'/cp['sha256']); assert set(z.files)==set(s['model'])
assert all(np.array_equal(z[k],v.numpy()) for k,v in s['model'].items())
result=dict(state_sha256=state_entry['sha256'],state_bytes=state_entry['bytes'],fields=list(s),arguments=s['arguments'],totals=s['totals'],torch_version=s['torch_version'],local_loader_torch_version=str(torch.__version__),source_files=s['source_files'],environment=s['environment'],replay=[{k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in b.items()} for b in s['replay']],optimizer_state_entries=len(s['optimizer']['state']),optimizer_steps=sorted(set(float(v['step']) for v in s['optimizer']['state'].values())),cuda_rng_streams=len(s['cuda_rng']),torch_rng_bytes=s['torch_rng'].numel(),numpy_rng_generator=s['numpy_rng']['bit_generator'],finite_tensor_count=tensors,tensor_elements=elements,model_exactly_matches_checkpoint_2048=True,scope='Read-only CPU weights_only load, structure/finiteness and archival hashes; no training, GPU resume or transfer validation.')
(out/'state-audit.json').write_text(json.dumps(result,indent=2)+'\n');(out/'evidence.json').write_text(json.dumps(evidence,indent=2)+'\n'); print(json.dumps({k:v for k,v in result.items() if k not in ['source_files','environment','replay','arguments']}))
