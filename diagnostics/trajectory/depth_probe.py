"""Frozen search-depth correctness/latency probe; no optimizer or game claims."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch

def main():
 p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--simulations',type=int,choices=[32,64,128],required=True);p.add_argument('--device',choices=['cpu','cuda'],required=True);p.add_argument('--count',type=int,default=2);a=p.parse_args();assert 1<=a.count<=64
 sys.path.insert(0,str(a.experiment/'model'));from tournament.run import load_model
 from batched_search_exploration import search_batch
 from arena.frozen_search import FrozenSearch
 torch.set_num_threads(2);torch.manual_seed(210301)
 e=json.loads((a.experiment/f'M2048-s{a.simulations}-E1024.json').read_text())['models'][0];e['checkpoint']=str(a.experiment/'weights'/Path(e['checkpoint']).name);assert hashlib.sha256(Path(e['checkpoint']).read_bytes()).hexdigest()==e['sha256']
 m=load_model(e,str(a.experiment/'site'),a.device);before={k:v.clone() for k,v in m.state_dict().items()};x=torch.rand(a.count,m.config.observations,device=a.device);roots=torch.stack([torch.randperm(153,device=a.device)[:8] for _ in range(a.count)])
 started=time.monotonic()
 if a.device=='cuda':torch.cuda.reset_peak_memory_stats()
 with torch.no_grad():
  expected=search_batch(m,x,static=True,root_candidates=roots)
  frozen=FrozenSearch(m,a.count,8);actual=frozen(x,roots)
  for lhs,rhs in zip(expected,actual):assert torch.allclose(lhs,rhs,atol=1e-5,rtol=1e-5)
  if a.device=='cpu':
   dynamic=search_batch(m,x,static=False,root_candidates=roots)
   for lhs,rhs in zip(expected,dynamic):assert torch.allclose(lhs,rhs,atol=1e-5,rtol=1e-5)
  assert torch.isfinite(actual[0]).all() and torch.isfinite(actual[1]).all();assert torch.allclose(actual[0].sum(-1),torch.ones(a.count,device=a.device))
  if a.device=='cuda':torch.cuda.synchronize()
  t=time.monotonic()
  for _ in range(3):frozen(x,roots)
  if a.device=='cuda':torch.cuda.synchronize()
  seconds=(time.monotonic()-t)/3
 for k,v in m.state_dict().items():assert torch.equal(v,before[k])
 report=dict(simulations=a.simulations,roots=8,count=a.count,device=a.device,seconds_per_search_batch=seconds,total_seconds=time.monotonic()-started,maximum_policy_error=float((expected[0]-actual[0]).abs().max()),maximum_value_error=float((expected[1]-actual[1]).abs().max()),weights_unchanged=True,torch=str(torch.__version__),scope='Synthetic inputs and fixed roots; not arena strength or end-to-end throughput',checkpoint_sha256=e['sha256'])
 if a.device=='cuda':
  assert seconds<=.6, 'Measured search too slow for prepared evaluation budget'
  free,total=torch.cuda.mem_get_info();report.update(gpu=torch.cuda.get_device_name(),free_bytes=free,total_bytes=total,peak_allocated_bytes=torch.cuda.max_memory_allocated());assert free/total>=.1
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
