"""Explicit single-matchup split candidate evaluator; distinct diagnostic protocol."""
import argparse,hashlib,json,sys,time
from pathlib import Path

EXPECTED={'tournament/run.py':'9b08fc5f16928fa29098cb1dbab1f15e27d7950599216954f0c29cd764fdcec5','tournament/controllers.py':'7c50f9217d5268b7d826b72c664200eafeac45ba599edb1c6fe8db8558addee7'}
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def summarize(records,seeds,limit):
 if len(records)!=len(seeds) or len(set(seeds))!=len(seeds):raise ValueError('incomplete/duplicate games')
 out=dict(games=len(records),wins=0,opponent_wins=0,draws=0,timeouts=0)
 for i,(r,seed) in enumerate(zip(records,seeds)):
  if r['seed']!=seed or r['seat_a']!=i%2 or type(r['finished']) is not bool:raise ValueError('seed/seat/finished mismatch')
  if r['winner'] not in (-1,0,1) or not 1<=r['steps']<=limit:raise ValueError('invalid outcome')
  if not r['finished']:
   if r['winner']!=-1 or r['steps']!=limit:raise ValueError('invalid cutoff')
   out['timeouts']+=1
  elif r['winner']==r['seat_a']:out['wins']+=1
  elif r['winner']==1-r['seat_a']:out['opponent_wins']+=1
  else:out['draws']+=1
 assert sum(out[k] for k in ('wins','opponent_wins','draws','timeouts'))==out['games']
 return out

def main():
 p=argparse.ArgumentParser()
 for n in ('source','site','catalog','output'):p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--challenger',default='M2048');p.add_argument('--opponent',required=True);p.add_argument('--mode',choices=['split','baseline'],required=True)
 p.add_argument('--seed',type=int,required=True);p.add_argument('--games',type=int,default=8);p.add_argument('--parallel',type=int,default=4);p.add_argument('--limit',type=int,default=256);p.add_argument('--device',default='cpu');a=p.parse_args()
 if a.games<2 or a.games%2 or a.parallel<1 or a.limit<1:raise ValueError('invalid match budget')
 for f,h in EXPECTED.items():
  if digest(a.source/f)!=h:raise ValueError('frozen evaluator changed: '+f)
 sys.path.insert(0,str(a.source))
 import torch
 from tournament import run
 from arena.history import DeviceHistory
 from batched_search_exploration import search_batch
 from split_controller import SplitController
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
 entries=json.loads(a.catalog.read_text())['models'];left=dict(next(e for e in entries if e['id']==a.challenger));right=dict(next(e for e in entries if e['id']==a.opponent))
 if left['family']!='independent_muzero' or left['config']['simulations']!=32:raise ValueError('unsupported candidate')
 for e in (left,right):
  if digest(e['checkpoint'])!=e['sha256']:raise ValueError('checkpoint changed')
 left['id']=a.challenger+'-'+a.mode
 files={str(f.relative_to(a.source)):digest(f) for f in a.source.rglob('*.py')}
 own={f:digest(Path(__file__).with_name(f)) for f in ['evaluate_split.py','split_controller.py','split_search.py']}
 manifest=dict(schema=1,protocol_id='split-search-development-v1',mode=a.mode,challenger={k:v for k,v in left.items() if k!='checkpoint'},opponent={k:v for k,v in right.items() if k!='checkpoint'},seed=a.seed,games=a.games,parallel=a.parallel,limit=a.limit,device=a.device,source=files,adapter=own,environment={f:digest(a.site/f) for f in ['src/sandboxCombat.ts','src/sandboxMotion.ts','src/lana/observation.ts']},root_noise=False,split=dict(groups=4,simulations=8,roots=2) if a.mode=='split' else None,scope='Frozen development only; not compatible ladder import or training replication')
 a.output.mkdir(parents=True,exist_ok=False);(a.output/'protocol.json').write_text(json.dumps(manifest,indent=2)+'\n')
 models=[run.load_model(e,str(a.site),a.device) for e in (left,right)];original=run.controller
 def dispatch(e,m,labels,site):
  if e['id']==left['id'] and a.mode=='split':return SplitController(m,labels,DeviceHistory,search_batch)
  return original(e,m,labels,site)
 run.controller=dispatch;records=[];started=time.perf_counter()
 try:
  for offset in range(0,a.games,a.parallel):
   seeds=list(range(a.seed+offset,a.seed+min(offset+a.parallel,a.games)))
   result=run.play(str(a.site),[left,right],models,seeds,offset,1,a.limit)
   path=a.output/f'batch-{offset:04d}.json';temp=path.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(path);records+=result['records']
   print(json.dumps(dict(completed=len(records),target=a.games)),flush=True)
 finally:run.controller=original
 summary=dict(challenger=left['id'],opponent=right['id'],**summarize(records,list(range(a.seed,a.seed+a.games)),a.limit),seconds=time.perf_counter()-started)
 (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
