"""Resumable two-agent round robin using canonical JS rules and frozen policies."""
import argparse,hashlib,itertools,json,math,sys,time
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tournament.controllers import load_model,controller
from arena.environment import Pool

def wilson(w,n):
 z=1.96;p=w/n;den=1+z*z/n;c=(p+z*z/(2*n))/den;d=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return [c-d,c+d]

def play(site,entries,models,seeds,game_offset,workers,limit,lana_microbatch=32):
 pool=Pool(site,workers=min(workers,len(seeds)),agents=2,cadence='homepage')
 seats=(np.arange(len(seeds))+game_offset)%2
 labels=[[(int(seed),int(seats[i] if side==0 else 1-seats[i])) for i,seed in enumerate(seeds)] for side in (0,1)]
 started=time.perf_counter();controllers=[controller(e,m,l,site,lana_microbatch) for e,m,l in zip(entries,models,labels)]
 setup=time.perf_counter()-started;state=pool.reset(seeds);finished=np.zeros(len(seeds),bool);steps=np.zeros(len(seeds),int)
 inference=0.;environment=0.;rollout=time.perf_counter()
 try:
  for _ in range(limit):
   actions=np.zeros((len(seeds),11),np.int64);tick=time.perf_counter()
   for side,c in enumerate(controllers):
    ids=seats if side==0 else 1-seats;rows=np.arange(len(seeds))
    actions[rows,ids]=c.decide(state['observation'][rows,ids].tolist(),(state['alive'][rows,ids]&~finished).tolist())
   inference+=time.perf_counter()-tick;actions[finished]=-1;steps+=~finished
   tick=time.perf_counter();state=pool.step(actions);environment+=time.perf_counter()-tick
   finished|=state['finished']
   if finished.all():break
  records=[dict(seed=int(seed),seat_a=int(seats[i]),winner=int(state['winner'][i]),finished=bool(finished[i]),steps=int(steps[i])) for i,seed in enumerate(seeds)]
  return dict(records=records,setup_seconds=setup,rollout_seconds=time.perf_counter()-rollout,inference_seconds=inference,environment_seconds=environment)
 finally:pool.close()

def main():
 p=argparse.ArgumentParser();p.add_argument('--catalog',type=Path,required=True);p.add_argument('--site',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu');p.add_argument('--games',type=int,default=128);p.add_argument('--parallel',type=int,default=128);p.add_argument('--workers',type=int,default=4);p.add_argument('--limit',type=int,default=256);p.add_argument('--seed',type=int,default=51000000);p.add_argument('--models',nargs='+');p.add_argument('--lana-microbatch',type=int,default=32);a=p.parse_args()
 if a.games<2 or a.games%2 or a.parallel<1 or a.lana_microbatch<1:raise ValueError('Need an even game count and positive batch')
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
 catalog=json.loads(a.catalog.read_text());entries=[e for e in catalog['models'] if not a.models or e['id'] in a.models]
 if a.models and set(a.models)!={e['id'] for e in entries}:raise ValueError('Unknown model id')
 if len(entries)<2:raise ValueError('Need two models')
 for e in entries:
  if hashlib.sha256(Path(e['checkpoint']).read_bytes()).hexdigest()!=e['sha256']:raise ValueError('Changed checkpoint')
 a.output.mkdir(parents=True,exist_ok=True)
 protocol=dict(catalog_sha256=hashlib.sha256(a.catalog.read_bytes()).hexdigest(),models=[{k:v for k,v in e.items() if k!='checkpoint'} for e in entries],games=a.games,parallel=a.parallel,lana_microbatch=a.lana_microbatch,workers=a.workers,limit=a.limit,seed=a.seed,device=a.device,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),controller_sha256=hashlib.sha256(Path(__file__).with_name('controllers.py').read_bytes()).hexdigest(),environment={f:hashlib.sha256((Path(a.site)/f).read_bytes()).hexdigest() for f in ['src/sandboxCombat.ts','src/sandboxMotion.ts','src/lana/observation.ts']},cadence='five .05s frames per decision',sampling='independent PCG64 per game and seat; no root noise',scope='two active agents, fixed spawn geometry; seeds vary policy sampling, not worlds')
 protocol['lana_batch_adapter_sha256']=hashlib.sha256(Path(__file__).with_name('lana_batch.py').read_bytes()).hexdigest()
 protocol['runtime']={'torch':str(torch.__version__),'gpu':torch.cuda.get_device_name() if a.device=='cuda' else None}
 protocol['legacy_sources']={str(f.relative_to(Path(a.site))):hashlib.sha256(f.read_bytes()).hexdigest() for folder in [Path(a.site)/'scripts/baselines',Path(a.site)/'.training/lana-hybrid-003/source/survival_agent'] for f in folder.rglob('*.py')}
 protocol['legacy_sources'].update({f:hashlib.sha256((Path(a.site)/f).read_bytes()).hexdigest() for f in ['scripts/evaluate-lana-100.py','scripts/lana_large.py','scripts/lana_device.py','scripts/lana_batched_planning.py']})
 frozen=a.output/'protocol.json'
 if frozen.exists():
  if json.loads(frozen.read_text())!=protocol:raise ValueError('Protocol changed; do not resume')
 else:frozen.write_text(json.dumps(protocol,indent=2)+'\n')
 summaries=[]
 for pair_index,(left,right) in enumerate(itertools.combinations(entries,2)):
  key=left['id']+'__'+right['id'];records=[];total=0.;models=None
  for offset in range(0,a.games,a.parallel):
   file=a.output/f'{key}-{offset}.json'
   if file.exists():result=json.loads(file.read_text())
   else:
    if models is None:models=[load_model(e,a.site,a.device) for e in (left,right)]
    count=min(a.parallel,a.games-offset);seeds=range(a.seed+pair_index*100000+offset,a.seed+pair_index*100000+offset+count)
    result=play(a.site,[left,right],models,seeds,offset,a.workers,a.limit,a.lana_microbatch)
    temp=file.with_suffix('.tmp');temp.write_text(json.dumps(result));temp.replace(file)
   records+=result['records'];total+=result['setup_seconds']+result['rollout_seconds']
  wins=sum(r['finished'] and r['winner']==r['seat_a'] for r in records);losses=sum(r['finished'] and r['winner']==1-r['seat_a'] for r in records);cutoffs=sum(not r['finished'] for r in records)
  summary=dict(a=left['id'],b=right['id'],games=len(records),wins_a=wins,wins_b=losses,cutoffs=cutoffs,draws=len(records)-wins-losses-cutoffs,wilson95_a=wilson(wins,len(records)),wilson95_b=wilson(losses,len(records)),seconds=total,games_per_second=len(records)/total)
  summaries.append(summary);(a.output/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n');print(json.dumps(summary),flush=True)
  del models
  if a.device=='cuda':torch.cuda.empty_cache()
if __name__=='__main__':main()
