"""Real historical-backend parity with explicit roots/noise on saved observations."""
import argparse,copy,hashlib,json,sys
from dataclasses import replace
from pathlib import Path
import numpy as np,torch
from split_search import split_search
from split_controller import SplitController

def main():
 p=argparse.ArgumentParser()
 for n in ('source','state','catalog','output'):p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--site',type=Path);a=p.parse_args();sys.path.insert(0,str(a.source))
 from model import Config,MuZero
 from arena.train_selfplay import replay_batch
 from batched_search_exploration import search_batch
 torch.set_num_threads(2);s=torch.load(str(a.state),weights_only=True,map_location='cpu',mmap=True)
 for n in ('model.py','arena/train_selfplay.py','arena/history.py','batched_search_exploration.py'):assert hashlib.sha256((a.source/n).read_bytes()).hexdigest()==s['source_files'][n]
 e=next(e for e in json.loads(a.catalog.read_text())['models'] if e['id']=='M2048');m=MuZero(Config(**e['config']));m.load_state_dict(s['model']);m.eval()
 b={k:v.numpy() for k,v in s['replay'][-1].items()};obs=replay_batch(b,np.random.default_rng(190001),2,64,5,'cpu')[0]
 torch.manual_seed(190002);noise=torch.distributions.Dirichlet(torch.full((2,4,m.config.actions),.3)).sample();roots=torch.stack([torch.randperm(m.config.actions)[:2] for _ in range(8)]).reshape(2,4,2)
 before={k:v.clone() for k,v in m.state_dict().items()};rng=torch.get_rng_state().clone()
 actual=split_search(m,obs,search_batch,root_noise=noise,root_candidates=roots)
 assert torch.equal(rng,torch.get_rng_state());view=copy.copy(m);view.config=replace(m.config,simulations=8);ps=[];vs=[]
 for i in range(2):
  p,v=search_batch(view,obs[i:i+1].repeat(4,1),root_noise=noise[i],root_candidates=roots[i]);ps.append(p.mean(0));vs.append(v.mean())
 expected=(torch.stack(ps),torch.stack(vs));errors=[]
 for x,y in zip(actual,expected):
  errors.append(float((x-y).abs().max()));torch.testing.assert_close(x,y,atol=1e-6,rtol=1e-6)
 assert m.config.simulations==32 and not m.training
 assert all(torch.equal(v,before[k]) for k,v in m.state_dict().items())
 from arena.history import DeviceHistory
 raw=obs[:,:4096].reshape(2,64,64)[:,-1].numpy()
 labels=[(190100,0),(190101,1)]
 grouped=SplitController(m,labels,DeviceHistory,search_batch)
 singles=[SplitController(m,[label],DeviceHistory,search_batch) for label in labels]
 actions=[]
 for tick in range(3):
  group=grouped.decide(raw.tolist(),[True,True])
  separate=[c.decide([raw[i].tolist()],[True])[0] for i,c in enumerate(singles)]
  assert group==separate
  actions.append(group)
 assert all(torch.equal(v,before[k]) for k,v in m.state_dict().items())
 report=dict(passed=True,controller_group_actions=actions,controller_per_stream_parity=True,observations=2,groups=4,policy_max_error=errors[0],value_max_error=errors[1],observation_sha256=[hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in obs],weights_unchanged=True,explicit_controls_rng_unchanged=True,new_games=0,optimizer_updates=0,scope='CPU explicit-control grouping parity only; not GPU, arena compatibility or exact training resume')
 if a.site:
  from arena.environment import Pool
  pool=Pool(str(a.site),workers=1,agents=2,cadence='homepage')
  try:
   state=pool.reset([190201,190202]);assert (state['alive'].sum(1)==2).all()
   controllers=[SplitController(m,[(190201,seat),(190202,seat)],DeviceHistory,search_batch) for seat in (0,1)]
   steps=0
   for tick in range(4):
    commands=np.zeros((2,11),dtype=np.int64)
    for seat,c in enumerate(controllers):commands[:,seat]=c.decide(state['observation'][:,seat].tolist(),state['alive'][:,seat].tolist())
    assert ((commands[:,:2]>=0)&(commands[:,:2]<153)).all()
    state=pool.step(commands);steps+=1
   report['arena_smoke']=dict(two_player_streams=2,decisions_per_stream=steps,completed_matchups=0,scope='Truncated wiring test, not scored games',environment={f:hashlib.sha256((a.site/f).read_bytes()).hexdigest() for f in ['src/sandboxCombat.ts','src/sandboxMotion.ts','src/lana/observation.ts']})
  finally:pool.close()
 a.output.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
