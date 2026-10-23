"""No-game correctness check for a proposed frozen planner-count comparison."""
import json,sys,hashlib,argparse
from pathlib import Path
import numpy as np,torch
p=argparse.ArgumentParser();p.add_argument('--model-source',type=Path,required=True);p.add_argument('--catalog',type=Path,required=True);p.add_argument('--trace',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.model_source));from tournament.controllers import load_model,IndependentController
torch.set_num_threads(2)
e=next(x for x in json.loads(a.catalog.read_text())['models'] if x['id']=='M2048');m=load_model(e,'','cpu');before={k:v.clone() for k,v in m.state_dict().items()};row=json.loads(a.trace.open().readline());obs=np.asarray([row['observation'][i][i%2] for i in range(2)],dtype=np.float32);labels=[(row['seeds'][i],i%2) for i in range(2)];results=[]
for count in [8,8,32]:
 c=IndependentController(m,labels,count);inner=c.search
 class Spy:
  def __call__(self,x,roots):
   assert roots.shape==(2,count)
   for row in roots:assert len(set(row.tolist()))==count
   policy,values=inner(x,roots);assert torch.isfinite(policy).all() and torch.isfinite(values).all();assert (policy.sum(-1)>0).all()
   self.roots=roots.clone();self.policy=policy.clone();return policy,values
 spy=Spy();c.search=spy;actions=c.decide(obs,[True,True]);assert all(a in roots for a,roots in zip(actions,spy.roots.tolist()));results.append(dict(root_candidates=count,actions=actions,roots=spy.roots.tolist()))
assert results[0]==results[1];assert all(torch.equal(v,before[k]) for k,v in m.state_dict().items())
a.output.write_text(json.dumps(dict(passed=True,repeated_baseline_exact=True,roots_unique=True,actions_within_roots=True,weights_unchanged=True,new_games=0,checks=results),indent=2));print('Root8 repeatability and root32 validity passed; no strength measured')
