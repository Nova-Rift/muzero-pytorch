"""Frozen policy adapters; no optimizer or scripted opponent."""
import json,runpy,sys
from pathlib import Path
import numpy as np,torch
from model import MuZero,Config
from arena.history import DeviceHistory
from arena.frozen_search import FrozenSearch

def legacy_types(site):
 scripts=Path(site)/'scripts'
 if str(scripts) not in sys.path:sys.path.insert(0,str(scripts))
 from baselines.muzero import MuZero as LegacyMu,Controller as MuController
 from baselines.dreamer4 import Dreamer4,Controller as DreamController
 # Import inference dependencies only; the old executable also imports
 # safetensors for training-checkpoint I/O, which this NPZ evaluator never uses.
 source=Path(site)/'.training/lana-hybrid-003/source'
 if str(source) not in sys.path:sys.path.insert(0,str(source))
 from lana_large import create_model
 from lana_batched_planning import BatchedController,section,stack,take,batched_mcts
 from survival_agent.hybrid.model import PlannerConfig
 class EvaluationController(BatchedController):
  def __init__(self,model,labels):
   super().__init__(model,PlannerConfig(simulations=8,horizon=3,root_noise=0.),count=len(labels))
   self.rng=[np.random.default_rng(23000000+seed*11+seat) for seed,seat in labels]
  @torch.no_grad()
  def decide(self,observations,alive):
   ids=[i for i,yes in enumerate(alive) if yes]
   if not ids:return [0]*self.count
   device=self.state.deter.device;obs=torch.tensor(observations,dtype=torch.float32,device=device);parts=[]
   for offset in range(0,self.count,32):
    end=min(offset+32,self.count)
    part,_,_=self.model.world_model.observe_step(section(self.state,offset,end),self.previous[offset:end],obs[offset:end],torch.full((end-offset,),self.first,dtype=torch.bool,device=device),stochastic=False)
    parts.append(part)
   self.state=stack(parts);self.first=False
   visits=batched_mcts(self.model,[take(self.state,i) for i in ids],self.config).numpy()
   actions=[0]*self.count
   for i,row in zip(ids,visits):
    probabilities=row.astype(np.float64)/row.sum()
    actions[i]=int(self.rng[i].choice(len(row),p=probabilities))
   self.previous=torch.tensor(actions,device=device,dtype=torch.long)
   return actions
 lana={'create_model':create_model,'EvaluationController':EvaluationController}
 return {'lana':(lana['create_model'],lana['EvaluationController']),
         'legacy_muzero':(LegacyMu,MuController),'dreamer4_unofficial':(Dreamer4,DreamController)}

def load_model(entry,site,device):
 if entry['family']=='independent_muzero':model=MuZero(Config(**entry['config']))
 else:
  factory,_=legacy_types(site)[entry['family']]
  model=factory('cpu') if entry['family']=='lana' else factory()
 with np.load(entry['checkpoint'],allow_pickle=False) as z:
  model.load_state_dict({k:torch.from_numpy(z[k]) for k in z.files},strict=True)
 return model.to(device).eval()

class IndependentController:
 def __init__(self,model,labels,candidates):
  self.model=model;self.count=len(labels);self.k=candidates
  self.history=DeviceHistory(self.count,64,next(model.parameters()).device)
  self.search=FrozenSearch(model,self.count,candidates)
  self.rng=[np.random.default_rng(23000000+s*11+seat) for s,seat in labels]
 @torch.no_grad()
 def decide(self,observations,alive):
  x=self.history.observe(np.asarray(observations,dtype=np.float32))
  roots=None
  if self.k:
   _,logits,_=self.model.initial(x);prior=logits.softmax(-1).cpu().numpy().astype(np.float64)
   chosen=[]
   for i,p in enumerate(prior):
    p/=p.sum()
    chosen.append(self.rng[i].choice(len(p),self.k,replace=False,p=p) if alive[i] else np.arange(self.k))
   roots=torch.tensor(np.asarray(chosen),device=x.device)
  policies,_=self.search(x,roots);policies=policies.cpu().numpy().astype(np.float64)
  actions=[int(self.rng[i].choice(len(p),p=p/p.sum())) if alive[i] else 0 for i,p in enumerate(policies)]
  self.history.acted(actions)
  return actions

def controller(entry,model,labels,site):
 if entry['family']=='independent_muzero':return IndependentController(model,labels,entry.get('root_candidates',0))
 return legacy_types(site)[entry['family']][1](model,labels)
