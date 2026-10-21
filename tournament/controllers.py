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
 lana=runpy.run_path(str(scripts/'evaluate-lana-100.py'))
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
