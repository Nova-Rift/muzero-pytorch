"""Experimental no-root-noise controller with per-game/seat RNG streams."""
import copy
from dataclasses import replace
import numpy as np
import torch
from split_search import split_search

class SplitController:
    def __init__(self,model,labels,history_factory,search_fn):
        self.model=copy.copy(model)
        self.model.config=replace(model.config,exploration_fraction=0.)
        self.count=len(labels);self.search_fn=search_fn
        self.history=history_factory(self.count,64,next(model.parameters()).device)
        self.rng=[np.random.default_rng(23000000+s*11+seat) for s,seat in labels]
    @torch.no_grad()
    def decide(self,observations,alive):
        if len(observations)!=self.count or len(alive)!=self.count:raise ValueError('stream count changed')
        x=self.history.observe(np.asarray(observations,dtype=np.float32))
        _,logits,_=self.model.initial(x);prior=logits.softmax(-1).cpu().numpy().astype(np.float64)
        roots=[]
        for i,p in enumerate(prior):
            p/=p.sum()
            roots.append([self.rng[i].choice(len(p),2,replace=False,p=p) if alive[i] else np.arange(2) for _ in range(4)])
        candidates=torch.tensor(np.asarray(roots),device=x.device,dtype=torch.long)
        noise=x.new_full((self.count,4,self.model.config.actions),1/self.model.config.actions)
        policies,_=split_search(self.model,x,self.search_fn,root_candidates=candidates,root_noise=noise)
        policies=policies.cpu().numpy().astype(np.float64)
        actions=[int(self.rng[i].choice(len(p),p=p/p.sum())) if alive[i] else 0 for i,p in enumerate(policies)]
        self.history.acted(actions)
        return actions
