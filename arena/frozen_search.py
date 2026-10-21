"""CUDA graph for frozen evaluation; no random operations captured inside search."""
import torch
from batched_search_exploration import search_batch

class FrozenSearch:
 def __init__(self,model,count,candidates=0,noisy=False):
  self.model=model;device=next(model.parameters()).device
  self.inputs=torch.zeros(count,model.config.observations,device=device)
  self.roots=torch.zeros(count,candidates,dtype=torch.long,device=device) if candidates else None
  self.noise=torch.full((count,model.config.actions),1/model.config.actions,device=device) if noisy else None
  self.graph=None
  if device.type=='cuda':
   stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
   with torch.cuda.stream(stream):
    for _ in range(2):self._run()
   torch.cuda.current_stream().wait_stream(stream)
   self.graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(self.graph):self.outputs=self._run()
 def _run(self):
  return search_batch(self.model,self.inputs,static=True,root_candidates=self.roots,root_noise=self.noise)
 @torch.no_grad()
 def __call__(self,inputs,roots=None):
  self.inputs.copy_(inputs)
  if self.roots is not None:self.roots.copy_(roots)
  if self.noise is not None:self.noise.copy_(torch.distributions.Dirichlet(torch.full_like(self.noise,self.model.config.dirichlet_alpha)).sample())
  if self.graph:self.graph.replay();return self.outputs
  return self._run()
