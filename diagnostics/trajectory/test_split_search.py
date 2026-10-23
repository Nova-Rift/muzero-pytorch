import unittest
from dataclasses import dataclass
import torch
from split_search import split_search
@dataclass
class Config:
 simulations:int=32
 actions:int=4
 dirichlet_alpha:float=.3
 exploration_fraction:float=.25
class Model(torch.nn.Module):
 def __init__(self):super().__init__();self.config=Config();self.weight=torch.nn.Parameter(torch.arange(4.))
 def initial(self,x):return x,self.weight.expand(len(x),-1),x[:,0]
def backend(model,x,root_noise,root_candidates):
 assert model.config.simulations==8
 return torch.nn.functional.one_hot(root_candidates[:,0],4).float(),x[:,0]
class Tests(unittest.TestCase):
 def test_grouping_and_isolation(self):
  m=Model();noise=torch.full((2,4,4),.25);roots=torch.tensor([[[0,1],[1,2],[2,3],[3,0]],[[1,0]]*4]);x=torch.tensor([[7.],[9.]])
  p,v=split_search(m,x,backend,root_noise=noise,root_candidates=roots)
  torch.testing.assert_close(p,torch.tensor([[.25]*4,[0.,1.,0.,0.]]));torch.testing.assert_close(v,torch.tensor([7.,9.]));self.assertEqual(m.config.simulations,32);self.assertIsNone(m.weight.grad)
 def test_rng_restore(self):
  m=Model();x=torch.ones(2,1);state=torch.get_rng_state();a=split_search(m,x,backend);torch.set_rng_state(state);b=split_search(m,x,backend)
  for u,v in zip(a,b):torch.testing.assert_close(u,v,rtol=0,atol=0)
 def test_invalid_controls(self):
  m=Model();x=torch.ones(1,1)
  with self.assertRaises(ValueError):split_search(m,x,backend,groups=3)
  with self.assertRaises(ValueError):split_search(m,x,backend,root_noise=torch.zeros(1,4,4))
  with self.assertRaises(ValueError):split_search(m,x,backend,root_candidates=torch.zeros(1,4,2,dtype=torch.long))
if __name__=='__main__':unittest.main()
