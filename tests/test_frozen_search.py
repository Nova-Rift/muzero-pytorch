import unittest
import torch
from model import Config,MuZero
from arena.frozen_search import FrozenSearch
from batched_search_exploration import search_batch
class Tests(unittest.TestCase):
 def check(self,device):
  torch.manual_seed(95)
  m=MuZero(Config(observations=5,actions=9,hidden=16,simulations=8)).to(device)
  for count in [0,4]:
   f=FrozenSearch(m,3,count)
   for _ in range(3):
    x=torch.randn(3,5,device=device);roots=torch.tensor([[0,1,2,3]]*3,device=device) if count else None
    a,b=f(x,roots);c,d=search_batch(m,x,root_candidates=roots)
    torch.testing.assert_close(a,c);torch.testing.assert_close(b,d)
 def test_cpu(self):self.check('cpu')
 @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
 def test_cuda(self):self.check('cuda')
