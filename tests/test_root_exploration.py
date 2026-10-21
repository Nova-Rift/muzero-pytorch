import unittest
import torch
from model import MuZero,Config
from batched_search import search_batch as original
from batched_search_exploration import search_batch
class Tests(unittest.TestCase):
 def test_unchanged_without_forcing(self):
  torch.manual_seed(3);m=MuZero(Config(observations=5,actions=9,hidden=16,simulations=8))
  x=torch.randn(3,5)
  a,b=original(m,x);c,d=search_batch(m,x)
  torch.testing.assert_close(a,c);torch.testing.assert_close(b,d)
 def test_reserved_candidates_are_visited_even_with_collapsed_prior(self):
  m=MuZero(Config(observations=5,actions=9,hidden=16,simulations=8))
  with torch.no_grad():m.policy[-1].weight.zero_();m.policy[-1].bias.fill_(-100);m.policy[-1].bias[0]=100
  x=torch.zeros(2,5);candidates=torch.tensor([[0,1,2,3],[4,5,6,7]])
  p,v=search_batch(m,x,root_candidates=candidates)
  self.assertTrue((p.gather(1,candidates)>=1/8).all())
  torch.testing.assert_close(p.sum(-1),torch.ones(2))
