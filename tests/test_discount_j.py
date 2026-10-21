"""Discount-1 target and scalar/batched/frozen search consistency."""
import unittest
import numpy as np
import torch
from model import Config,MuZero,search
from arena.returns import episode_returns
from arena.frozen_search import FrozenSearch
from batched_search import search_batch

class Tests(unittest.TestCase):
 def test_winner_return_is_one_at_every_preterminal_step(self):
  r=np.zeros((103,2),np.float32);r[-1,0]=1
  terminal=np.zeros_like(r,dtype=bool);terminal[-1]=True
  out=episode_returns(r,terminal,np.array([99,99],np.float32),1)
  np.testing.assert_array_equal(out[:-1,0],1)
  np.testing.assert_array_equal(out[:-1,1],0)
  old=episode_returns(r,terminal,np.zeros(2,np.float32),.99)
  self.assertAlmostEqual(float(old[0,0]),.99**102,places=6)
 def test_undiscounted_search_matches_scalar_and_frozen(self):
  torch.set_num_threads(1);torch.manual_seed(18)
  m=MuZero(Config(observations=3,actions=4,hidden=16,simulations=20,discount=1))
  x=torch.randn(5,3)
  p,v=search_batch(m,x)
  for i in range(5):
   expected,value=search(m,x[i]);torch.testing.assert_close(p[i],expected)
   self.assertAlmostEqual(float(v[i]),value,places=3)
  fp,fv=FrozenSearch(m,5)(x)
  torch.testing.assert_close(fp,p);torch.testing.assert_close(fv,v)
