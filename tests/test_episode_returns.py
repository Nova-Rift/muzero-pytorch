import unittest
import numpy as np
from arena.returns import episode_returns
class Tests(unittest.TestCase):
 def test_terminal_outcome_propagates_and_truncation_keeps_bootstrap(self):
  rewards=np.array([[0,0],[0,0],[1,0]],np.float32)
  terminated=np.array([[False,False],[False,False],[True,False]])
  result=episode_returns(rewards,terminated,np.array([99,2],np.float32),.5)
  np.testing.assert_array_equal(result[:3],[[.25,.25],[.5,.5],[1,1]])
 def test_absorbing_death_masks_later_value(self):
  r=np.zeros((3,1),np.float32);t=np.array([[True],[True],[True]])
  np.testing.assert_array_equal(episode_returns(r,t,np.array([5],np.float32),.99)[:3],0)
