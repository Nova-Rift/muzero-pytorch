import unittest
from tournament.run import wilson
class TournamentStatisticsTests(unittest.TestCase):
 def test_zero_wins_still_has_uncertainty(self):
  lower,upper=wilson(0,128);self.assertAlmostEqual(lower,0);self.assertGreater(upper,0)
 def test_complement_intervals(self):
  a=wilson(18,32);b=wilson(14,32)
  self.assertAlmostEqual(a[0],1-b[1]);self.assertAlmostEqual(a[1],1-b[0])

import numpy as np
import tournament.run as run_module
class FakePool:
 """Captures the wire actions sent for each decision; terminates every game after one step."""
 sent=[]
 def __init__(self,site,workers=1,agents=2,cadence='homepage'):
  self.sent=FakePool.sent
 def reset(self,seeds):
  n=len(seeds);return dict(observation=np.zeros((n,11,64),np.float32),alive=np.ones((n,11),bool),reward=np.zeros((n,11)),finished=np.zeros(n,bool),winner=np.full(n,-1))
 def step(self,actions):
  self.sent.append(np.array(actions));n=len(actions)
  return dict(observation=np.zeros((n,11,64),np.float32),alive=np.ones((n,11),bool),reward=np.zeros((n,11)),finished=np.ones(n,bool),winner=np.zeros(n,int))
 def close(self):pass
class FixedController:
 def __init__(self,action):self.action=action
 def decide(self,observations,alive):return [self.action]*len(observations)
class PerModelDecodingTests(unittest.TestCase):
 def test_only_the_relative_model_is_offset_and_absolute_is_unchanged(self):
  FakePool.sent.clear();original=(run_module.Pool,run_module.controller)
  run_module.Pool=FakePool;run_module.controller=lambda entry,model,labels,site,micro:FixedController(13)
  try:
   entries=[dict(id='EGO',family='independent_muzero',aim_decoding='relative'),dict(id='E1024',family='independent_muzero')]
   result=run_module.play('site',entries,[None,None],[100,101],0,1,4)
  finally:run_module.Pool,run_module.controller=original
  self.assertEqual(len(FakePool.sent),1);sent=FakePool.sent[0]
  # game 0: EGO on seat 0, E1024 on seat 1; game 1: swapped. Only EGO's actions carry the +153 wire offset.
  self.assertEqual(sent[0,0],166);self.assertEqual(sent[0,1],13);self.assertEqual(sent[1,1],166);self.assertEqual(sent[1,0],13)
  self.assertEqual([r['seat_a'] for r in result['records']],[0,1])
 def test_absent_key_means_absolute_for_both_models(self):
  FakePool.sent.clear();original=(run_module.Pool,run_module.controller)
  run_module.Pool=FakePool;run_module.controller=lambda entry,model,labels,site,micro:FixedController(13)
  try:run_module.play('site',[dict(id='A',family='independent_muzero'),dict(id='B',family='independent_muzero')],[None,None],[100,101],0,1,4)
  finally:run_module.Pool,run_module.controller=original
  self.assertTrue((FakePool.sent[0][:,:2]==13).all())
 def test_unknown_decoding_rejected(self):
  with self.assertRaises(ValueError):run_module.wire_offset('mirror')
