import unittest
from types import SimpleNamespace
import numpy as np
import torch
from arena.train_selfplay import collect

class Model:
 def __init__(self):self.config=SimpleNamespace(discount=1.0);self.parameter=torch.nn.Parameter(torch.zeros(1));self.bootstrap_calls=0
 def parameters(self):return iter([self.parameter])
 def initial(self,x):
  self.bootstrap_calls+=1
  logits=torch.zeros(len(x),21);logits[:,-1]=5
  return None,None,logits
class Pool:
 def reset(self,seeds):self.t=0;return self.state()
 def state(self):
  alive=np.zeros((2,11),bool);alive[:,:2]=True
  finished=np.array([self.t>=2,False]);reward=np.zeros((2,11),np.float32)
  if self.t>=2:alive[0,1]=False
  if self.t==2:reward[0,0]=1
  return dict(observation=np.zeros((2,11,64),np.float32),alive=alive,finished=finished,reward=reward)
 def step(self,commands):self.t+=1;return self.state()
class Search:
 indices=list(range(22))
 def __call__(self,x):return np.zeros(22,np.int64),np.ones((22,153),np.float32)/153,np.zeros(22,np.float32)
class Tests(unittest.TestCase):
 def run_mode(self,mode=None):
  model=Model();kw={} if mode is None else {'timeout_bootstrap':mode}
  block,stats=collect(Pool(),model,[0,1],3,1,Search(),return_mode='episode',**kw)
  return block,stats,model
 def test_default_reproduces_continue_and_zero_changes_only_timeout_targets(self):
  default,_,_=self.run_mode();old,stats,oldmodel=self.run_mode('continue');new,newstats,newmodel=self.run_mode('zero')
  for key in old:np.testing.assert_array_equal(default[key],old[key])
  for key in ('observation','actions','rewards','policies','valid','absorbing','indices'):np.testing.assert_array_equal(old[key],new[key])
  np.testing.assert_array_equal(new['targets'][:2,0],1)
  np.testing.assert_array_equal(new['targets'][:2,1],0)
  np.testing.assert_array_equal(new['targets'][:3,11:13],0)
  self.assertTrue((old['targets'][:3,11:13]>0).all())
  self.assertFalse(new['absorbing'][:,11:13].any())
  self.assertEqual(stats['natural_terminal'],newstats['natural_terminal']);self.assertEqual(newstats['cutoffs'],1)
  self.assertEqual(oldmodel.bootstrap_calls,1);self.assertEqual(newmodel.bootstrap_calls,0)
 def test_invalid_mode_rejected(self):
  with self.assertRaises(ValueError):self.run_mode('typo')
if __name__=='__main__':unittest.main()
