import unittest
import torch
from probe import coefficients, terms, gradients, measure
from model import Config, MuZero
class ProbeTests(unittest.TestCase):
 def setUp(self):
  torch.set_num_threads(1);torch.manual_seed(7)
  self.model=MuZero(Config(observations=4,actions=3,hidden=8,support=2)).double()
  self.batch=(torch.randn(3,4,dtype=torch.float64),torch.randint(3,(3,5)),torch.randn(3,5,dtype=torch.float64),torch.randn(3,6,dtype=torch.float64),torch.softmax(torch.randn(3,6,3,dtype=torch.float64),-1),torch.ones(3,6,dtype=torch.float64),torch.ones(3,6,dtype=torch.float64),torch.ones(3,5,dtype=torch.float64))
 def test_coefficients(self):
  m=torch.tensor([[1.,1,1,1,1,1],[1,0,0,0,0,0],[0,1,0,1,0,0],[0,0,0,0,0,0]])
  c=coefficients(m,5)
  torch.testing.assert_close(c[0],torch.tensor([.5,.1,.1,.1,.1,.1]));torch.testing.assert_close(c.sum(1),m.sum(1)/6)
  torch.testing.assert_close(c[1],m[1]/6);torch.testing.assert_close(c[2],m[2]/6)
  self.assertTrue(torch.isfinite(c).all())
  with self.assertRaises(ValueError):coefficients(m+.1,5)
 def test_baseline_and_frozen_parameters(self):
  self.batch[5][1,2:]=0;self.batch[6][2,1:]=0;self.batch[7][1,3:]=0
  before={k:v.clone() for k,v in self.model.state_dict().items()};r=measure(self.model,self.batch)
  self.assertEqual(r['reconstructed']['max_gradient_error'],0)
  for k,v in before.items():self.assertTrue(torch.equal(v,self.model.state_dict()[k]))
 def test_reward_only_identical(self):
  self.batch[5].zero_();self.batch[6].zero_()
  losses=[sum(t[0] for t in terms(self.model,self.batch,r)).mean() for r in (1,5)]
  self.assertTrue(torch.equal(*losses));g=[gradients(l,self.model) for l in losses]
  for n in g[0]:self.assertTrue(torch.equal(g[0][n],g[1][n]))
 def test_empty_masks(self):
  for t in self.batch[5:]:t.zero_()
  r=measure(self.model,self.batch)
  for key in ('original','reconstructed','root5'):
   self.assertEqual(r[key]['loss'],0);self.assertEqual(r[key]['gradients']['total'],0)
 def test_real_replay_function_reproducibility(self):
  import numpy as np
  from probe import RecordingRNG, replay_batch, tensor_sha
  block={'observation':np.arange(5*2*64,dtype=np.float32).reshape(5,2,64),'actions':np.zeros((4,2),dtype=np.int64),'targets':np.zeros((5,2),dtype=np.float32),'policies':np.ones((5,2,153),dtype=np.float32)/153,'valid':np.ones((5,2),dtype=bool),'absorbing':np.zeros((5,2),dtype=bool),'rewards':np.zeros((4,2),dtype=np.float32),'indices':np.array([[0,0],[2,1],[3,0]])}
  block['absorbing'][4]=True
  before={k:v.copy() for k,v in block.items()}
  runs=[]
  for _ in range(2):
   rng=RecordingRNG(np.random.default_rng(123));b=replay_batch(block,rng,8,2,5,'cpu');runs.append(([tensor_sha(t) for t in b],rng.calls[0].tolist()))
  self.assertEqual(runs[0],runs[1])
  for k,v in before.items():self.assertTrue(np.array_equal(v,block[k]))
 def test_nonfinite_and_unknown_group_rejected(self):
  with self.assertRaises(ValueError):gradients(next(self.model.parameters()).sum()*float('nan'),self.model)
  self.model.extra=torch.nn.Parameter(torch.ones(1))
  with self.assertRaises(ValueError):gradients(self.model.extra.sum(),self.model)
 def test_hidden_scaling(self):
  class Chain:
   config=Config(observations=1,actions=2,hidden=1,support=1)
   def initial(self,x):return x,torch.cat((x,torch.zeros_like(x)),-1),x.expand(-1,3)*0
   def recurrent(self,x,a):
    h,p,v=self.initial(x);return h,v,p,v
  for depth in range(6):
   x=torch.zeros(1,1,dtype=torch.float64,requires_grad=True);m=torch.zeros(1,6,dtype=torch.float64);m[:,depth]=1
   b=(x,torch.zeros(1,5,dtype=torch.long),torch.zeros(1,5),torch.zeros(1,6),torch.tensor([[[1.,0.]]*6]),m,torch.zeros_like(m),torch.zeros(1,5))
   for ratio in (1,5):
    loss=sum(t[0] for t in terms(Chain(),b,ratio)).sum();g=torch.autograd.grad(loss,x,retain_graph=True)[0]
    self.assertAlmostEqual(float(g),-.5*.5**depth/6,places=12)
if __name__=='__main__':unittest.main()
