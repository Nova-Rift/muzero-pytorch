import unittest,copy
import torch
from test_split_search import Model,backend
from split_controller import SplitController
class History:
 def __init__(self,n,context,device):self.n=n;self.last=None
 def observe(self,x):return torch.tensor(x)
 def acted(self,a):self.last=list(a)
class Tests(unittest.TestCase):
 def test_batch_isolation_and_reset(self):
  m=Model();labels=[(123,0),(124,1)];both=SplitController(m,labels,History,backend);alone=SplitController(m,labels[:1],History,backend);again=SplitController(m,labels[:1],History,backend)
  for _ in range(8):
   a=both.decide([[1.],[2.]],[True,False])[0];b=alone.decide([[1.]],[True])[0];c=again.decide([[1.]],[True])[0];self.assertEqual(a,b);self.assertEqual(b,c)
  self.assertEqual(m.config.exploration_fraction,.25);self.assertEqual(m.config.simulations,32)
 def test_dead_stream_consumes_no_rng(self):
  c=SplitController(Model(),[(12,1)],History,backend);before=copy.deepcopy(c.rng[0].bit_generator.state);self.assertEqual(c.decide([[1.]],[False]),[0]);self.assertEqual(before,c.rng[0].bit_generator.state)
 def test_stream_count_guard(self):
  c=SplitController(Model(),[(12,1)],History,backend)
  with self.assertRaises(ValueError):c.decide([],[])
if __name__=='__main__':unittest.main()
