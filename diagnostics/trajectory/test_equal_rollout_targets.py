import unittest
import torch
from equal_rollout_targets import aggregate,metrics
class Tests(unittest.TestCase):
 def test_group_order(self):
  p=torch.tensor([[1.,0.],[0.,1.],[0.,1.],[0.,1.]])
  torch.testing.assert_close(aggregate(p,2),torch.tensor([[.5,.5],[0.,1.]]))
 def test_bad_inputs(self):
  for p,g in [(torch.tensor([[1.,1.]]),1),(torch.tensor([[1.,0.]]),2),(torch.tensor([[float('nan'),0.]]),1)]:
   with self.assertRaises(ValueError):aggregate(p,g)
 def test_mse_reference(self):
  self.assertAlmostEqual(metrics(torch.tensor([[1.,0.],[0.,1.]]),torch.tensor([.5,.5]))['mean_squared_distance'],.5)
if __name__=='__main__':unittest.main()
