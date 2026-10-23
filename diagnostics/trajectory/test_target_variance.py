import unittest,torch,math
from audit_target_variance import decompose,heldout
class Tests(unittest.TestCase):
 def test_stochastic_targets_fit_mean(self):
  r=decompose(torch.eye(2),torch.tensor([.5,.5]));self.assertAlmostEqual(r['sampling_js'],math.log(2),places=6);self.assertAlmostEqual(r['averaged_target_kl_to_prior'],0,places=6)
 def test_fixed_target_mismatch(self):
  r=decompose(torch.tensor([[1.,0.],[1.,0.]]),torch.tensor([.5,.5]));self.assertAlmostEqual(r['sampling_js'],0,places=6);self.assertAlmostEqual(r['averaged_target_kl_to_prior'],math.log(2),places=6)
class HeldoutTests(unittest.TestCase):
 def test_matching_draws_help(self):
  r=heldout(torch.tensor([[1.,0.]]*4),torch.tensor([.5,.5]));self.assertLess(r['mixture_minus_prior_ce'],0)
 def test_opposite_validation_detects_overfit(self):
  r=heldout(torch.tensor([[1.,0.],[1.,0.],[0.,1.],[0.,1.]]),torch.tensor([.5,.5]));self.assertGreater(r['mixture_minus_prior_ce'],0)
if __name__=='__main__':unittest.main()
