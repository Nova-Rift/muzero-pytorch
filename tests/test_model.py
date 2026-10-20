import unittest
import torch
from model import Config, MuZero, search, nstep_returns, support_targets, transform, inverse_transform

class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(4)
        self.model=MuZero(Config(observations=2,actions=3,hidden=16,simulations=12))

    def test_scalar_transform_roundtrip(self):
        x=torch.tensor([-20.,-1.,0.,1.,20.])
        torch.testing.assert_close(inverse_transform(transform(x)),x,atol=.001,rtol=.001)
        y=support_targets(x,20)
        torch.testing.assert_close(y.sum(-1),torch.ones_like(x))

    def test_terminal_vs_truncation(self):
        self.assertEqual(nstep_returns([1,2],[0,0,100],[False,True],.5),[2,2])
        self.assertEqual(nstep_returns([1,2],[0,0,100],[False,False],.5),[27,52])

    def test_search_mask_and_visits(self):
        p,v=search(self.model,torch.zeros(2),legal=torch.tensor([True,False,True]))
        self.assertEqual(p[1],0);self.assertAlmostEqual(float(p.sum()),1.)
        torch.testing.assert_close(p*12,(p*12).round())
        self.assertTrue(torch.isfinite(torch.tensor(v)))
        with self.assertRaises(ValueError):search(self.model,torch.zeros(2),legal=torch.zeros(3,dtype=torch.bool))

    def test_unrolled_loss_trains_dynamics(self):
        m=self.model; opt=torch.optim.Adam(m.parameters(),lr=.01)
        obs=torch.randn(4,2);actions=torch.randint(3,(4,2));rewards=torch.ones(4,2)
        values=torch.ones(4,3);policies=torch.ones(4,3,3)/3
        before=m.dynamics[0].weight.detach().clone()
        losses=[]
        for _ in range(20):
            loss=m.loss(obs,actions,rewards,values,policies).mean()
            opt.zero_grad();loss.backward();opt.step();losses.append(float(loss))
        self.assertLess(losses[-1],losses[0]);self.assertFalse(torch.equal(before,m.dynamics[0].weight))

if __name__=='__main__': unittest.main()
