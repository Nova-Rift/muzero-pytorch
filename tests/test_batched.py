import unittest
import torch
from model import Config,MuZero,search
from batched_search import search_batch

class Tests(unittest.TestCase):
    def test_matches_scalar_search_and_keeps_streams_independent(self):
        torch.set_num_threads(1);torch.manual_seed(18)
        m=MuZero(Config(observations=3,actions=4,hidden=16,simulations=20))
        obs=torch.randn(5,3)
        legal=torch.tensor([[True,True,False,True]]*5)
        p,v=search_batch(m,obs,legal=legal)
        for i in range(5):
            expected,value=search(m,obs[i],legal=legal[i])
            torch.testing.assert_close(p[i],expected)
            self.assertAlmostEqual(float(v[i]),value,places=3)
        static,sv=search_batch(m,obs,legal=legal,static=True)
        torch.testing.assert_close(static,p);torch.testing.assert_close(sv,v)
        single,_=search_batch(m,obs[2:3],legal=legal[2:3])
        torch.testing.assert_close(single[0],p[2])

if __name__=='__main__':unittest.main()
