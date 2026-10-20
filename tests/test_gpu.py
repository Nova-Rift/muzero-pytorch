import unittest
import numpy as np
import torch
from model import Config,MuZero
from batched_search import search_batch
from arena.train import Search

@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class Tests(unittest.TestCase):
    def test_graph_matches_eager_and_sees_weight_updates(self):
        torch.manual_seed(5)
        model=MuZero(Config(observations=5,actions=4,hidden=16,simulations=8)).cuda()
        searcher=Search(model,3,True)
        for _ in range(3):
            x=torch.randn(3,5,device='cuda')
            _,p,v=searcher(x)
            ep,ev=search_batch(model,x,root_noise=searcher.noise,static=True)
            np.testing.assert_allclose(p,ep.cpu().numpy(),atol=1e-6)
            np.testing.assert_allclose(v,ev.cpu().numpy(),atol=1e-3)
            with torch.no_grad():model.value[-1].bias.add_(torch.randn_like(model.value[-1].bias)*.01)

if __name__=='__main__':unittest.main()
