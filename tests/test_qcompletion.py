import unittest
from pathlib import Path
import torch
from model import Config, MuZero
from batched_search_qcompletion import search_batch


class QCompletionTests(unittest.TestCase):
    def test_only_search_behavior_change(self):
        root=Path(__file__).resolve().parents[1]
        old=(root/'batched_search_exploration.py').read_text()
        new=(root/'batched_search_qcompletion.py').read_text()
        extra='\n            q = torch.where(nv>0,q,torch.zeros_like(q))'
        # Original raw-Q mask is retained; only the second mask is new.
        index=new.index('            q = torch.where((high>low)')
        prefix,tail=new[:index],new[index:]
        self.assertEqual(tail.count(extra),1)
        self.assertEqual(prefix+tail.replace(extra,'',1),old)

    def test_static_batch_independence_and_forced_roots(self):
        torch.set_num_threads(1);torch.manual_seed(23001)
        m=MuZero(Config(observations=3,actions=5,hidden=16,simulations=16))
        x=torch.randn(3,3);candidates=torch.tensor([[0,2],[2,4],[0,4]])
        legal=torch.tensor([[True,False,True,False,True]]*3)
        p,v=search_batch(m,x,legal=legal,root_candidates=candidates)
        sp,sv=search_batch(m,x,legal=legal,root_candidates=candidates,static=True)
        torch.testing.assert_close(p,sp);torch.testing.assert_close(v,sv)
        self.assertTrue((p.gather(1,candidates)>=1/16).all())
        self.assertTrue((p[:,[1,3]]==0).all())
        torch.testing.assert_close(p.sum(-1),torch.ones(3))
        for i in range(3):
            ip,iv=search_batch(m,x[i:i+1],legal=legal[i:i+1],root_candidates=candidates[i:i+1])
            torch.testing.assert_close(ip[0],p[i]);torch.testing.assert_close(iv[0],v[i])


if __name__=='__main__':unittest.main()
