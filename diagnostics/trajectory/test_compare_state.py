import unittest,copy,torch,numpy as np
from compare_state import compare
class Tests(unittest.TestCase):
 def state(self):return dict(model={"w":torch.tensor([1.,2.])},optimizer={"step":torch.tensor(2048.)},replay=[np.array([1,2],dtype=np.int32)],rng={"counter":5})
 def test_exact(self):a=self.state();self.assertIsNone(compare(a,copy.deepcopy(a)))
 def test_model(self):a=self.state();b=copy.deepcopy(a);b["model"]["w"][0]+=1;self.assertEqual(compare(a,b)["path"],"state.model.w")
 def test_replay_dtype(self):a=self.state();b=copy.deepcopy(a);b["replay"][0]=b["replay"][0].astype(np.int64);self.assertIsNotNone(compare(a,b))
 def test_rng(self):a=self.state();b=copy.deepcopy(a);b["rng"]["counter"]+=1;self.assertEqual(compare(a,b)["path"],"state.rng.counter")
 def test_optimizer(self):a=self.state();b=copy.deepcopy(a);b["optimizer"]["step"]+=1;self.assertIsNotNone(compare(a,b))
 def test_missing(self):a=self.state();b=copy.deepcopy(a);del b["rng"];self.assertEqual(compare(a,b)["reason"],"keys")
if __name__=="__main__":unittest.main()
