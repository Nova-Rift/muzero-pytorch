import json,tempfile,unittest
from pathlib import Path
import numpy as np
from trajectory_recorder import RecordingPool
class Fake:
 def __init__(self):self.n=0;self.closed=False
 def state(self):return {k:np.full((3,11,64) if k=='observation' else (3,11) if k in ['alive','reward'] else (3,),self.n) for k in ['observation','alive','reward','finished','winner']}
 def reset(self,seeds):self.seeds=seeds;self.last=self.state();return self.last
 def step(self,a):self.actions=a;self.n+=1;self.last=self.state();return self.last
 def close(self):self.closed=True
class Tests(unittest.TestCase):
 def test_forwarding_bounds_and_no_alias(self):
  with tempfile.TemporaryDirectory() as d:
   inner=Fake();p=Path(d)/'trace';r=RecordingPool(inner,p,2,2)
   self.assertIs(r.reset([5,6,7]),inner.last);a=np.zeros((3,11),dtype=int)
   for i in range(3):self.assertIs(r.step(a),inner.last);self.assertIs(inner.actions,a)
   a[:]=9;r.close();rows=[json.loads(x) for x in p.read_text().splitlines()]
   self.assertEqual(len(rows),3);self.assertEqual(rows[0]['seeds'],[5,6]);self.assertEqual(rows[1]['actions'],[[0]*11]*2);self.assertTrue(inner.closed)
 def test_no_overwrite_and_bad_bounds(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'trace';p.write_text('history')
   with self.assertRaises(FileExistsError):RecordingPool(Fake(),p)
   with self.assertRaises(ValueError):RecordingPool(Fake(),Path(d)/'other',0)
   self.assertEqual(p.read_text(),'history')
 def test_reset_required(self):
  with tempfile.TemporaryDirectory() as d:
   r=RecordingPool(Fake(),Path(d)/'trace')
   with self.assertRaises(RuntimeError):r.step(np.zeros((3,11)))
   r.reset([1,2,3])
   with self.assertRaises(RuntimeError):r.reset([1,2,3])
   r.close()
if __name__=='__main__':unittest.main()
