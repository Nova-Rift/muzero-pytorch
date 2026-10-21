import tempfile,unittest
from pathlib import Path
import numpy as np
from scripts.compare_selfplay_prefix import compare_traces

class PrefixTests(unittest.TestCase):
 def test_detects_trace_change_even_with_equal_histogram(self):
  with tempfile.TemporaryDirectory() as d:
   a=Path(d)/'a';b=Path(d)/'b';a.mkdir();b.mkdir()
   np.savez(a/'trace-0.npz',actions=np.array([[0,1],[1,0]]),valid=np.ones((2,2),bool))
   np.savez(b/'trace-0.npz',actions=np.array([[0,1],[1,0]]),valid=np.ones((2,2),bool))
   self.assertTrue(compare_traces(a,b,128,128)['all_equal'])
   np.savez(b/'trace-0.npz',actions=np.array([[1,0],[0,1]]),valid=np.ones((2,2),bool))
   r=compare_traces(a,b,128,128)
   self.assertFalse(r['all_equal']);self.assertEqual(r['blocks'][0]['fields']['actions']['max_abs_difference'],1.)
if __name__=='__main__':unittest.main()
