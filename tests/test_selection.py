import unittest,tempfile,json,hashlib
from pathlib import Path
import numpy as np
from experiments.selection import validate,development_pass,confirmation,freeze

def result(wins,cutoffs=0,n=200):
 rows=[dict(seat=i%2,winner=i%2 if i<wins else 1-i%2,finished=i<n-cutoffs) for i in range(n)]
 return dict(games=n,seed=42000000,decision='search',reference_decision='search',records=rows,wins=wins,losses=n-wins-cutoffs,draws=0,cutoffs=cutoffs)
class SelectionTests(unittest.TestCase):
 def test_development_boundaries(self):
  self.assertFalse(development_pass(19,0,32));self.assertTrue(development_pass(20,6,32));self.assertFalse(development_pass(20,7,32))
 def test_confirmation_requires_interval_and_cutoff_bound(self):
  self.assertFalse(confirmation(result(100),42000000)['passed'])
  self.assertTrue(confirmation(result(130,40),42000000)['passed'])
  self.assertFalse(confirmation(result(130,41),42000000)['passed'])
 def test_mismatched_counts_and_unbalanced_seats_rejected(self):
  r=result(100);r['wins']=101
  with self.assertRaises(ValueError):validate(r,200,42000000)
  r=result(100);r['records'][1]['seat']=0
  with self.assertRaises(ValueError):validate(r,200,42000000)
 def test_frozen_selection_and_hash_guard(self):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);(root/'report.json').write_text(json.dumps({'games':8192,'requested_run_completed':True}));np.savez(root/'initial.npz',x=np.zeros(1));paths=[]
   for epoch,cutoffs in [(2048,2),(4096,1),(8192,1)]:
    name=f'checkpoint-{epoch}.npz';np.savez(root/name,x=np.ones(1)*epoch)
    r=result(20,cutoffs,32);r.update(seed=41000000,challenger=name,reference='initial.npz',checkpoint_sha256=hashlib.sha256((root/name).read_bytes()).hexdigest())
    path=root/f'{epoch}.json';path.write_text(json.dumps(r));paths.append(path)
   selected=freeze(root,paths,root/'selection.json');self.assertEqual(selected['checkpoint'],'checkpoint-4096.npz')
   with self.assertRaisesRegex(ValueError,'already frozen'):freeze(root,paths,root/'selection.json')
   np.savez(root/'checkpoint-4096.npz',x=np.zeros(1))
   with self.assertRaisesRegex(ValueError,'changed after'):freeze(root,paths,root/'other.json')
if __name__=='__main__':unittest.main()
