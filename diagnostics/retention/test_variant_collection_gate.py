import json,shutil,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import variant_collection_gate as g
class GateTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.old=self.root/'old';self.out=self.root/'out'
  self.old.mkdir();self.out.mkdir();np.savez_compressed(self.old/'initial.npz',w=np.array([1.,2.]));shutil.copyfile(self.old/'initial.npz',self.out/'initial.npz')
  for d in (self.old,self.out):(d/'manifest.json').write_text(json.dumps({'gpu':'test-H100'}))
  self.block={'x':np.ones((2,22),dtype=np.float32),'indices':np.zeros(2,dtype=np.int64)};self.stats={'games':128,'search':.1}
  np.savez_compressed(self.old/'trace-0.npz',x=self.block['x']);(self.old/'metrics.jsonl').write_text(json.dumps({'batch_seconds':self.stats})+'\n')
  self.trainer=SimpleNamespace(collect=lambda:(self.block,self.stats))
 def tearDown(self):self.tmp.cleanup()
 def run_gate(self,variant):
  checked=g.install_gate(self.trainer,self.root,self.out,self.old,{'pinned':True},variant)
  with patch.object(g,'validate_declared') as v:self.trainer.collect();v.assert_called_once()
  return checked
 def test_mir_exact_archived_evidence(self):
  x=self.run_gate('mir')[0];self.assertTrue(x['complete']);self.assertEqual(x['trajectory_games'],2);self.assertFalse(x['full_128_trajectory_parity'])
 def test_mir_changed_trace_rejected(self):
  self.block['x'][0,0]=2
  with self.assertRaisesRegex(ValueError,'Array differs'):self.run_gate('mir')
 def test_mir_changed_aggregate_rejected(self):
  self.stats['games']=127
  with self.assertRaisesRegex(ValueError,'aggregates differ'):self.run_gate('mir')
 def test_ego_different_collection_accepted(self):
  self.block['x'][0,0]=2;self.assertFalse(self.run_gate('ego')[0]['collection_parity_required'])
 def test_changed_initial_rejected_before_collect(self):
  np.savez_compressed(self.out/'initial.npz',w=np.array([9.,2.]));self.trainer.collect=lambda:(_ for _ in ()).throw(AssertionError('must not collect'))
  with self.assertRaisesRegex(ValueError,'Array differs'):self.run_gate('ego')
 def test_source_failure_before_collect(self):
  g.install_gate(self.trainer,self.root,self.out,self.old,{},'ego')
  with patch.object(g,'validate_declared',side_effect=ValueError('source bad')):
   with self.assertRaisesRegex(ValueError,'source bad'):self.trainer.collect()
 def test_subsequent_collection_not_recompared(self):
  self.run_gate('mir');self.block['x'][0,0]=9;self.trainer.collect()
if __name__=='__main__':unittest.main()
