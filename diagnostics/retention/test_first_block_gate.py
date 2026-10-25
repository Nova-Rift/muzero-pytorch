import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import first_block_gate as gate


class FirstBlockTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base = self.root/'baseline'; self.ref = self.root/'reference'; self.out = self.root/'candidate'
        self.block = {'observation':np.zeros((3,1408,2),np.float32), 'indices':np.array([[0,0]],np.int64)}
        self.stats = {'games':128, 'hits':4, 'search':1.2}
        for directory, ratio in ((self.base,1),(self.ref,1),(self.out,2)):
            directory.mkdir()
            (directory/'manifest.json').write_text(json.dumps({'arguments':{'updates_per_game':ratio},'gpu':'H100'}))
            np.savez_compressed(directory/'initial.npz', w=np.ones(2,np.float32))
        np.savez_compressed(self.base/'trace-0.npz', observation=self.block['observation'][:,:22])
        (self.base/'metrics.jsonl').write_text(json.dumps({'batch_seconds':self.stats})+'\n')
        self.validation = patch.object(gate, 'validate_continuation')
        self.validation.start(); self.addCleanup(self.validation.stop)

    def trainer(self, block=None, stats=None):
        return SimpleNamespace(collect=lambda *a, **k:(copy.deepcopy(self.block if block is None else block), dict(self.stats if stats is None else stats)))

    def reference(self):
        trainer=self.trainer()
        gate.install_gate(trainer,self.root,self.ref,self.base,self.ref,'reference')
        with self.assertRaises(gate.ReferenceComplete):trainer.collect()
        return trainer

    def candidate(self, block=None, stats=None):
        trainer=self.trainer(block,stats)
        checked=gate.install_gate(trainer,self.root,self.out,self.base,self.ref,'candidate')
        return trainer,checked

    def test_reference_stops_before_update(self):
        trainer=self.trainer(); gate.install_gate(trainer,self.root,self.ref,self.base,self.ref,'reference')
        updates=[]
        with self.assertRaises(gate.ReferenceComplete):
            trainer.collect(); updates.append(1)
        self.assertEqual(updates,[])
        self.assertEqual(json.loads((self.ref/'first-block-reference.json').read_text())['optimizer_updates'],0)

    def test_candidate_full_block_and_rng_unchanged(self):
        self.reference(); trainer,checked=self.candidate()
        before=np.random.get_state(); returned=trainer.collect(); after=np.random.get_state()
        self.assertEqual(before[0],after[0]); np.testing.assert_array_equal(before[1],after[1]); self.assertEqual(before[2:],after[2:])
        self.assertTrue(checked[0]['full_128_trajectory_parity'])
        self.assertEqual(checked[0]['trajectory_games'],2)
        np.testing.assert_array_equal(returned[0]['observation'],self.block['observation'])
        trainer.collect(); self.assertEqual(len(checked),1)

    def test_reject_unsaved_game_difference(self):
        self.reference(); block=copy.deepcopy(self.block);block['observation'][0,1400,0]=1
        trainer,_=self.candidate(block)
        with self.assertRaisesRegex(ValueError,'Array differs'):trainer.collect()
        self.assertFalse((self.out/'first-block-gate.json').exists())

    def test_reject_indices_difference(self):
        self.reference(); block=copy.deepcopy(self.block);block['indices'][0,1]=1
        trainer,_=self.candidate(block)
        with self.assertRaisesRegex(ValueError,'indices'):trainer.collect()

    def test_reject_reference_corruption(self):
        self.reference();(self.ref/'full-block.npz').write_bytes(b'corrupt')
        trainer,_=self.candidate()
        with self.assertRaisesRegex(ValueError,'Reference changed'):trainer.collect()

    def test_reject_initial_difference(self):
        self.reference();np.savez_compressed(self.out/'initial.npz',w=np.zeros(2,np.float32))
        trainer,_=self.candidate()
        with self.assertRaisesRegex(ValueError,'Array differs'):trainer.collect()

    def test_reject_aggregate_difference(self):
        self.reference();trainer,_=self.candidate(stats=dict(self.stats,hits=5))
        with self.assertRaisesRegex(ValueError,'aggregates'):trainer.collect()

    def test_reject_dtype_keys_and_nonfinite(self):
        for actual in ({'x':np.ones(2,np.float64)}, {'y':np.ones(2,np.float32)}, {'x':np.array([1,np.nan],np.float32)}):
            with self.assertRaises(ValueError):gate.equal_arrays(actual,{'x':np.ones(2,np.float32)})

    def test_archive_hash_rejected(self):
        with self.assertRaisesRegex(ValueError,'Archived evidence changed'):gate.bind_archive(self.base)

    def test_source_validation_failure_prevents_collection(self):
        with patch.object(gate,'validate_continuation',side_effect=ValueError('source')):
            trainer,_=self.candidate()
            with self.assertRaisesRegex(ValueError,'source'):trainer.collect()


if __name__=='__main__':unittest.main()
