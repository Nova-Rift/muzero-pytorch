import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import first_block_gate as gate
import sys


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
            (directory/'manifest.json').write_text(json.dumps({'arguments':{'updates_per_game':ratio,'timeout_bootstrap':'zero','discount':1.0},'gpu':'H100'}))
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

    def candidate(self, block=None, stats=None, allowed=()):
        trainer=self.trainer(block,stats)
        checked=gate.install_gate(trainer,self.root,self.out,self.base,self.ref,'candidate',allowed)
        return trainer,checked

    def set_candidate_argument(self, name, value):
        manifest=json.loads((self.out/'manifest.json').read_text())
        manifest['arguments'][name]=value
        (self.out/'manifest.json').write_text(json.dumps(manifest))

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

    def test_default_allowlist_empty_and_recorded(self):
        self.reference(); trainer,checked=self.candidate(); trainer.collect()
        for call in gate.validate_continuation.call_args_list[-2:]:
            self.assertEqual(call.kwargs['allowed_arguments'],())
        receipt=json.loads((self.out/'first-block-gate.json').read_text())
        self.assertEqual(receipt['argument_allowlist'],[])
        self.assertEqual(receipt['tolerated_argument_differences'],{})

    def test_receipt_records_allowlist_and_actual_difference(self):
        self.reference(); self.set_candidate_argument('timeout_bootstrap','continue')
        trainer,checked=self.candidate(allowed=['timeout_bootstrap']); trainer.collect()
        calls=gate.validate_continuation.call_args_list[-2:]
        self.assertEqual(len(calls),2)
        for call in calls:
            self.assertEqual(call.kwargs['allowed_arguments'],('timeout_bootstrap',))
            self.assertTrue(call.kwargs['upg2'])
        receipt=json.loads((self.out/'first-block-gate.json').read_text())
        self.assertEqual(receipt['argument_allowlist'],['timeout_bootstrap'])
        self.assertEqual(receipt['tolerated_argument_differences'],
                         {'timeout_bootstrap':{'candidate':'continue','baseline':'zero','reference':'zero'}})
        self.assertEqual(checked[0]['argument_allowlist'],['timeout_bootstrap'])
        self.assertTrue(receipt['full_128_trajectory_parity'])

    def test_allowlist_does_not_relax_array_parity(self):
        self.reference(); self.set_candidate_argument('timeout_bootstrap','continue')
        block=copy.deepcopy(self.block); block['observation'][0,1400,0]=1
        trainer,_=self.candidate(block,allowed=('timeout_bootstrap',))
        with self.assertRaisesRegex(ValueError,'Array differs'):trainer.collect()
        self.assertFalse((self.out/'first-block-gate.json').exists())

    def test_reference_mode_rejects_allowlist(self):
        trainer=self.trainer()
        with self.assertRaisesRegex(ValueError,'Reference mode accepts no argument allowlist'):
            gate.install_gate(trainer,self.root,self.ref,self.base,self.ref,'reference',('timeout_bootstrap',))
        self.assertEqual(gate.resolve_allowlist((),'reference'),())

    def test_resolve_allowlist_rejects_unknown_names(self):
        self.assertEqual(gate.resolve_allowlist(['discount','timeout_bootstrap','discount'],'candidate'),('discount','timeout_bootstrap'))
        for names in (['seed'],['simulations'],['timeout_bootstrap','updates_per_game'],['timeout-bootstrap']):
            with self.subTest(names=names), self.assertRaisesRegex(ValueError,'Unsupported --allow-arg'):
                gate.resolve_allowlist(names,'candidate')

    def test_cli_rejects_unknown_allow_arg_before_archive_binding(self):
        argv=['first_block_gate.py','--source',str(self.root),'--baseline',str(self.base),'--output',str(self.out),
              '--mode','candidate','--reference',str(self.ref),'--allow-arg','seed','--','--games','2048']
        with patch.object(sys,'argv',argv), patch.object(gate,'bind_archive') as bind:
            with self.assertRaisesRegex(ValueError,'Unsupported --allow-arg: seed'):gate.main()
        bind.assert_not_called()
        argv[argv.index('seed')]='timeout_bootstrap'; argv[argv.index('candidate')]='reference'; argv[argv.index('--output')+1]=str(self.ref)
        with patch.object(sys,'argv',argv), patch.object(gate,'bind_archive') as bind:
            with self.assertRaisesRegex(ValueError,'Reference mode accepts no argument allowlist'):gate.main()
        bind.assert_not_called()

    def test_tolerated_differences_lists_only_changed_allowed_values(self):
        recorded=gate.tolerated_differences(('discount','timeout_bootstrap'),{'timeout_bootstrap':'continue','discount':1.0},
                                            baseline={'timeout_bootstrap':'zero','discount':1.0},reference={'timeout_bootstrap':'zero','discount':1.0})
        self.assertEqual(recorded,{'timeout_bootstrap':{'candidate':'continue','baseline':'zero','reference':'zero'}})


if __name__=='__main__':unittest.main()
