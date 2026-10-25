import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from compressed_checkpoint import compress
from verify_training_states import digest
from verify_compressed_states import verify


class VerifyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'compressed').mkdir()
        self.source = self.root / 'source.json'
        self.source.write_text(json.dumps({'files': {'trainer.py': 'trusted-sha'}}))
        self.arguments = {'device': 'cuda', 'replay_blocks': 3, 'updates_per_game': 1, 'batch': 256}
        self.write('manifest.json', dict(arguments=self.arguments, environment='fixture', torch='2.8.0+cu128'))
        self.write('report.json', dict(requested_run_completed=True, games=4, optimizer_updates=4, training_examples=1024))
        self.state = dict(arguments=self.arguments, environment='fixture', source_files={'trainer.py': 'trusted-sha'}, torch_version='2.8.0+cu128', initial={'w': torch.tensor([1.])}, model={'w': torch.tensor([2.])}, optimizer={'state': {0: {'step': torch.tensor(4.), 'exp_avg': torch.tensor([0.1])}}, 'param_groups': [{'params': [0]}]}, totals={'games': 4, 'optimizer_updates': 4, 'training_examples': 1024}, replay=[{'x': torch.tensor([1.])} for _ in range(3)], numpy_rng=np.random.default_rng(5).bit_generator.state, torch_rng=torch.get_rng_state(), cuda_rng=[torch.tensor([1, 2], dtype=torch.uint8)])
        np.savez(self.root / 'initial.npz', w=np.array([1.], dtype=np.float32))
        np.savez(self.root / 'weights.npz', w=np.array([2.], dtype=np.float32))
        self.encode()

    def write(self, name, value):
        (self.root / name).write_text(json.dumps(value))

    def encode(self):
        name = f'games-4-updates-{4*self.arguments["updates_per_game"]}.pt'
        raw = self.root / name
        torch.save(self.state, raw)
        target = self.root / 'compressed' / (name + '.gz')
        target.unlink(missing_ok=True)
        record = compress(raw, target, digest(raw))
        record.update(raw_name=name, saved_at_unix=raw.stat().st_mtime)
        self.write('compressed/manifest.json', dict(states={name: record}, encoded_bytes=record['compressed_bytes']))
        raw.unlink()

    def verify(self):
        return verify(self.root, self.source, 'arena.train_selfplay', 'externally-bound-baseline', [4])

    def upg2(self):
        self.arguments.update(updates_per_game=2, loss_contract='baseline', games=4, seed=52731415, resume_state=None)
        source = dict(commit='2e67c2b9b7652ccbe75dbed249708356f38c3608', dirty=False,
                      files={'trainer.py': 'trusted-sha'})
        self.source.write_text(json.dumps(source))
        self.write('manifest.json', dict(arguments=self.arguments, environment='fixture', torch='2.8.0+cu128',
                                        source_commit=source['commit'], source_provenance=source))
        self.write('report.json', dict(requested_run_completed=True, games=4, optimizer_updates=8, training_examples=2048))
        self.state['totals'].update(optimizer_updates=8, training_examples=2048)
        self.encode()

    def verify_upg2(self):
        return verify(self.root, self.source, 'arena.train_loss', 'UPG2', [4], 'baseline', 2)

    def test_upg2_complete_chain(self):
        self.upg2()
        result = self.verify_upg2()
        self.assertEqual(result['states'][0]['optimizer_updates'], 8)
        self.assertEqual(result['states'][0]['training_examples'], 2048)

    def test_upg2_requires_explicit_ratio(self):
        self.upg2()
        with self.assertRaisesRegex(ValueError, 'ratio mismatch'): self.verify()

    def test_upg2_wrong_counters(self):
        self.upg2()
        for key, value in [('games', 8), ('optimizer_updates', 4), ('optimizer_updates', 8.0), ('training_examples', 1024)]:
            with self.subTest(key=key, value=value):
                old = self.state['totals'][key]; self.state['totals'][key] = value; self.encode()
                with self.assertRaises(ValueError): self.verify_upg2()
                self.state['totals'][key] = old

    def test_upg2_wrong_report(self):
        self.upg2()
        path = self.root / 'report.json'; original = json.loads(path.read_text())
        for key, value in [('optimizer_updates', 4), ('training_examples', 1024)]:
            self.write('report.json', dict(original, **{key: value}))
            with self.assertRaises(ValueError): self.verify_upg2()

    def test_upg2_fresh_arguments(self):
        self.upg2()
        manifest=json.loads((self.root/'manifest.json').read_text())
        for key,value in [('seed',1),('games',8),('games',4.0),('resume_state','state.pt'),('batch',128)]:
            old=manifest['arguments'][key];manifest['arguments'][key]=value
            self.write('manifest.json',manifest)
            with self.assertRaises(ValueError):self.verify_upg2()
            manifest['arguments'][key]=old

    def test_upg2_wrong_source_commit(self):
        self.upg2()
        source = json.loads(self.source.read_text()); source['commit'] = 'wrong'
        self.source.write_text(json.dumps(source))
        with self.assertRaisesRegex(ValueError, 'source commit'): self.verify_upg2()

    def test_upg2_wrong_objective(self):
        self.upg2()
        for contract in (None, 'root5-mass-v1'):
            with self.assertRaisesRegex(ValueError, 'explicit baseline'):
                verify(self.root, self.source, 'entry', 'arm', [4], contract, 2)

    def test_valid_different_runtime_and_own_cleanup(self):
        protected = self.root / 'existing.pt'; protected.write_bytes(b'keep')
        self.assertTrue(self.verify()['verified'])
        self.assertEqual(protected.read_bytes(), b'keep')
        self.assertFalse(list(self.root.glob('.verify-state-*')))
        self.assertEqual(len(list((self.root / 'compressed').glob('*.gz'))), 1)

    def set_contract(self, contract):
        self.arguments['loss_contract'] = contract
        self.write('manifest.json', dict(arguments=self.arguments, environment='fixture', torch='2.8.0+cu128'))
        self.encode()

    def test_intended_contracts_verified_in_existing_decode(self):
        from verify_compressed_states import restore
        for contract in ('baseline', 'root5-mass-v1', 'first-transition1-v1'):
            with self.subTest(contract=contract):
                self.set_contract(contract)
                with patch('verify_compressed_states.restore', wraps=restore) as decode:
                    result = verify(self.root, self.source, 'entry', 'arm', [4], contract)
                self.assertTrue(result['verified'])
                self.assertEqual(result['expected_loss_contract'], contract)
                self.assertEqual(decode.call_count, 1)

    def test_consistently_wrong_contract_rejected(self):
        self.set_contract('baseline')
        with self.assertRaisesRegex(ValueError, 'Manifest objective differs'):
            verify(self.root, self.source, 'entry', 'arm', [4], 'root5-mass-v1')
        self.assertFalse(list(self.root.glob('.verify-state-*')))

    def test_first_transition_wrong_but_consistent_and_cleanup(self):
        for intended,actual in [('first-transition1-v1','baseline'),('baseline','first-transition1-v1')]:
            with self.subTest(intended=intended,actual=actual):
                self.set_contract(actual)
                with self.assertRaisesRegex(ValueError, 'Manifest objective differs'):
                    verify(self.root, self.source, 'entry', 'arm', [4], intended)
                self.assertFalse(list(self.root.glob('.verify-state-*')))

    def test_corrupt_compressed(self):
        path = next((self.root / 'compressed').glob('*.gz'))
        path.write_bytes(path.read_bytes()[:-2])
        with self.assertRaisesRegex(ValueError, 'Compressed'): self.verify()

    def test_missing_chain(self):
        with self.assertRaisesRegex(ValueError, 'chain'):
            verify(self.root, self.source, 'entry', 'arm', [2, 4])

    def test_disk_reserve(self):
        with patch('verify_compressed_states.shutil.disk_usage', return_value=type('Disk', (), {'free': 750 * 1024**2})()):
            with self.assertRaisesRegex(ValueError, 'Insufficient disk'): self.verify()

    def test_nonfinite_and_cleanup(self):
        self.state['replay'][0]['x'][0] = float('nan'); self.encode()
        with self.assertRaisesRegex(ValueError, 'Nonfinite'): self.verify()
        self.assertFalse(list(self.root.glob('.verify-state-*')))

    def test_wrong_source(self):
        self.state['source_files'] = {}; self.encode()
        with self.assertRaisesRegex(ValueError, 'Source mismatch'): self.verify()

    def test_final_alignment(self):
        np.savez(self.root / 'weights.npz', w=np.array([3.]))
        with self.assertRaisesRegex(ValueError, 'Weight mismatch'): self.verify()

    def test_configuration_runtime_counters_and_rng_rejected(self):
        for field, value in [('arguments', {}), ('torch_version', 'wrong'), ('totals', {'games': 3, 'optimizer_updates': 4}), ('numpy_rng', {}), ('optimizer', {'state': {}, 'param_groups': []})]:
            with self.subTest(field=field):
                original = self.state[field]
                self.state[field] = value; self.encode()
                with self.assertRaises(ValueError): self.verify()
                self.state[field] = original

    def test_partial_manifest(self):
        (self.root / 'compressed/manifest.json').write_text('{"states":')
        with self.assertRaises(ValueError): self.verify()

    def test_raw_digest_corruption(self):
        path = self.root / 'compressed/manifest.json'; manifest = json.loads(path.read_text())
        next(iter(manifest['states'].values()))['raw_sha256'] = '0' * 64
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'expected content'): self.verify()
        self.assertFalse(list(self.root.glob('.verify-state-*')))


if __name__ == '__main__': unittest.main()
