import ast
from pathlib import Path
import unittest
from unittest.mock import patch
import sys
import torch
from model import Config, MuZero
from arena.rootweighted import RootWeightedMuZero, coefficients, model_class
from arena import train_loss, train_selfplay
from diagnostics.real_gradients.probe import terms, gradients

ROOT = Path(__file__).resolve().parents[1]


class RootWeightedTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)
        self.model = RootWeightedMuZero(Config(observations=4, actions=3, hidden=8, support=2)).double()
        self.batch = (torch.randn(3, 4, dtype=torch.float64), torch.randint(3, (3, 5)),
                      torch.randn(3, 5, dtype=torch.float64), torch.randn(3, 6, dtype=torch.float64),
                      torch.randn(3, 6, 3, dtype=torch.float64).softmax(-1),
                      torch.ones(3, 6, dtype=torch.float64), torch.ones(3, 6, dtype=torch.float64),
                      torch.ones(3, 5, dtype=torch.float64))

    def test_independent_coefficients(self):
        mask = torch.tensor([[1.,1,1,1,1,1], [1,0,0,0,0,0], [0,1,0,1,0,0], [0,0,0,0,0,0], [1,1,0,0,0,0]])
        expected = torch.tensor([[.5,.1,.1,.1,.1,.1], [1/6,0,0,0,0,0], [0,1/6,0,1/6,0,0], [0,0,0,0,0,0], [5/18,1/18,0,0,0,0]])
        torch.testing.assert_close(coefficients(mask), expected)
        torch.testing.assert_close(coefficients(mask.bool()), expected)
        torch.testing.assert_close(coefficients(mask).sum(1), mask.sum(1)/6)
        for value in (.5, float('nan'), -1):
            with self.assertRaises(ValueError): coefficients(torch.full((1,6),value))

    def test_matches_frozen_diagnostic_loss_and_every_gradient(self):
        for sparse in (False, True):
            if sparse:
                self.batch[5][1, 2:] = 0
                self.batch[6][2, :3] = 0
                self.batch[7][1, 3:] = 0
            actual = self.model.loss(*self.batch)
            expected = sum(row[0] for row in terms(self.model, self.batch, 5))
            self.assertTrue(torch.equal(actual, expected))
            ga, ge = gradients(actual.mean(), self.model), gradients(expected.mean(), self.model)
            for name in ga: self.assertTrue(torch.equal(ga[name], ge[name]), name)

    def test_baseline_exact_state_loss_and_gradient(self):
        self.assertIs(model_class('baseline'), MuZero)
        base = model_class('baseline')(self.model.config).double()
        base.load_state_dict(self.model.state_dict())
        original = MuZero(self.model.config).double()
        original.load_state_dict(self.model.state_dict())
        a, b = base.loss(*self.batch), original.loss(*self.batch)
        self.assertTrue(torch.equal(a, b))
        ga, gb = gradients(a.mean(), base), gradients(b.mean(), original)
        for name in ga: self.assertTrue(torch.equal(ga[name], gb[name]))
        self.assertEqual(set(base.state_dict()), set(self.model.state_dict()))

    def test_reward_only_and_empty_masks(self):
        self.batch[5].zero_(); self.batch[6].zero_()
        a = self.model.loss(*self.batch)
        b = MuZero.loss(self.model, *self.batch)
        self.assertTrue(torch.equal(a, b))
        ga, gb = gradients(a.mean(), self.model), gradients(b.mean(), self.model)
        for name in ga: self.assertTrue(torch.equal(ga[name], gb[name]), name)
        self.batch[7].zero_()
        loss = self.model.loss(*self.batch)
        self.assertTrue(torch.equal(loss, torch.zeros_like(loss)))
        self.assertTrue(all(torch.count_nonzero(g)==0 for g in gradients(loss.mean(), self.model).values()))
        self.batch[7][0,0] = .5
        with self.assertRaises(ValueError): self.model.loss(*self.batch)

    def test_default_masks(self):
        self.assertTrue(torch.equal(self.model.loss(*self.batch[:5]), self.model.loss(*self.batch)))

    def test_every_hidden_transition_half(self):
        class Chain:
            config = Config(observations=1, actions=2, hidden=1, support=1)
            def initial(self, x): return x, torch.cat((x, torch.zeros_like(x)), -1), x.expand(-1,3)*0
            def recurrent(self, x, a):
                h,p,v = self.initial(x)
                return h,v,p,v
        for depth in range(6):
            x = torch.zeros(1,1,dtype=torch.float64,requires_grad=True)
            mask = torch.zeros(1,6,dtype=torch.float64); mask[:,depth] = 1
            b = (x,torch.zeros(1,5,dtype=torch.long),torch.zeros(1,5),torch.zeros(1,6),torch.tensor([[[1.,0.]]*6]),mask,torch.zeros_like(mask),torch.zeros(1,5))
            loss = RootWeightedMuZero.loss(Chain(), *b).sum()
            self.assertAlmostEqual(float(torch.autograd.grad(loss,x)[0]), -.5*.5**depth/6, places=12)

    def test_actual_main_dispatch_and_original_search(self):
        self.assertIs(train_loss.search_batch, train_selfplay.search_batch)
        import tempfile
        class Selected(Exception): pass
        for contract, cls in [('baseline', MuZero), ('root5-mass-v1', RootWeightedMuZero)]:
            def select(name):
                self.assertEqual(name, contract)
                self.assertIs(model_class(name), cls)
                raise Selected
            with tempfile.TemporaryDirectory() as tmp:
                args = ['train_loss', '--site', tmp, '--output', tmp+'/out', '--loss-contract', contract, '--device', 'cpu']
                with patch.object(sys, 'argv', args), patch.object(train_loss, 'model_class', select):
                    with self.assertRaises(Selected): train_loss.main()

    def test_cross_contract_resume_rejected_before_state_load(self):
        import argparse
        import tempfile
        original_parse = argparse.ArgumentParser.parse_args
        captured = {}
        def parse(parser, *args, **kwargs):
            result = original_parse(parser, *args, **kwargs)
            captured.update(vars(result))
            return result
        def restored(*args, **kwargs):
            saved = dict(captured)
            saved['loss_contract'] = 'root5-mass-v1' if saved['loss_contract'] == 'baseline' else 'baseline'
            return {'arguments': saved}
        for contract in ('baseline', 'root5-mass-v1'):
            with tempfile.TemporaryDirectory() as tmp:
                args = ['train_loss', '--site', tmp, '--output', tmp+'/out', '--loss-contract', contract,
                        '--device', 'cpu', '--hidden', '8', '--context', '2', '--resume-state', tmp+'/state.pt']
                with patch.object(sys, 'argv', args), patch.object(argparse.ArgumentParser, 'parse_args', parse), patch.object(torch, 'load', restored):
                    with self.assertRaisesRegex(ValueError, 'Resume setting changed: loss_contract'):
                        train_loss.main()

    def test_trainer_only_three_scoped_changes(self):
        source = (ROOT/'arena/train_loss.py').read_text()
        source = source.replace('from arena.rootweighted import model_class\n', '')
        source = source.replace("    p.add_argument('--loss-contract',choices=['baseline','root5-mass-v1'],default='baseline')\n", '')
        source = source.replace('model_class(a.loss_contract)(Config(', 'MuZero(Config(')
        self.assertEqual(source, (ROOT/'arena/train_selfplay.py').read_text())
        # The added field participates in the original generic resume-argument guard.
        guard = ast.parse((ROOT/'arena/train_loss.py').read_text())
        self.assertIn("restored['arguments'][key]", ast.unparse(guard))

if __name__ == '__main__': unittest.main()
