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
        # Three loss-contract changes plus the explicit aim-decoding flag (egocentric-aim branch); nothing else.
        source = (ROOT/'arena/train_loss.py').read_text()
        # Reverse the scoped --partner-pool edits first (tests/test_partner_pool.py pins them against 14fece12).
        from test_partner_pool import PARTNER_EDITS
        for added, original in PARTNER_EDITS:
            self.assertEqual(source.count(added), 1, added[:80])
            source = source.replace(added, original)
        # Reverse the scoped --mirror-augment edits (arena/mirror.py; tests/test_mirror_augment.py pins their behaviour).
        mirror_edits = [
            ('from arena.mirror import ACTION as MIRROR_ACTION,mirror_observation,mirror_policy\n', ''),
            ('def history(observations,actions,times,agents,context,mirror=None):', 'def history(observations,actions,times,agents,context):'),
            ("    if mirror is None:\n        obs=observations[safe,agents[:,None]]*mask[...,None]\n        ai=times[:,None]-np.arange(context-1,0,-1)[None]\n        previous=actions[np.maximum(ai,0),agents[:,None]]*(ai>=0)/153.\n    else:\n        # --mirror-augment: mirror selected samples before padding, so padding frames stay zero.\n        flag=np.asarray(mirror,bool)[:,None,None]\n        obs=observations[safe,agents[:,None]];obs=np.where(flag,mirror_observation(obs),obs)*mask[...,None]\n        ai=times[:,None]-np.arange(context-1,0,-1)[None]\n        previous=actions[np.maximum(ai,0),agents[:,None]];previous=np.where(flag[...,0],MIRROR_ACTION[previous],previous)*(ai>=0)/153.\n",
             "    obs=observations[safe,agents[:,None]]*mask[...,None]\n    ai=times[:,None]-np.arange(context-1,0,-1)[None]\n    previous=actions[np.maximum(ai,0),agents[:,None]]*(ai>=0)/153.\n"),
            ('def replay_batch(block,rng,batch,context,unroll,device,mirror=False):', 'def replay_batch(block,rng,batch,context,unroll,device):'),
            ("    if not mirror:x=history(block['observation'],block['actions'],times,agents,context)\n", "    x=history(block['observation'],block['actions'],times,agents,context)\n"),
            ("    if mirror:\n        # --mirror-augment: half of each batch (per-sample coin drawn after every frozen RNG draw) is the\n        # east-west mirror of the sampled transition; batch size, update count and targets are unchanged.\n        flags=rng.integers(2,size=batch).astype(bool)\n        x=history(block['observation'],block['actions'],times,agents,context,flags)\n        a=np.where(flags[:,None],MIRROR_ACTION[a],a);p=np.where(flags[:,None,None],mirror_policy(p),p)\n", ''),
            ("    p.add_argument('--mirror-augment',action='store_true',help='training-only east-west mirror of half of every replay batch (arena/mirror.py); off reproduces the frozen trainer exactly')\n", ''),
            ('replay_batch(chosen,rng,a.batch,a.context,a.unroll,a.device,a.mirror_augment)', 'replay_batch(chosen,rng,a.batch,a.context,a.unroll,a.device)'),
        ]
        for added, original in mirror_edits:
            self.assertEqual(source.count(added), 1, added)
            source = source.replace(added, original)
        source = source.replace('from arena.rootweighted import model_class\n', '')
        source = source.replace('from arena.aim_decoding import MODES as AIM_MODES,wire_offset\n', '')
        source = source.replace(',aim_decoding="absolute"):\n    # Recorded actions stay model indices (0..152); only the command sent to the bridge carries the decoding offset.\n    aim_offset=wire_offset(aim_decoding)\n', '):\n')
        source = source.replace('        commands=(act+aim_offset).reshape(games,11);commands[finished]=-1\n', '        commands=act.reshape(games,11).copy();commands[finished]=-1\n')
        source = source.replace("    p.add_argument('--aim-decoding',choices=list(AIM_MODES),default='absolute',help='relative interprets aim bins as offsets from the bearing to the nearest live opponent; default keeps historical absolute angles')\n", '')
        source = source.replace('a.return_mode,a.timeout_bootstrap,a.aim_decoding)', 'a.return_mode,a.timeout_bootstrap)')
        source = source.replace("    p.add_argument('--loss-contract',choices=['baseline','root5-mass-v1'],default='baseline')\n", '')
        source = source.replace('model_class(a.loss_contract)(Config(', 'MuZero(Config(')
        self.assertEqual(source, (ROOT/'arena/train_selfplay.py').read_text())
        # The added field participates in the original generic resume-argument guard.
        guard = ast.parse((ROOT/'arena/train_loss.py').read_text())
        self.assertIn("restored['arguments'][key]", ast.unparse(guard))

if __name__ == '__main__': unittest.main()
