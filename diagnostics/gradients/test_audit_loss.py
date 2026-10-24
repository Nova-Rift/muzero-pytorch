import os
from pathlib import Path
import tempfile
import unittest

import torch
from audit_loss import checked_source, GOOGLE_SHA, run


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reference = os.environ.get('REFERENCE_LEARNER')
        if not reference:
            raise unittest.SkipTest('Set REFERENCE_LEARNER to pinned Google learner.py')
        cls.result = run(Path(reference))

    def test_all_valid_relative_weights(self):
        case = self.result['cases']['all_valid']
        for contract in ('local','reference'):
            for head in ('policy','value','reward'):
                actual = torch.tensor(case[contract]['coefficients'][head])
                expected = torch.full((2,6), 1/6 if contract == 'local' else 1/5)
                if contract == 'reference':
                    expected[:,0] = 1
                if head == 'reward':
                    expected[:,0] = 0
                torch.testing.assert_close(actual, expected)

    def test_distinct_component_masks_and_zero_recurrence(self):
        case = self.result['cases']['component_masks']
        for contract in ('local','reference'):
            for head in ('policy','value','reward'):
                masks = torch.tensor(case['masks'][head])
                expected = masks / 6
                if contract == 'reference':
                    expected = masks / masks[:,1:].sum(-1).clamp_min(1)[:,None]
                    expected[:,0] = masks[:,0]
                actual = torch.tensor(case[contract]['coefficients'][head])
                torch.testing.assert_close(actual, expected)
                self.assertTrue(torch.isfinite(actual).all())

    def test_hidden_path_first_transition_and_later_halving(self):
        for case in self.result['hidden_isolated']:
            t = case['target_step']
            for contract in ('local','reference'):
                expected = .5 * (.5 ** t) / 6 if contract == 'local' else .5 * (.5 ** max(t-1,0))
                torch.testing.assert_close(torch.tensor(case[contract]['root_hidden_gradient']),
                                           torch.full((2,), expected))
            # Hidden probes do not change zero logits or their loss values.
            self.assertGreater(case['reference']['loss_forward'], 0)

    def test_forward_values_do_not_equal_gradient_weighting(self):
        # Reference scale_gradient preserves forward values; local divides values too.
        case = self.result['cases']['all_valid']
        self.assertAlmostEqual(case['reference']['loss_forward'],
                               6 * case['local']['loss_forward'], places=10)

    def test_source_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'learner.py'
            p.write_text('def compute_loss(): pass\n')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                checked_source(p, GOOGLE_SHA)


if __name__ == '__main__':
    unittest.main()
