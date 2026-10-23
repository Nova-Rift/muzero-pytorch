"""Set M8_SOURCE to the archived model source; these tests never collect games."""
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from horizon_contract import (CONTEXT, finite_horizon_return, load_history,
                              remaining_clock, run, synthetic_windows)


class ClockTests(unittest.TestCase):
    def test_boundary_and_shape(self):
        np.testing.assert_array_equal(remaining_clock([0, 1, 255, 256]),
                                      np.array([[1], [255/256], [1/256], [0]], np.float32))

    def test_malformed_time(self):
        for bad in ([-1], [257], [1.0], [float('nan')], [True], [[1]], 1, ['1']):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                remaining_clock(bad)
        for bad in (0, -1, 256., True):
            with self.subTest(limit=bad), self.assertRaises(ValueError):
                remaining_clock([0], bad)

    def test_return_cutoff_includes_last_transition(self):
        self.assertEqual(finite_horizon_return(30, 20), 1)
        self.assertEqual(finite_horizon_return(10, 20), 0)
        self.assertEqual(finite_horizon_return(20, 20), 1)
        self.assertEqual(finite_horizon_return(19, 20), 0)
        self.assertEqual(finite_horizon_return(0, 1), 0)
        for remaining, delay in ((-1, 1), (1, 0), (1., 2), (True, 2), (1, False)):
            with self.assertRaises(ValueError):
                finite_horizon_return(remaining, delay)

    def test_source_hash_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = Path(tmp) / 'arena/train_selfplay.py'
            trainer.parent.mkdir()
            trainer.write_text('def history(): pass\n')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                load_history(tmp)


class HistoricalEncodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Required rather than skipped: a test pass must include historical encoding.
        cls.history = staticmethod(load_history(os.environ['M8_SOURCE']))

    def test_late_windows_and_clock(self):
        result = run(self.history)
        self.assertTrue(result['historical_inputs_identical'])
        self.assertFalse(result['proposed_inputs_identical'])
        self.assertEqual(result['input_dimension'], 4159)
        self.assertEqual(result['analytic_returns'], [1., 0.])

    def test_padding_and_first_full_window(self):
        observations = np.ones((65, 2, 64), np.float32)
        actions = np.full((64, 2), 51, np.int64)
        for t in (0, 1, 62, 63, 64):
            encoded = self.history(observations, actions, np.array([t]), np.array([1]), CONTEXT)[0]
            observed = min(t+1, CONTEXT)
            expected_obs = np.zeros((CONTEXT, 64), np.float32)
            expected_obs[-observed:] = 1
            previous = min(t, CONTEXT-1)
            expected_actions = np.zeros(CONTEXT-1, np.float32)
            if previous:
                expected_actions[-previous:] = 51/153
            np.testing.assert_array_equal(encoded[:4096], expected_obs.ravel())
            np.testing.assert_array_equal(encoded[4096:], expected_actions)
        # Padding makes early times different even under constant observations/actions.
        encoded = self.history(observations, actions, np.array([0, 1]), np.array([0, 1]), CONTEXT)
        self.assertFalse(np.array_equal(encoded[0], encoded[1]))

    def test_prefix_irrelevance_and_suffix_sensitivity(self):
        times = np.array([226, 246])
        obs, actions, _, _ = synthetic_windows(times)
        original = self.history(obs, actions, times, np.arange(2), CONTEXT)
        obs[:163, 0] = 0.125
        actions[:163, 0] = 1
        np.testing.assert_array_equal(original, self.history(obs, actions, times, np.arange(2), CONTEXT))
        obs[226, 0, 0] += 0.25
        changed = self.history(obs, actions, times, np.arange(2), CONTEXT)
        self.assertNotEqual(original[0, 4032], changed[0, 4032])
        actions[245, 1] = (actions[245, 1]+1) % 153
        changed = self.history(obs, actions, times, np.arange(2), CONTEXT)
        self.assertNotEqual(original[1, -1], changed[1, -1])


if __name__ == '__main__':
    unittest.main()
