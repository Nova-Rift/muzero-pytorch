"""Synthetic finite-horizon input counterexample; no gameplay, fitting or training."""
import argparse
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np

M8_TRAINER_SHA256 = '9d373ae8ef63d131b003a87db992e4429e14fa38eaac5e4483567fe3971ca18a'
CONTEXT = 64
LIMIT = 256


def load_history(source):
    """Import the real historical function, refusing a changed/wrong module."""
    source = Path(source).resolve()
    trainer = source / 'arena/train_selfplay.py'
    if hashlib.sha256(trainer.read_bytes()).hexdigest() != M8_TRAINER_SHA256:
        raise ValueError('historical trainer hash mismatch')
    sys.path.insert(0, str(source))
    module = importlib.import_module('arena.train_selfplay')
    if Path(inspect.getfile(module.history)).resolve() != trainer:
        raise ValueError('history imported from another source directory; use a fresh process')
    return module.history


def remaining_clock(times, limit=LIMIT):
    """Decision t precedes transition t; t=limit has zero remaining transitions."""
    if isinstance(limit, (bool, np.bool_)) or not isinstance(limit, (int, np.integer)) or limit <= 0:
        raise ValueError('limit must be a positive integer')
    times = np.asarray(times)
    if times.ndim != 1 or times.dtype.kind not in 'iu' or np.any(times < 0) or np.any(times > limit):
        raise ValueError('times must be a one-dimensional integer array within [0, limit]')
    return ((limit - times) / limit).astype(np.float32)[:, None]


def finite_horizon_return(remaining, reward_after):
    """Analytic reward 1 on transition reward_after (one-based), discount 1, zero cutoff."""
    for value in (remaining, reward_after):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError('transition counts must be integers')
    if remaining < 0 or reward_after < 1:
        raise ValueError('remaining must be nonnegative and reward_after positive')
    return float(reward_after <= remaining)


def synthetic_windows(times, context=CONTEXT):
    """Two synthetic agents with identical trailing windows, distinct older prefixes."""
    times = np.asarray(times)
    remaining_clock(times)
    if len(times) != 2 or context != CONTEXT or np.any(times < context - 1):
        raise ValueError('fixture requires two full 64-observation windows')
    observations = np.full((LIMIT + 1, 2, 64), -0.5, dtype=np.float32)
    actions = np.full((LIMIT, 2), 152, dtype=np.int64)
    # Nonconstant suffix catches order, index, action normalization and agent mistakes.
    suffix_obs = np.arange(context * 64, dtype=np.float32).reshape(context, 64) / 4096
    suffix_actions = (np.arange(context - 1, dtype=np.int64) * 7) % 153
    for agent, time in enumerate(times):
        observations[:time-context+1, agent] = -0.25 * (agent + 1)
        observations[time-context+1:time+1, agent] = suffix_obs
        actions[time-context+1:time, agent] = suffix_actions
    return observations, actions, suffix_obs, suffix_actions


def run(history):
    times = np.array([226, 246], dtype=np.int64)
    observations, actions, suffix_obs, suffix_actions = synthetic_windows(times)
    encoded = history(observations, actions, times, np.arange(2), CONTEXT)
    expected = np.concatenate((suffix_obs.ravel(), suffix_actions / 153)).astype(np.float32)
    np.testing.assert_array_equal(encoded, np.stack((expected, expected)))
    clock = remaining_clock(times)
    augmented = np.concatenate((encoded, clock), axis=1)
    assert not np.array_equal(augmented[0], augmented[1])
    returns = [finite_horizon_return(LIMIT-int(t), 20) for t in times]
    assert returns == [1., 0.]
    return {
        'kind': 'analytic synthetic counterexample, not an actual game trace',
        'historical_trainer_sha256': M8_TRAINER_SHA256,
        'historical_source_revision': 'd10af2d72dbf03dfec113c0b6e7e47481c5cc88d',
        'context_observations': CONTEXT, 'context_actions': CONTEXT-1,
        'time_convention': 't is number of completed transitions; reward after 20 means on the 20th future transition',
        'global_decision_times': times.tolist(), 'remaining_transitions': [30, 10],
        'input_dimension': int(encoded.shape[1]), 'historical_inputs_identical': True,
        'input_sha256': hashlib.sha256(encoded[0].tobytes()).hexdigest(),
        'proposed_normalized_remaining_clock': clock[:, 0].tolist(),
        'proposed_inputs_identical': False, 'reward_after_transitions': 20,
        'discount': 1, 'timeout_bootstrap': 'zero', 'analytic_returns': returns,
        'new_games': 0, 'optimizer_updates': 0,
        'limitations': [
            'Synthetic windows are not asserted reachable in the arena or present in replay.',
            'This proves the rolling input need not determine global time; it does not prove causal weakness or improved playing strength.',
            'No explicit clock does not imply no time inference from observations or action history; early zero padding can reveal age.',
            'A clock-conditioned learned value may harm an anytime arena policy or mismatch a different evaluation cutoff.',
            'A new input contract requires trainer, replay, device-history, planner and browser adapter validation.',
            'Changing input size invalidates exact resume; explicitly choose and validate compatible-weight warm start versus restart.',
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = run(load_history(args.source))
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
