"""Check the available archived first-block evidence before any optimizer update.

The archive contains two games' trajectories from a 128-game collection batch.
This diagnostic never represents that evidence as full 128-trajectory equality.

Argument allowlist (candidate mode only). ``--allow-arg NAME`` excludes NAME from
the training-argument comparison inside ``validate_continuation``; accepted names
are exactly ``evaluate_endpoint.ALLOWED_ARGUMENTS`` (timeout_bootstrap, discount).
It relaxes nothing else: source hashes, config, environment, torch/python, clean
provenance, games 2048, ratio, seed, batch, loss contract, initial tensors and
every array/aggregate parity check stay strict. The receipt records the allowlist
and the actual tolerated argument values so evidence shows what was accepted.

Only arguments that provably do not change collection before the first update can
yield parity. Against frozen trainer 2e67c2b9b7652ccbe75dbed249708356f38c3608:
* ``timeout_bootstrap`` is consumed INSIDE ``collect`` (arena/train_loss.py:108-114):
  with ``continue`` and any unfinished game, ``values[length]`` is filled from
  ``model.initial`` and flows into ``targets`` (lines 116-122, via
  ``episode_returns`` in return mode ``episode``). The returned block therefore
  differs in ``targets`` for cutoff slots whenever the block has cutoffs; the
  archived baseline first block records 17 cutoffs of 128 games. Actions, policies,
  rewards, observations, valid/absorbing masks and non-timing aggregates are
  unchanged by the flag. So a ``continue`` candidate passes the argument check with
  this allowlist but still fails ``Array differs: targets`` unless the block has
  zero cutoffs. That failure is correct and must not be suppressed.
* ``discount`` reaches collection through the model config (train_loss.py:176),
  the MCTS backups (batched_search_exploration.py:47,72,79,82) and the target
  weights (train_loss.py:119,121). It is accepted by the CLI for the argument
  relaxation only; the strict ``config`` comparison still rejects a changed
  discount, and search parity would fail regardless.
Reference mode accepts no allowlist and is unchanged.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from evaluate_endpoint import ALLOWED_ARGUMENTS, validate_continuation

ARCHIVE = {
    'initial.npz': 'ef8ed2fb76948db444f838d3a769870af89528f5861df43d63a327b76ee5fc0a',
    'trace-0.npz': '0fbcd69c70c58491329b61f4ef7f5225afd3bb1531e06c8c45bb969e8cc819fe',
    'manifest.json': '1a4b4160722e5ce844299d0d7843483ee32d8da5fb74e65e1ba8b11e05ac0e29',
    'metrics.jsonl': '201450e3ebefdced5cca158e9032a681aef624fde7a95abbc8e4eebedbb7f399',
}
TIMINGS = {'search', 'environment', 'history'}


def bind_archive(baseline):
    for name, expected in ARCHIVE.items():
        if hashlib.sha256((baseline/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Archived evidence changed: '+name)


def equal_arrays(actual, expected):
    if set(actual) != set(expected):
        raise ValueError('Array keys differ')
    for key in expected:
        x, y = actual[key], expected[key]
        if (x.dtype != y.dtype or x.shape != y.shape or
                not np.isfinite(x).all() or not np.isfinite(y).all() or
                not np.array_equal(x, y)):
            raise ValueError('Array differs: '+key)
    return len(expected)


def compare_block(block, stats, baseline):
    with np.load(baseline/'trace-0.npz', allow_pickle=False) as expected:
        count = equal_arrays({k:v[:,:22] for k,v in block.items() if k != 'indices'}, expected)
    old = json.loads((baseline/'metrics.jsonl').read_text().splitlines()[0])['batch_seconds']
    actual = {k:v for k,v in stats.items() if k not in TIMINGS}
    expected = {k:v for k,v in old.items() if k not in TIMINGS}
    if actual != expected or actual.get('games') != 128:
        raise ValueError('First-block non-timing aggregates differ')
    return dict(trace_arrays=count, trajectory_games=2, collection_games=128,
                full_128_trajectory_parity=False, aggregate_fields=len(expected))


class ReferenceComplete(Exception):
    """Deliberate collection-only exit before the trainer's update loop."""


def resolve_allowlist(names, mode):
    """Return the sorted allowlist tuple; reject unknown names and reference use."""
    names = tuple(names or ())
    unknown = sorted(set(names) - ALLOWED_ARGUMENTS)
    if unknown:
        raise ValueError('Unsupported --allow-arg: ' + ', '.join(unknown))
    if names and mode != 'candidate':
        raise ValueError('Reference mode accepts no argument allowlist')
    return tuple(sorted(set(names)))


def tolerated_differences(allowed, candidate, **others):
    """Record the actual values of allowed arguments that differ from any comparison run."""
    recorded = {}
    for name in allowed:
        values = {'candidate': candidate.get(name), **{label: m.get(name) for label, m in others.items()}}
        if any(v != values['candidate'] for v in values.values()):
            recorded[name] = values
    return recorded


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install_gate(trainer, source, output, baseline, reference, mode, allowed=()):
    """Wrap only collection; no replacement model, source edits or RNG operations."""
    allowed = resolve_allowlist(allowed, mode)
    original = trainer.collect
    checked = []
    def collect(*args, **kwargs):
        if checked:
            return original(*args, **kwargs)
        manifest = json.loads((output/'manifest.json').read_text())
        validation = copy.deepcopy(manifest)
        if mode == 'reference':
            if validation['arguments'].get('updates_per_game') != 1:
                raise ValueError('Reference must use frozen baseline ratio1')
            validation['arguments']['updates_per_game'] = 2
        validate_continuation(validation, baseline, source, upg2=True, allowed_arguments=allowed)
        old = json.loads((baseline/'manifest.json').read_text())
        if manifest.get('gpu') != old.get('gpu') or not manifest.get('gpu'):
            raise ValueError('Archived GPU runtime differs')
        with np.load(output/'initial.npz', allow_pickle=False) as actual, np.load(baseline/'initial.npz', allow_pickle=False) as expected:
            initial_count = equal_arrays(actual, expected)
        if mode == 'candidate':
            receipt = json.loads((reference/'first-block-reference.json').read_text())
            if (receipt.get('complete') is not True or receipt.get('optimizer_updates') != 0 or
                    receipt.get('diagnostic_sha256') != digest(Path(__file__)) or
                    receipt.get('archive_sha256') != ARCHIVE):
                raise ValueError('Invalid reference completion receipt')
            for name in ('initial.npz', 'manifest.json', 'full-block.npz'):
                if digest(reference/name) != receipt['sha256'][name]:
                    raise ValueError('Reference changed: '+name)
            ref_manifest = json.loads((reference/'manifest.json').read_text())
            validate_continuation(manifest, reference, source, upg2=True, allowed_arguments=allowed)
            if ref_manifest.get('gpu') != manifest.get('gpu'):
                raise ValueError('Reference GPU differs')
            with np.load(output/'initial.npz', allow_pickle=False) as actual, np.load(reference/'initial.npz', allow_pickle=False) as expected:
                equal_arrays(actual, expected)
        block, stats = original(*args, **kwargs)
        result = compare_block(block, stats, baseline)
        result.update(initial_tensors=initial_count, checked_before_first_update=True,
                      archive_sha256=ARCHIVE, mode=mode, diagnostic_sha256=digest(Path(__file__)))
        if mode == 'reference':
            np.savez_compressed(output/'full-block.npz', **block)
            result.update(complete=True, optimizer_updates=0,
                          sha256={name:digest(output/name) for name in ('initial.npz','manifest.json','full-block.npz')})
            (output/'first-block-reference.json').write_text(json.dumps(result, indent=2)+'\n')
            checked.append(result)
            raise ReferenceComplete()
        with np.load(reference/'full-block.npz', allow_pickle=False) as expected:
            result['full_block_arrays'] = equal_arrays(block, expected)
        result.update(full_128_trajectory_parity=True,
                      reference_receipt_sha256=digest(reference/'first-block-reference.json'),
                      reference_sha256=receipt['sha256'],
                      argument_allowlist=list(allowed),
                      tolerated_argument_differences=tolerated_differences(
                          allowed, manifest.get('arguments', {}),
                          baseline=old.get('arguments', {}), reference=ref_manifest.get('arguments', {})))
        (output/'first-block-gate.json').write_text(json.dumps(result, indent=2)+'\n')
        checked.append(result)
        return block, stats
    trainer.collect = collect
    return checked


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['reference','candidate'], required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--allow-arg', action='append', default=[], metavar='NAME',
                   help='candidate only; exclude NAME from the training-argument comparison. '
                        'Accepted: ' + ', '.join(sorted(ALLOWED_ARGUMENTS)) + '. Array parity stays strict.')
    p.add_argument('trainer_args', nargs=argparse.REMAINDER)
    a = p.parse_args()
    allowed = resolve_allowlist(a.allow_arg, a.mode)
    bind_archive(a.baseline)
    source = a.source.resolve()
    sys.path.insert(0, str(source))
    from arena import train_loss
    if Path(train_loss.__file__).resolve() != source/'arena/train_loss.py':
        raise ValueError('Wrong trainer import')
    command = a.trainer_args[1:] if a.trainer_args[:1] == ['--'] else a.trainer_args
    # Output has one authoritative binding; reject duplicate spellings.
    if any(x == '--output' or x.startswith('--output=') for x in command):
        raise ValueError('Supply output only to gate wrapper')
    if a.mode == 'reference' and a.output.resolve() != a.reference.resolve():
        raise ValueError('Reference mode output must equal reference directory')
    if a.mode == 'candidate' and a.output.resolve() == a.reference.resolve():
        raise ValueError('Candidate must use a separate output')
    checked = install_gate(train_loss, source, a.output, a.baseline, a.reference, a.mode, allowed)
    sys.argv = [str(source/'arena/train_loss.py'), '--output', str(a.output), *command]
    try:
        train_loss.main()
    except ReferenceComplete:
        if a.mode != 'reference':
            raise
    if not checked:
        raise ValueError('No successful first-block gate')


if __name__ == '__main__':
    main()
