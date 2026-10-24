"""Verify a complete recovered compressed chain without retaining decoded states.

Entrypoint/arm are externally trusted labels, never inferred from checkpoint bytes.
"""
import argparse
import gc
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
import torch
from compressed_checkpoint import restore
from verify_training_states import FIELDS, digest, finite

RESERVE = 750 * 1024**2


def require(condition, message):
    if not condition:
        raise ValueError(message)


def arrays_match(path, saved):
    with np.load(path, allow_pickle=False) as arrays:
        require(set(arrays.files) == set(saved), 'Weight keys mismatch')
        for key in arrays.files:
            require(np.array_equal(arrays[key], saved[key].numpy()), 'Weight mismatch: ' + key)


def verify(root, source_path, entrypoint, arm, expected=(512, 1024, 1536, 2048)):
    root = Path(root)
    expected = list(expected)
    require(expected and expected == sorted(set(expected)) and all(type(x) is int and x > 0 for x in expected), 'Invalid expected chain')
    require(bool(entrypoint) and bool(arm), 'Trusted entrypoint and arm required')
    source = json.loads(Path(source_path).read_text())
    require(isinstance(source.get('files'), dict) and bool(source['files']), 'Empty source provenance')
    manifest = json.loads((root / 'manifest.json').read_text())
    report = json.loads((root / 'report.json').read_text())
    encoded = json.loads((root / 'compressed/manifest.json').read_text())
    names = [f'games-{n}-updates-{n}.pt' for n in expected]
    require(set(encoded['states']) == set(names), 'Incomplete or unexpected milestone chain')
    require(report['requested_run_completed'] is True and report['games'] == expected[-1], 'Incomplete training report')
    require(encoded['encoded_bytes'] == sum(r['compressed_bytes'] for r in encoded['states'].values()), 'Encoded total mismatch')
    torch.set_num_threads(2)
    rows = []
    for count, name in zip(expected, names):
        record = encoded['states'][name]
        require(record['raw_name'] == name, 'Raw filename mismatch')
        for field in ('raw_bytes', 'compressed_bytes'):
            require(type(record[field]) is int and record[field] > 0, 'Invalid size')
        compressed = root / 'compressed' / (name + '.gz')
        require(compressed.stat().st_size == record['compressed_bytes'] and digest(compressed) == record['compressed_sha256'], 'Compressed hash/size mismatch')
        require(shutil.disk_usage(root).free >= record['raw_bytes'] + RESERVE, 'Insufficient disk for decode plus 750 MiB reserve')
        # Private unique directory: cleanup cannot touch existing recovered raw files.
        with tempfile.TemporaryDirectory(prefix='.verify-state-', dir=root) as temporary:
            raw = Path(temporary) / name
            restore(compressed, raw, record['raw_sha256'], record['raw_bytes'])
            state = torch.load(raw, map_location='cpu', weights_only=True)
            try:
                require(set(state) == FIELDS, 'Saved field set mismatch')
                require(state['source_files'] == source['files'], 'Source mismatch')
                require(state['environment'] == manifest['environment'] and state['arguments'] == manifest['arguments'], 'Configuration mismatch')
                require(state['torch_version'] == manifest['torch'], 'Saved runtime mismatch')
                require(state['totals']['games'] == count and state['totals']['optimizer_updates'] == count, 'Counter mismatch')
                require(bool(state['optimizer']['state']) and bool(state['optimizer']['param_groups']), 'Missing optimizer state')
                require(len(state['replay']) == manifest['arguments'].get('replay_blocks', 3), 'Replay block count mismatch')
                require(all(isinstance(block, dict) and block for block in state['replay']), 'Empty replay block')
                require(isinstance(state['numpy_rng'], dict) and bool(state['numpy_rng']), 'Missing NumPy RNG')
                rng = np.random.default_rng(); rng.bit_generator.state = state['numpy_rng']
                generator = torch.Generator(); generator.set_state(state['torch_rng'])
                require(len(state['cuda_rng']) == (1 if manifest['arguments']['device'] == 'cuda' else 0), 'CUDA RNG count mismatch')
                for value in state['cuda_rng']:
                    require(isinstance(value, torch.Tensor) and value.dtype == torch.uint8 and value.ndim == 1 and value.numel() > 0, 'Malformed CUDA RNG')
                finite(state)
                arrays_match(root / 'initial.npz', state['initial'])
                if count == expected[-1]:
                    arrays_match(root / 'weights.npz', state['model'])
                rows.append(dict(games=count, optimizer_updates=count, raw_sha256=record['raw_sha256'], compressed_sha256=record['compressed_sha256'], saved_torch_version=state['torch_version'], verified=True))
            finally:
                del state
                gc.collect()
    return dict(verified=True, states=rows, loader_torch_version=str(torch.__version__), source_provenance_sha256=digest(source_path), trusted_entrypoint=entrypoint, trusted_arm=arm, scope='Recovery integrity and saved fields only. Entrypoint/arm supplied externally; source/config alone cannot infer search variant. No numerical resume equality or playing strength claim.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--source-provenance', type=Path, required=True)
    p.add_argument('--entrypoint', required=True)
    p.add_argument('--arm', required=True)
    p.add_argument('--expected', type=int, nargs='+', default=[512, 1024, 1536, 2048])
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    try:
        result = verify(a.root, a.source_provenance, a.entrypoint, a.arm, a.expected)
    except Exception as error:
        a.output.write_text(json.dumps(dict(verified=False, error=str(error)), indent=2) + '\n')
        raise
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
