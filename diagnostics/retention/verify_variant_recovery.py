"""Session 33 production recovery binding; unchanged compressed verifier does decoding.

Run sequentially on a host (one shared --lock file for both arms). This verifies
saved full states, not numerical CUDA resume or playing strength.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
from pathlib import Path
from evaluate_endpoint import validate_declared
from verify_compressed_states import verify, require, digest, RESERVE

PIN = '14fece1209067004f4c3936a6171bc40da55577d'
COUNTS = [512, 1024, 1536, 2048]
SCRATCH_BOUND = 1250000000

def expected_arguments(arm):
    require(arm in ('ego', 'mir'), 'Unknown arm')
    loop, label = ('loop-a', 'EGO2048') if arm == 'ego' else ('loop-b', 'MIR2048')
    return dict(loss_contract='baseline', site='/workspace/experiment/site',
        output=f'/workspace/runs/{loop}/{label}', resume_state=None, games=2048,
        parallel=128, replay_blocks=3, workers=4, hidden=448, context=64,
        simulations=32, batch=256, unroll=5, return_mode='episode',
        timeout_bootstrap='zero', root_candidates=8,
        aim_decoding='relative' if arm == 'ego' else 'absolute', mirror_augment=arm == 'mir',
        cadence='homepage', agents=2, policy_floor=0.0, noise_fraction=0.25,
        discount=1.0, limit=256, updates_per_game=1, seed=52731415,
        max_seconds=850.0, device='cuda', device_history=True, cuda_graph=True, tf32=True)

def bind(root, source, provenance, baseline, contract_path, arm):
    manifest=json.loads((root/'manifest.json').read_text())
    trusted=json.loads(provenance.read_text())
    require(trusted.get('commit') == PIN and trusted.get('dirty') is False, 'Wrong or dirty source pin')
    require(manifest.get('source_commit') == PIN, 'Manifest source pin mismatch')
    require(manifest.get('source_provenance', {}).get('files') == trusted.get('files'), 'Manifest source files mismatch')
    actual=manifest['arguments']; expected=expected_arguments(arm)
    require(set(actual)==set(expected), 'Argument keys mismatch')
    require(all(type(actual[k]) is type(v) and actual[k]==v for k,v in expected.items()), 'Production argument/flag mismatch')
    contract=json.loads(contract_path.read_text())
    declared=['aim_decoding' if arm=='ego' else 'mirror_augment']
    original=validate_declared(manifest, baseline, source, declared, contract)
    require(manifest.get('gpu') and manifest['gpu']==original.get('gpu'), 'GPU evidence mismatch')
    gate=json.loads((root/'variant-collection-gate.json').read_text())
    require(gate.get('complete') is True and gate.get('checked_before_first_update') is True and gate.get('variant')==arm and gate.get('source_contract')==contract, 'Missing or wrong collection gate')
    require(gate.get('initial_sha256')==digest(root/'initial.npz')==digest(baseline/'initial.npz'), 'Initial archive alignment mismatch')
    return manifest

def run(root, source, provenance, baseline, contract_path, arm):
    root, source, provenance, baseline, contract_path=map(Path,(root,source,provenance,baseline,contract_path))
    bind(root,source,provenance,baseline,contract_path,arm)
    report=json.loads((root/'report.json').read_text())
    require(report.get('checkpoint_sha256')==digest(root/'weights.npz'), 'Report final checkpoint hash mismatch')
    encoded=json.loads((root/'compressed/manifest.json').read_text())
    names=[f'games-{n}-updates-{n}.pt' for n in COUNTS]
    require(set(encoded['states'])==set(names), 'Incomplete milestone chain')
    peak=max(row['raw_bytes'] for row in encoded['states'].values())
    require(type(peak) is int and 0 < peak <= SCRATCH_BOUND, 'Decode exceeds admitted 1.25GB scratch; reassess storage')
    times=[encoded['states'][name].get('saved_at_unix') for name in names]
    require(all(type(t) in (int,float) and math.isfinite(t) and t>0 for t in times) and times==sorted(times), 'Missing or invalid checkpoint completion times')
    result=verify(root,provenance,'arena.train_loss',arm.upper()+'2048',COUNTS,'baseline',1)
    paths=['manifest.json','report.json','initial.npz','weights.npz','variant-collection-gate.json','compressed/manifest.json']+[f'compressed/{n}.gz' for n in names]
    result.update(arm=arm,source_commit=PIN,source_contract_sha256=digest(contract_path),
        training_completed_at=datetime.fromtimestamp(times[-1],timezone.utc).isoformat(),
        completion_evidence='compressed/manifest.json final raw checkpoint saved_at_unix captured by encoder; remote host clock',
        decode_peak_bytes=peak, decode_required_free_bytes=peak+RESERVE,
        evidence=[dict(path=n,bytes=(root/n).stat().st_size,sha256=digest(root/n)) for n in paths],
        scope='Production source/argument binding and complete recovered state integrity. No CUDA split-resume equality or playing strength claim.')
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','source','source-provenance','baseline','contract','output','lock'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--arm',choices=['ego','mir'],required=True);a=p.parse_args()
    # The coordinator supplies one common lock path for every local arm invocation.
    with a.lock.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            result=run(a.root,a.source,a.source_provenance,a.baseline,a.contract,a.arm)
        except Exception as error:
            a.output.write_text(json.dumps(dict(verified=False,error=str(error)),indent=2)+'\n')
            raise
        a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(verified=True,arm=a.arm,states=len(result['states']))))

if __name__=='__main__':main()
