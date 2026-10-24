"""Synthetic retention/transport gate. Never a training or resume correctness test."""
import argparse
import hashlib
import importlib.util
import json
import shutil
import time
from pathlib import Path
import torch

FIELDS = {'arguments','environment','source_files','torch_version','initial','model',
          'optimizer','totals','replay','numpy_rng','torch_rng','cuda_rng'}

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()

def binding(path, expected):
    if digest(path) != expected: raise ValueError('Source hash mismatch')

def generate(output, retention, expected, payload_bytes):
    binding(retention, expected)
    if payload_bytes < 4 or payload_bytes > 2*1024**3: raise ValueError('Payload out of bounds')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(output).free < payload_bytes*3 + 128*1024**2:
        raise ValueError('Insufficient disk headroom')
    spec = importlib.util.spec_from_file_location('pinned_retention', retention)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    torch.set_num_threads(1)
    gen = torch.Generator().manual_seed(220023)
    # Nonconstant data: transfer compression must not turn a zero-filled fixture
    # into an unrealistically small recovery. CPU payload tests host serialization.
    payload = torch.rand(payload_bytes//4, generator=gen)
    state = dict(arguments={'synthetic':True}, environment={},
        source_files={'state_history.py':expected}, torch_version=str(torch.__version__),
        initial={'fixture':torch.tensor([0.])}, model={'fixture':torch.tensor([1.])},
        optimizer={'state':{},'param_groups':[]}, totals={},
        replay=[{'synthetic_payload':payload}], numpy_rng={},
        torch_rng=torch.get_rng_state(), cuda_rng=[])
    rows = []
    for milestone in (1,2):
        state['totals'] = {'games':milestone, 'optimizer_updates':milestone}
        start=time.monotonic()
        p=module.save_milestone(state, output, torch.save)
        elapsed=time.monotonic()-start
        rows.append({'path':str(p.relative_to(output)), 'bytes':p.stat().st_size,
                     'sha256':digest(p), 'games':milestone, 'save_seconds':elapsed})
        if rows[0]['sha256'] != digest(output/rows[0]['path']):
            raise ValueError('Old immutable milestone changed')
    report=dict(schema=1, scope='synthetic host serialization and recovery only',
        synthetic=True, payload_bytes=payload_bytes, torch_version=str(torch.__version__),
        retention_sha256=expected, gate_source_sha256=digest(__file__), milestones=rows,
        alias_sha256=digest(output/'resume-state.pt'),
        excludes=['actual training state completeness','CUDA serialization','GPU split resume equality',
                  'actual historical checkpoint recovery','training throughput'])
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

def verify(directory, expected_manifest):
    directory=Path(directory); manifest=directory/'manifest.json'
    binding(manifest, expected_manifest)
    report=json.loads(manifest.read_text())
    if report.get('synthetic') is not True or len(report['milestones']) != 2:
        raise ValueError('Invalid manifest scope/count')
    start=time.monotonic()
    for row in report['milestones']:
        relative=Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts: raise ValueError('Unsafe path')
        p=directory/relative
        if p.stat().st_size != row['bytes']: raise ValueError('Size mismatch')
        binding(p,row['sha256'])
        state=torch.load(p,map_location='cpu',weights_only=True)
        if set(state)!=FIELDS or state['arguments']!={'synthetic':True}:
            raise ValueError('State field/scope mismatch')
        if state['totals']!={'games':row['games'],'optimizer_updates':row['games']}:
            raise ValueError('Counter mismatch')
        if state['source_files'] != {'state_history.py':report['retention_sha256']}:
            raise ValueError('Source binding mismatch')
        if state['torch_version'] != report['torch_version']: raise ValueError('Stack mismatch')
        payload=state['replay'][0]['synthetic_payload']
        if payload.dtype!=torch.float32 or payload.numel()!=report['payload_bytes']//4:
            raise ValueError('Payload mismatch')
        del state, payload
    if report['alias_sha256'] != report['milestones'][-1]['sha256']:
        raise ValueError('Latest alias binding mismatch')
    # Alias need not be transferred; if present check it too.
    if (directory/'resume-state.pt').exists(): binding(directory/'resume-state.pt',report['alias_sha256'])
    return dict(verified=True, scope=report['scope'], bytes=sum(x['bytes'] for x in report['milestones']),
                verify_seconds=time.monotonic()-start, loader_torch_version=str(torch.__version__))

if __name__=='__main__':
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='command',required=True)
    g=sub.add_parser('generate'); g.add_argument('--output',required=True)
    g.add_argument('--retention',required=True); g.add_argument('--retention-sha256',required=True)
    g.add_argument('--payload-bytes',type=int,default=1014868120)
    v=sub.add_parser('verify'); v.add_argument('--directory',required=True); v.add_argument('--manifest-sha256',required=True)
    a=p.parse_args()
    result=(generate(a.output,a.retention,a.retention_sha256,a.payload_bytes) if a.command=='generate'
            else verify(a.directory,a.manifest_sha256))
    print(json.dumps(result,indent=2))
