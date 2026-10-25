"""External EGO continuation gates. Never changes frozen training source."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys

COMMIT = '14fece1209067004f4c3936a6171bc40da55577d'
RAW = '50b360b503b67b13474bd901c082b4168d19e8a8c8c2d25ebd910974bb24bda9'
ENCODED = '8ebbe6203897735e2cccee748e1e444d4b0b2941ab60836b889db0a64a2b16e4'
SHA_INITIAL = 'ef8ed2fb76948db444f838d3a769870af89528f5861df43d63a327b76ee5fc0a'
SHA_WEIGHTS = '0e263ee21ce56721dad3479648136a725b512138e1be4c73e30e689d6e49cf08'
NAME = 'games-2048-updates-2048.pt'
ALLOW = {'site', 'output', 'resume_state', 'games', 'max_seconds'}

def require(value, message):
    if not value: raise ValueError(message)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def validate_manifest(m, receipt):
    require(m['source_commit']==COMMIT and m['source_provenance']['files']==receipt['files'] and receipt['commit']==COMMIT and receipt['dirty'] is False, 'Frozen source identity differs')
    require(m['source_provenance']['dirty'] is False, 'Original source dirty')
    a=m['arguments']
    expected={'loss_contract':'baseline','games':2048,'updates_per_game':1,'batch':256,'seed':52731415,'aim_decoding':'relative','mirror_augment':False,'resume_state':None,'device':'cuda','replay_blocks':3}
    require(all(type(a.get(k)) is type(v) and a[k]==v for k,v in expected.items()), 'Original EGO scientific contract differs')
    require(m['torch']=='2.8.0+cu128' and m['python']=='3.12.3', 'Original runtime differs')

def training_command(manifest, source, site, output, state):
    a=dict(manifest['arguments']);a.update(site=str(site),output=str(output),resume_state=str(state),games=4096,max_seconds=600)
    command=['python',str(Path(source)/'arena/train_loss.py')]
    for key,value in a.items():
        if isinstance(value,bool):
            if value:command.append('--'+key.replace('_','-'))
        elif value is not None:command.extend(['--'+key.replace('_','-'),str(value)])
    return command

def input_gate(source, site, original, output):
    import platform
    import torch
    sys.path.insert(0,str(source/'diagnostics/retention'))
    from compressed_checkpoint import restore
    from verify_compressed_states import arrays_match
    from verify_training_states import FIELDS, finite
    receipt=json.loads((source/'.source-provenance.json').read_text())
    m=json.loads((original/'manifest.json').read_text());validate_manifest(m,receipt)
    for name,sha in receipt['files'].items():
        path=(source/name).resolve()
        require(path.is_relative_to(source.resolve()) and digest(path)==sha,'Frozen source changed: '+name)
    for name,sha in m['environment'].items():require(digest(site/name)==sha,'Environment changed: '+name)
    require(str(torch.__version__)==m['torch'] and platform.python_version()==m['python'],'Exact runtime unavailable')
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'One visible CUDA device required')
    require(digest(original/'weights.npz')==SHA_WEIGHTS and digest(original/'initial.npz')==SHA_INITIAL,'Original endpoint weights differ')
    encoded=json.loads((original/'compressed/manifest.json').read_text())['states'][NAME]
    require(encoded['raw_sha256']==RAW and encoded['compressed_sha256']==ENCODED and encoded['raw_bytes']==710894055,'Original checkpoint identity differs')
    compressed=original/'compressed'/(NAME+'.gz')
    require(digest(compressed)==ENCODED and compressed.stat().st_size==76602287,'Compressed input changed')
    require(not output.exists(),'Fresh gate output required')
    require(shutil.disk_usage(output.parent).free>encoded['raw_bytes']+2*1024**3,'Insufficient decode headroom')
    output.mkdir(); raw=output/'resume.pt';restore(compressed,raw,RAW,encoded['raw_bytes'])
    torch.set_num_threads(2)
    s=torch.load(raw,map_location='cpu',weights_only=True)
    require(set(s)==FIELDS and s['arguments']==m['arguments'] and s['source_files']==receipt['files'] and s['environment']==m['environment'] and s['torch_version']==m['torch'],'Saved full-state binding differs')
    require(all(s['totals'].get(k)==v for k,v in {'games':2048,'optimizer_updates':2048,'training_examples':524288}.items()),'Saved counters differ')
    require(s['optimizer']['state'] and s['optimizer']['param_groups'] and len(s['replay'])==3 and all(s['replay']),'Optimizer/replay missing')
    import numpy as np
    rng=np.random.default_rng();rng.bit_generator.state=s['numpy_rng'];torch.Generator().set_state(s['torch_rng'])
    require(len(s['cuda_rng'])==1,'CUDA RNG count differs');torch.cuda.set_rng_state_all(s['cuda_rng'])
    finite(s);arrays_match(original/'initial.npz',s['initial']);arrays_match(original/'weights.npz',s['model'])
    del s;gc.collect()
    result={'verified':True,'raw_sha256':digest(raw),'original_games':2048,'next_collection_seed':52731415+2048,'scope':'Verified exact input state and runtime; paired tiny CUDA split-resume remains separate prerequisite.'}
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')

def post_gate(source, original, training, output):
    sys.path.insert(0,str(source/'diagnostics/retention'))
    from verify_compressed_states import verify
    from evaluate_endpoint import validate_continuation
    m=json.loads((training/'manifest.json').read_text())
    validate_continuation(m,original,source)
    require(digest(training/'initial.npz')==SHA_INITIAL,'Initialization changed')
    result=verify(training,source/'.source-provenance.json','arena.train_loss','EGO4096',[2560,3072,3584,4096],'baseline',1)
    result['scope']='Post-training compressed recovery, full-state and exact continuation manifest; no playing-strength claim'
    output.write_text(json.dumps(result,indent=2)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['input','post']);p.add_argument('--source',type=Path,required=True);p.add_argument('--site',type=Path);p.add_argument('--original',type=Path,required=True);p.add_argument('--training',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.stage=='input':input_gate(a.source,a.site,a.original,a.output)
    else:post_gate(a.source,a.original,a.training,a.output)
if __name__=='__main__':main()
