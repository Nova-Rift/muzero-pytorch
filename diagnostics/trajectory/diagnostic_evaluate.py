"""Entry point for frozen evaluator plus bounded observation-only trajectories."""
import argparse,hashlib,json,sys
from pathlib import Path
from trajectory_recorder import RecordingPool
p=argparse.ArgumentParser(add_help=False)
p.add_argument('--model-source',type=Path,required=True);p.add_argument('--trace-directory',type=Path,required=True)
a,rest=p.parse_known_args()
expected='9b08fc5f16928fa29098cb1dbab1f15e27d7950599216954f0c29cd764fdcec5'
source=a.model_source/'tournament/run.py'
if hashlib.sha256(source.read_bytes()).hexdigest()!=expected:raise RuntimeError('Frozen evaluator source changed')
a.trace_directory.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(a.model_source))
from tournament import run
original=run.Pool;calls=0
manifest=dict(evaluator_sha256=expected,recorder_sha256=hashlib.sha256(Path(__file__).with_name('trajectory_recorder.py').read_bytes()).hexdigest(),scope='Diagnostic only; separate protocol and no automatic ladder import',max_sampled_games_per_batch=8,max_decisions=256)
(a.trace_directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
def factory(*args,**kwargs):
 global calls
 calls+=1
 inner=original(*args,**kwargs)
 try:return RecordingPool(inner,a.trace_directory/f'batch-{calls:03d}.jsonl',8,256)
 except BaseException:inner.close();raise
run.Pool=factory;sys.argv=[str(source),*rest];run.main()
