"""Observation-only replay wrapper; recorded games are diagnostic, not new confirmation."""
import argparse,hashlib,json,runpy,sys
from pathlib import Path
from trajectory_recorder import RecordingPool

def main():
 p=argparse.ArgumentParser(add_help=False)
 p.add_argument('--trace-directory',type=Path,required=True)
 a,rest=p.parse_known_args()
 # The evaluator owns argument validation, frozen source checks and outcome accounting.
 source=Path(rest[rest.index('--source')+1])
 sys.path.insert(0,str(source))
 from tournament import run
 original=run.Pool
 evaluator=Path(__file__).with_name('evaluate_split.py')
 a.trace_directory.mkdir(parents=True,exist_ok=False)
 hashes={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ('trace_split.py','trajectory_recorder.py','evaluate_split.py','split_controller.py','split_search.py')}
 (a.trace_directory/'manifest.json').write_text(json.dumps(dict(schema=1,source=hashes,scope='Observation-only development replay; not independent confirmation',max_games=8,max_steps=256),indent=2)+'\n')
 calls=0
 def factory(*args,**kwargs):
  nonlocal calls
  calls+=1;inner=original(*args,**kwargs)
  try:return RecordingPool(inner,a.trace_directory/f'batch-{calls:03d}.jsonl',8,256)
  except BaseException:inner.close();raise
 old_argv=sys.argv
 try:
  run.Pool=factory;sys.argv=[str(evaluator),*rest]
  runpy.run_path(str(evaluator),run_name='__main__')
 finally:run.Pool=original;sys.argv=old_argv
if __name__=='__main__':main()
