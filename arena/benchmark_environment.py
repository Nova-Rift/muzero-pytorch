import argparse,json,time,platform,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from arena.environment import Pool
p=argparse.ArgumentParser();p.add_argument('--site',required=True);p.add_argument('--games',type=int,default=64);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
rng=np.random.default_rng(20261027);started=time.perf_counter();pool=Pool(a.site,a.workers)
try:
 state=pool.reset(range(27000000,27000000+a.games));active=np.ones(a.games,bool);steps=transitions=0
 for step in range(256):
  actions=rng.integers(153,size=(a.games,11));actions[~active]=-1
  steps+=int(active.sum());transitions+=int((state['alive']&active[:,None]).sum())
  state=pool.step(actions);active &= ~state['finished']
  if not active.any():break
 elapsed=time.perf_counter()-started
 print(json.dumps(dict(kind='random-action environment-only ceiling; NO neural training',games=a.games,natural_terminal=int((~active).sum()),cutoffs=int(active.sum()),joint_steps=steps,active_transitions=transitions,seconds=elapsed,games_per_second=a.games/elapsed,projected_10000_seconds=10000*elapsed/a.games,workers=a.workers,cpu=platform.processor())))
finally:pool.close()
