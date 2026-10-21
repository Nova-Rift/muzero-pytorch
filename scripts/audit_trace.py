"""Replay a saved two-game trace against canonical physics, without a policy."""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from arena.environment import Pool
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--site',required=True);p.add_argument('--trace',required=True)
a=p.parse_args();meta=json.loads((a.run/'manifest.json').read_text());offset=int(Path(a.trace).stem.split('-')[-1])
with np.load(a.run/a.trace,allow_pickle=False) as z:data={k:z[k] for k in z.files}
games=data['actions'].shape[1]//11
pool=Pool(a.site,agents=meta['arguments']['agents'],cadence=meta['arguments']['cadence'])
try:
 s=pool.reset(range(meta['arguments']['seed']+offset,meta['arguments']['seed']+offset+games))
 for t,actions in enumerate(data['actions']):
  np.testing.assert_array_equal(s['observation'].reshape(-1,64),data['observation'][t])
  valid=data['valid'][t];pol=data['policies'][t][valid]
  np.testing.assert_allclose(pol.sum(-1),1,atol=1e-6)
  commands=actions.reshape(games,11).copy();commands[~valid.reshape(games,11).any(-1)]=-1
  s=pool.step(commands)
  expected=s['reward'].reshape(-1)*valid
  np.testing.assert_array_equal(expected,data['rewards'][t])
 np.testing.assert_array_equal(s['observation'].reshape(-1,64),data['observation'][-1])
 print(json.dumps(dict(trace=a.trace,games=games,steps=len(data['actions']),observation_reward_alignment='exact',policy_sums='pass')))
finally:pool.close()
