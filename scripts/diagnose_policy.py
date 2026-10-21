"""Read-only initial-state audit; no training or environment rule changes."""
import sys,json
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model import MuZero,Config,scalar
from batched_search import search_batch
from arena.environment import Pool
from arena.history import DeviceHistory
torch.set_num_threads(1)
run=Path(sys.argv[1]);meta=json.loads((run/'manifest.json').read_text());out={}
pool=Pool(sys.argv[2]);state=pool.reset([41000000]);pool.close()
x=DeviceHistory(11,64,'cpu').observe(state['observation'].reshape(11,64))
for label in ['initial','trained']:
 torch.manual_seed(meta['arguments']['seed']);m=MuZero(Config(**meta['config'])).eval()
 if label=='trained':
  with np.load(run/'weights.npz',allow_pickle=False) as z:m.load_state_dict({k:torch.from_numpy(z[k]) for k in z.files})
 with torch.no_grad():
  h,p,v=m.initial(x);p=p.softmax(-1);vis,sv=search_batch(m,x,explore=True)
  out[label]={'policy_entropy':float(-(p*p.clamp_min(1e-30).log()).sum(-1).mean()),'max_policy_probability':p.max(-1).values.tolist(),'top_actions':p.argmax(-1).tolist(),'search_nonzero_actions':(vis>0).sum(-1).tolist(),'search_top_actions':vis.argmax(-1).tolist(),'value':scalar(v).tolist(),'search_value':sv.tolist()}
print(json.dumps(out,indent=2))
