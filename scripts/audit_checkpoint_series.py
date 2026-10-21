"""Read-only initial-state neural diagnostics; no optimizer steps."""
import sys,json
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model import MuZero,Config,scalar
from arena.environment import Pool
from arena.history import DeviceHistory
torch.set_num_threads(1)
run=Path(sys.argv[1]);meta=json.loads((run/'manifest.json').read_text())
p=Pool(sys.argv[2],agents=2,cadence=meta['arguments'].get('cadence','legacy'))
state=p.reset([41000300]);p.close()
x=DeviceHistory(11,64,'cpu').observe(state['observation'].reshape(11,64))[:2]
report={}
for name in ['initial.npz','checkpoint-512.npz','checkpoint-1024.npz','checkpoint-2048.npz']:
 m=MuZero(Config(**meta['config'])).eval()
 with np.load(run/name,allow_pickle=False) as z:m.load_state_dict({k:torch.from_numpy(z[k]) for k in z.files})
 with torch.no_grad():
  h,logits,v=m.initial(x);p=logits.softmax(-1);actions=p.argmax(-1)
  _,r,_,nv=m.recurrent(h,actions)
  report[name]=dict(policy_entropy=(-(p*p.clamp_min(1e-30).log()).sum(-1)).tolist(),max_probability=p.max(-1).values.tolist(),top_actions=actions.tolist(),root_value=scalar(v).tolist(),top_action_predicted_reward=scalar(r).tolist(),top_action_one_step_q=(scalar(r)+m.config.discount*scalar(nv)).tolist())
Path(sys.argv[3]).write_text(json.dumps(report,indent=2))
print(json.dumps(report['checkpoint-2048.npz']))
