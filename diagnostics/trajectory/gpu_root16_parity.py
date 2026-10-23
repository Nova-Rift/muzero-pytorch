import sys,json,hashlib,time
from pathlib import Path
import numpy as np,torch
E=Path('/workspace/experiment');W=Path('/workspace/runs/loop-a/parity');W.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,'/workspace/experiment/model')
from tournament import run
from trajectory_recorder import RecordingPool
from arena.environment import Pool
class Spy(Pool):
 def __init__(self,*a,**kw):super().__init__(*a,**kw);self.sent=[];self.states=[]
 def step(self,actions):
  self.sent.append(actions.copy());s=super().step(actions);self.states.append({k:v.copy() for k,v in s.items()});return s
torch.set_num_threads(2)
entries=[e for e in json.loads((E/'M2048-root16-E1024.json').read_text())['models'] if e['id'] in ['M2048-root16','E1024']]
for e in entries:
 e['checkpoint']=str(E/'weights'/Path(e['checkpoint']).name);assert hashlib.sha256(Path(e['checkpoint']).read_bytes()).hexdigest()==e['sha256']
models=[run.load_model(e,str(E/'site'),'cuda') for e in entries]
def digest():
 h=hashlib.sha256()
 for m in models:
  for v in m.state_dict().values():h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
before=digest();spies=[]
def factory(*a,**kw):
 p=Spy(*a,**kw);spies.append(p)
 return p if len(spies)==1 else RecordingPool(p,W/'policy-engineering-trace.jsonl',2,256)
run.Pool=factory;start=time.monotonic()
x=run.play(str(E/'site'),entries,models,[91600000,91600001],0,1,256)
y=run.play(str(E/'site'),entries,models,[91600000,91600001],0,1,256)
assert x['records']==y['records'];assert len(spies[0].sent)==len(spies[1].sent)
for a,b in zip(spies[0].sent,spies[1].sent):assert np.array_equal(a,b)
for a,b in zip(spies[0].states,spies[1].states):
 for k in a:assert np.array_equal(a[k],b[k])
assert before==digest()
report=dict(exact_actions=True,exact_states=True,exact_outcomes=True,model_bytes_unchanged=True,decision_limit=256,actual_decisions=len(spies[0].sent),streams=2,seconds=time.monotonic()-start,torch=torch.__version__,device='cuda',checkpoint_hashes={e['id']:e['sha256'] for e in entries},scope='Full-horizon two-stream GPU parity check; duplicate probe games excluded from rankings.')
(W/'policy-parity.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
