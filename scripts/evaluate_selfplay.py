"""Frozen checkpoint tournament, no training; both sides are learned policies."""
import sys,json,argparse,time
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model import MuZero,Config
from batched_search import search_batch
from arena.environment import Pool
from arena.history import DeviceHistory
p=argparse.ArgumentParser()
p.add_argument('--site',required=True);p.add_argument('--run',type=Path,required=True)
p.add_argument('--challenger',required=True);p.add_argument('--reference',default='initial.npz')
p.add_argument('--games',type=int,default=32);p.add_argument('--seed',type=int,default=41000000)
p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu')
a=p.parse_args();torch.set_num_threads(1);torch.manual_seed(a.seed)
if a.output.exists():raise ValueError('Preserve existing evaluation')
meta=json.loads((a.run/'manifest.json').read_text());c=Config(**meta['config'])
models=[]
for file in [a.challenger,a.reference]:
 m=MuZero(c).to(a.device).eval()
 with np.load(a.run/file,allow_pickle=False) as z:m.load_state_dict({k:torch.from_numpy(z[k]) for k in z.files})
 models.append(m)
pool=Pool(a.site,workers=1,agents=2,cadence=meta['arguments'].get('cadence','legacy'));s=pool.reset(range(a.seed,a.seed+a.games))
history=DeviceHistory(a.games*11,64,a.device)
seats=np.arange(a.games)%2;finished=np.zeros(a.games,bool);steps=np.zeros(a.games,int)
actions=np.zeros((a.games,11),np.int64);started=time.perf_counter()
indices=[np.array([g*11+(seats[g] if j==0 else 1-seats[g]) for g in range(a.games)]) for j in range(2)]
try:
 with torch.no_grad():
  for t in range(256):
   x=history.observe(s['observation'].reshape(-1,64))
   for m,ids in zip(models,indices):
    pol,_=search_batch(m,x[torch.as_tensor(ids,device=a.device)])
    actions.reshape(-1)[ids]=torch.multinomial(pol,1).squeeze(-1).cpu().numpy()
   history.acted(actions.reshape(-1));commands=actions.copy();commands[finished]=-1
   steps+=~finished;s=pool.step(commands);finished|=s['finished']
   if finished.all():break
 win=finished&(s['winner']==seats);loss=finished&(s['winner']==1-seats)
 result=dict(challenger=a.challenger,reference=a.reference,seed=a.seed,games=a.games,wins=int(win.sum()),losses=int(loss.sum()),cutoffs=int((~finished).sum()),draws=int((finished&~win&~loss).sum()),mean_steps=float(steps.mean()),seconds=time.perf_counter()-started,seat_swapped=True,records=[dict(seat=int(seats[i]),winner=int(s['winner'][i]),finished=bool(finished[i]),steps=int(steps[i])) for i in range(a.games)])
 a.output.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='records'}))
finally:pool.close()
