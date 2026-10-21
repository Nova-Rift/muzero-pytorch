"""Bounded frozen-feature reward-head probe; no policy training or new games."""
import argparse,copy,hashlib,json,sys
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model import MuZero,Config,scalar,support_targets
from arena.train_selfplay import history

def main():
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
 if a.output.exists():raise ValueError('Preserve original output')
 torch.set_num_threads(1);meta=json.loads((a.run/'manifest.json').read_text())
 model=MuZero(Config(**meta['config'])).eval()
 checkpoint=a.run/'checkpoint-1024.npz'
 with np.load(checkpoint,allow_pickle=False) as z:model.load_state_dict({k:torch.from_numpy(z[k]) for k in z.files})
 datasets=[[],[]];files=sorted(a.run.glob('trace-*.npz'),key=lambda f:int(f.stem.split('-')[1]))
 for i,file in enumerate(files):
  with np.load(file,allow_pickle=False) as z:b={k:z[k] for k in z.files}
  indices=np.argwhere(b['valid'][:-1])
  for offset in range(0,len(indices),128):
   t,ids=indices[offset:offset+128].T
   with torch.no_grad():
    x=torch.from_numpy(history(b['observation'],b['actions'],t,ids,64))
    h,_,_=model.initial(x)
    raw=model.dynamics(torch.cat((h,F.one_hot(torch.from_numpy(b['actions'][t,ids]),153).float()),-1))
   datasets[i>=12].append((raw,torch.from_numpy(b['rewards'][t,ids])))
 data=[tuple(torch.cat([row[k] for row in ds]) for k in (0,1)) for ds in datasets]
 def score(head):
  results=[]
  with torch.no_grad():
   for x,y in data:
    pred=scalar(head(x));positive=y>0
    results.append(dict(count=len(y),wins=int(positive.sum()),mse=float(((pred-y)**2).mean()),positive_mean=float(pred[positive].mean()),zero_mean=float(pred[~positive].mean()),positive_mse=float(((pred[positive]-y[positive])**2).mean()),zero_mse=float((pred[~positive]**2).mean())))
  return dict(fitting=results[0],diagnostic=results[1])
 report=dict(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),scope='frozen features; already observed training-derived traces; not policy evaluation',split=dict(fitting=[f.name for f in files[:12]],diagnostic=[f.name for f in files[12:]]),before=score(model.reward),runs=[])
 x,y=data[0];n=len(y);uniform=torch.full((n,),1/n);positive=y>0
 assert positive.any()
 for seed in (123,456,789):
  for arm in ('uniform','importance_corrected_positive_mix'):
   rng=torch.Generator().manual_seed(seed);head=copy.deepcopy(model.reward)
   optimizer=torch.optim.Adam(head.parameters(),lr=3e-4,weight_decay=1e-4)
   prob=uniform if arm=='uniform' else .5*uniform+.5*positive/positive.sum()
   seen=0
   for step in range(512):
    ids=torch.multinomial(prob,32,replacement=True,generator=rng);seen+=int(positive[ids].sum())
    loss=-(support_targets(y[ids],model.config.support)*head(x[ids]).log_softmax(-1)).sum(-1)
    loss=(loss/(n*prob[ids])).mean()
    optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
   report['runs'].append(dict(seed=seed,arm=arm,steps=512,batch=32,positive_samples=seen,after=score(head)))
 a.output.write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
