"""Frozen-input linear diagnostic; probe scores are never used by an agent."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model import MuZero,Config
from arena.train_selfplay import history

def score(pred,y):
 pos=pred[y>0];neg=pred[y==0]
 comparison=pos[:,None]-neg[None,:]
 return dict(count=len(y),positive=len(pos),auc=float(((comparison>0)+.5*(comparison==0)).mean()),positive_mean=float(pos.mean()),negative_mean=float(neg.mean()),balanced_mse=float(.5*((pos-1)**2).mean()+.5*(neg**2).mean()))

def main():
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
 if a.output.exists():raise ValueError('Refusing to overwrite evidence')
 torch.set_num_threads(1);meta=json.loads((a.run/'manifest.json').read_text())
 model=MuZero(Config(**meta['config'])).eval();checkpoint=a.run/'checkpoint-1024.npz'
 with np.load(checkpoint,allow_pickle=False) as z:model.load_state_dict({k:torch.from_numpy(z[k]) for k in z.files})
 rows=[[],[]];files=sorted(a.run.glob('trace-*.npz'),key=lambda f:int(f.stem.split('-')[1]))
 for i,file in enumerate(files):
  with np.load(file,allow_pickle=False) as z:b={k:z[k] for k in z.files}
  indices=np.argwhere(b['valid'][:-1])
  for start in range(0,len(indices),128):
   t,ids=indices[start:start+128].T
   with torch.no_grad():
    x=torch.from_numpy(history(b['observation'],b['actions'],t,ids,64))
    onehot=F.one_hot(torch.from_numpy(b['actions'][t,ids]),153).float()
    h,_,_=model.initial(x);raw=model.dynamics(torch.cat((h,onehot),-1))
   current=np.concatenate((b['observation'][t,ids],onehot.numpy()),-1)
   recent=np.concatenate((x.numpy()[:,4096-4*64:4096],onehot.numpy()),-1)
   rows[i>=12].append((current,recent,raw.numpy(),b['rewards'][t,ids]))
 data=[tuple(np.concatenate([r[k] for r in rs]) for k in range(4)) for rs in rows]
 report=dict(scope='already observed training-derived diagnostic data; no policy score',checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),fitting=[f.name for f in files[:12]],diagnostic=[f.name for f in files[12:]],ridge=1,arms={})
 y=data[0][3].astype(np.float64);weights=np.where(y>0,len(y)/(2*(y>0).sum()),len(y)/(2*(y==0).sum()))
 for j,name in enumerate(('current_and_action','last4_and_action','frozen_recurrent')):
  x=data[0][j].astype(np.float64);mean=x.mean(0);std=x.std(0);std[std<1e-8]=1
  x=np.column_stack(((x-mean)/std,np.ones(len(x))));penalty=np.eye(x.shape[1]);penalty[-1,-1]=0
  lhs=torch.from_numpy(x.T@(weights[:,None]*x)+penalty);rhs=torch.from_numpy(x.T@(weights*y))
  coefficients=torch.linalg.solve(lhs,rhs).numpy()
  predictions=[np.column_stack(((d[j]-mean)/std,np.ones(len(d[j]))))@coefficients for d in data]
  report['arms'][name]=dict(dimensions=x.shape[1]-1,fitting=score(predictions[0],data[0][3]),diagnostic=score(predictions[1],data[1][3]))
 a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['arms'],indent=2))
if __name__=='__main__':main()
