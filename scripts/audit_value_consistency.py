"""Read-only value-equivalence diagnostic on preserved self-play transitions."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model import MuZero,Config,scalar
from arena.train_selfplay import history

def stats(pred,target):
 if not len(pred):return {'count':0}
 return dict(count=len(pred),predicted_mean=float(pred.mean()),target_mean=float(target.mean()),mse=float(((pred-target)**2).mean()),target_variance=float(target.var()),prediction_std=float(pred.std()))
def main():
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
 if a.output.exists():raise ValueError('Do not overwrite evidence')
 torch.set_num_threads(1);meta=json.loads((a.run/'manifest.json').read_text());files=sorted(a.run.glob('trace-*.npz'));result={'scope':'saved training-derived traces, not an unbiased policy evaluation; stochastic opponent and latent-state aliasing limit one-step equivalence','trace_hashes':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files},'checkpoints':{}}
 for name in ['initial.npz','checkpoint-512.npz','checkpoint-1024.npz','checkpoint-2048.npz']:
  m=MuZero(Config(**meta['config'])).eval()
  with np.load(a.run/name,allow_pickle=False) as z:m.load_state_dict({k:torch.from_numpy(z[k]) for k in z.files})
  rows=[]
  for file in files:
   with np.load(file,allow_pickle=False) as z:b={k:z[k] for k in z.files}
   indices=np.argwhere(b['valid'][:-1]);length=len(b['actions'])
   for start in range(0,len(indices),128):
    t,ids=indices[start:start+128].T
    with torch.no_grad():
     x=torch.from_numpy(history(b['observation'],b['actions'],t,ids,64));nx=torch.from_numpy(history(b['observation'],b['actions'],t+1,ids,64))
     state,_,v=m.initial(x);_,r,_,nv=m.recurrent(state,torch.from_numpy(b['actions'][t,ids]));_,_,actualv=m.initial(nx)
    terminal=b['absorbing'][t+1,ids];surviving=b['valid'][t+1,ids]
    rows.append(np.stack([scalar(v).numpy(),scalar(r).numpy(),scalar(nv).numpy(),scalar(actualv).numpy(),b['rewards'][t,ids],terminal,surviving],-1))
  data=np.concatenate(rows);terminal=data[:,5].astype(bool);surviving=data[:,6].astype(bool)
  imagined=data[:,1]+m.config.discount*data[:,2]
  observed=data[:,4]+m.config.discount*data[:,3]*(~terminal)
  result['checkpoints'][name]={'sha256':hashlib.sha256((a.run/name).read_bytes()).hexdigest(),'surviving_next_value':stats(data[surviving,2],data[surviving,3]),'terminal_next_value':stats(data[terminal,2],np.zeros(terminal.sum())),'one_step_imagined_vs_observed_backup':stats(imagined,observed),'root_vs_observed_backup':stats(data[:,0],observed),'terminal_reward':stats(data[terminal,1],data[terminal,4]),'cutoff_transitions':int((~terminal&~surviving).sum())}
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['checkpoints'],indent=2))
if __name__=='__main__':main()
