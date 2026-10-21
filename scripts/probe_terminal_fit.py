"""Bounded copied-network fit diagnostic; no policy deployment or new games.

Uniform versus 50% terminal-transition sampling intentionally changes the fitting
measure. This is a learnability probe, NOT an unbiased replay-policy comparison.
"""
import argparse,copy,hashlib,json,sys,time
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model import MuZero,Config,scalar
from arena.train_selfplay import history

def main():
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('output',type=Path);p.add_argument('--steps',type=int,default=128);a=p.parse_args()
 if a.output.exists():raise ValueError('Preserve evidence')
 torch.set_num_threads(1);meta=json.loads((a.run/'manifest.json').read_text());files=sorted(a.run.glob('trace-*.npz'),key=lambda p:int(p.stem.split('-')[1]));data=[[],[]]
 for i,f in enumerate(files):
  with np.load(f,allow_pickle=False) as z:b={k:z[k] for k in z.files}
  t,ids=np.argwhere(b['valid'][:-1]).T;term=b['absorbing'][t+1,ids];validnext=b['valid'][t+1,ids]
  data[i>=12].append((history(b['observation'],b['actions'],t,ids,64),b['actions'][t,ids,None],b['rewards'][t,ids,None],np.stack([b['targets'][t,ids],b['targets'][t+1,ids]],1),np.stack([b['policies'][t,ids],b['policies'][t+1,ids]],1),np.stack([np.ones(len(t)),validnext],1),np.stack([np.ones(len(t)),validnext|term],1),np.ones((len(t),1)),term))
 data=[tuple(torch.from_numpy(np.ascontiguousarray(np.concatenate([r[k] for r in rows]))) for k in range(9)) for rows in data]
 model=MuZero(Config(**meta['config'])).eval();path=a.run/'checkpoint-1024.npz'
 with np.load(path,allow_pickle=False) as z:model.load_state_dict({k:torch.from_numpy(z[k]) for k in z.files})
 @torch.no_grad()
 def evaluate(m,ds):
  values=[];rewards=[]
  for offset in range(0,len(ds[0]),128):
   h,_,_=m.initial(ds[0][offset:offset+128]);_,r,_,v=m.recurrent(h,ds[1][offset:offset+128,0]);values.append(scalar(v));rewards.append(scalar(r))
  v=torch.cat(values);r=torch.cat(rewards);term=ds[-1].bool();win=ds[2][:,0]>0
  return dict(rows=len(v),terminals=int(term.sum()),wins=int(win.sum()),terminal_value_mean=float(v[term].mean()),terminal_value_mse=float((v[term]**2).mean()),win_reward_mean=float(r[win].mean()),win_reward_mse=float(((r[win]-1)**2).mean()),other_reward_mean=float(r[~win].mean()),reward_mse=float(((r-ds[2][:,0])**2).mean()))
 report={'scope':__doc__,'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'trace_hashes':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files},'fit_files':[f.name for f in files[:12]],'diagnostic_files':[f.name for f in files[12:]],'steps':a.steps,'batch':32,'seed':0,'learning_rate':3e-4,'unroll':1,'arms':{},'initial':[evaluate(model,d) for d in data]}
 terminal=np.flatnonzero(data[0][-1].numpy());other=np.flatnonzero(~data[0][-1].numpy())
 for arm in ['uniform','half_terminal','half_terminal_corrected']:
  m=copy.deepcopy(model);optimizer=torch.optim.Adam(m.parameters(),lr=3e-4,weight_decay=1e-4);rng=np.random.default_rng(0);start=time.monotonic()
  for step in range(a.steps):
   if time.monotonic()-start>90:raise RuntimeError('CPU probe exceeded per-arm bound')
   ids=rng.integers(len(data[0][0]),size=32) if arm=='uniform' else np.concatenate([rng.choice(terminal,16),rng.choice(other,16)])
   batch=[t[ids] for t in data[0][:-1]];optimizer.zero_grad();losses=m.loss(*batch)
   if arm=='half_terminal_corrected':
    fraction=len(terminal)/len(data[0][0]);weights=torch.where(data[0][-1][ids],2*fraction,2*(1-fraction));losses=losses*weights
   loss=losses.mean();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),10.);optimizer.step()
  report['arms'][arm]={'seconds':time.monotonic()-start,'evaluation':[evaluate(m,d) for d in data]}
 assert hashlib.sha256(path.read_bytes()).hexdigest()==report['checkpoint_sha256']
 a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'initial':report['initial'],'arms':report['arms']},indent=2))
if __name__=='__main__':main()
