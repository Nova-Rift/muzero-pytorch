"""Fixed offline clock calibration on retained replay; no model updates or new games."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch

def features(value,remaining,clock):
 z=remaining if clock else np.zeros_like(remaining)
 return np.stack((np.ones_like(value),value,z,z*value),axis=1)
def fit(x,y):
 penalty=np.eye(x.shape[1])*.001;penalty[0,0]=0
 return np.linalg.solve(x.T@x+penalty,x.T@y)
def score(y,p):
 raw=p;p=p.clip(0,1)
 return dict(brier=float(np.mean((p-y)**2)),positive_brier=float(np.mean((p[y==1]-1)**2)) if (y==1).any() else None,negative_brier=float(np.mean(p[y==0]**2)) if (y==0).any() else None,positives=int((y==1).sum()),negatives=int((y==0).sum()),clipped=int(((raw<0)|(raw>1)).sum()))
def main():
 p=argparse.ArgumentParser()
 for n in ('source','state','catalog','output'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();sys.path.insert(0,str(a.source));from model import MuZero,Config,scalar
 from arena.train_selfplay import history
 torch.set_num_threads(2);s=torch.load(a.state,map_location='cpu',weights_only=True)
 for n in ('model.py','arena/train_selfplay.py'):assert hashlib.sha256((a.source/n).read_bytes()).hexdigest()==s['source_files'][n]
 args=s['arguments'];assert (args['return_mode'],args['timeout_bootstrap'],args['discount'],args['limit'])==('episode','zero',1.0,256)
 assert len(s['replay'])==3
 e=next(e for e in json.loads(a.catalog.read_text())['models'] if e['id']=='M2048');m=MuZero(Config(**e['config']));m.load_state_dict(s['model']);m.eval();rng=np.random.default_rng(210201);blocks=[]
 for raw in s['replay']:
  b={k:v.numpy() for k,v in raw.items()};picks=b['indices'][rng.integers(len(b['indices']),size=256)];times,agents=picks.T
  x=history(b['observation'],b['actions'],times,agents,args['context']);y=b['targets'][times,agents].astype(np.float64);assert np.isin(y,[0,1]).all()
  with torch.no_grad():v=scalar(m.initial(torch.from_numpy(x))[2]).numpy().astype(np.float64)
  blocks.append((v,(256-times)/256,y,len(np.unique(picks,axis=0))))
 train=[np.concatenate([blocks[i][j] for i in (0,1)]) for j in range(3)];v,t,y=blocks[2][:3];results={'frozen':score(y,v)};coeff={}
 for name,clock in [('constant',False),('clock',True)]:
  w=fit(features(train[0],train[1],clock),train[2]);assert np.isfinite(w).all();coeff[name]=w.tolist();results[name]=score(y,features(v,t,clock)@w)
 c,b=results['clock'],results['constant'];gate=c['positives']>=10 and c['negatives']>=10 and c['brier']<=b['brier']-.005 and c['positive_brier']<=b['positive_brier']+.01 and c['negative_brier']<=b['negative_brier']+.01
 for name,vv in m.state_dict().items():assert torch.equal(vv,s['model'][name])
 out=dict(schema=1,seed=210201,train_blocks=[0,1],validation_blocks=[2],samples_per_block=256,unique_samples_per_block=[b[3] for b in blocks],ridge=.001,coefficients=coeff,validation=results,gate_pass=bool(gate),model_optimizer_updates=0,new_games=0,weights_unchanged=True,source_files=s['source_files'],limitations='Offline fitted diagnostic only; correlated replay, one checkpoint and one validation block. Not an online training/strength result. No validation tuning.')
 a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(results));print('gate_pass',gate)
if __name__=='__main__':main()
