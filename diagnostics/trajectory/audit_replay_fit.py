"""Read-only fixed replay-sample loss decomposition; not training metrics."""
import argparse,json,sys,hashlib
from pathlib import Path
import numpy as np
import torch

def audit(model,tensors,support_targets):
 x,a,r,v,p,valid,vvalid,rvalid=tensors
 state,policy,value=model.initial(x);rows=[];total=torch.zeros(len(x));steps=a.shape[1]
 for t in range(steps+1):
  logp=policy.log_softmax(-1);target_entropy=-(p[:,t]*p[:,t].clamp_min(1e-30).log()).sum(-1)
  ploss=-(p[:,t]*logp).sum(-1);vloss=-(support_targets(v[:,t],model.config.support)*value.log_softmax(-1)).sum(-1)
  mask=valid[:,t].bool();vmask=vvalid[:,t].bool()
  row={'step':t,'policy_samples':int(mask.sum()),'value_samples':int(vmask.sum())}
  if mask.any():row.update(policy_ce=float(ploss[mask].mean()),target_entropy=float(target_entropy[mask].mean()),policy_kl=float((ploss-target_entropy)[mask].mean()),predicted_entropy=float((-(logp.exp()*logp).sum(-1))[mask].mean()),target_support=float((p[:,t]>0).sum(-1)[mask].float().mean()))
  if vmask.any():row['value_ce']=float(vloss[vmask].mean())
  term=vloss*vvalid[:,t]+ploss*valid[:,t]
  if t:
   rloss=-(support_targets(r[:,t-1],model.config.support)*reward.log_softmax(-1)).sum(-1);rmask=rvalid[:,t-1].bool();row['reward_samples']=int(rmask.sum())
   if rmask.any():row['reward_ce']=float(rloss[rmask].mean())
   term+=rloss*rvalid[:,t-1]
  total+=term/(steps+1);rows.append(row)
  if t<steps:state,reward,policy,value=model.recurrent(state,a[:,t])
 expected=model.loss(*tensors);difference=float((expected-total).abs().max());assert torch.allclose(expected,total,atol=1e-6,rtol=1e-6)
 return {'loss_reconstruction_max_error':difference,'mean_recomputed_loss':float(total.mean()),'steps':rows}

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--state',type=Path,required=True);p.add_argument('--catalog',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--batch',type=int,default=128);a=p.parse_args()
 sys.path.insert(0,str(a.source));from model import MuZero,Config,support_targets
 from arena.train_selfplay import replay_batch
 torch.set_num_threads(2);state=torch.load(a.state,map_location='cpu',weights_only=True)
 for name in ['model.py','arena/train_selfplay.py','arena/history.py']:
  assert hashlib.sha256((a.source/name).read_bytes()).hexdigest()==state['source_files'][name]
 entry=next(e for e in json.loads(a.catalog.read_text())['models'] if e['id']=='M2048');model=MuZero(Config(**entry['config']));model.load_state_dict(state['model'],strict=True);model.eval();rng=np.random.default_rng(170001);reports=[]
 with torch.no_grad():
  for i,b in enumerate(state['replay']):
   block={k:v.numpy() for k,v in b.items()};tensors=replay_batch(block,rng,a.batch,state['arguments']['context'],state['arguments']['unroll'],'cpu');reports.append({'block':i,**audit(model,tensors,support_targets)})
 report={'state_games':state['totals']['games'],'sample_seed':170001,'examples_per_block':a.batch,'sampled_with_replacement':True,'new_games':0,'optimizer_updates':0,'scope':'Recomputed frozen diagnostics on saved replay, not original logged loss components; no strength claim','blocks':reports}
 a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
