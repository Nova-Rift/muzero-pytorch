"""Read-only gradient decomposition on source-bound retained replay; no updates."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch

def main():
 p=argparse.ArgumentParser()
 for n in ('source','state','catalog','output'):p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--seed',type=int,default=210001);p.add_argument('--batch',type=int,default=64)
 a=p.parse_args();assert 1<=a.batch<=256
 sys.path.insert(0,str(a.source));from model import MuZero,Config,support_targets,scale_gradient
 from arena.train_selfplay import replay_batch
 torch.set_num_threads(2);s=torch.load(a.state,map_location='cpu',weights_only=True)
 for name in ('model.py','arena/train_selfplay.py'):
  assert hashlib.sha256((a.source/name).read_bytes()).hexdigest()==s['source_files'][name]
 e=next(x for x in json.loads(a.catalog.read_text())['models'] if x['id']=='M2048')
 m=MuZero(Config(**e['config']));m.load_state_dict(s['model']);m.eval()
 params=list(m.parameters());names=[n for n,_ in m.named_parameters()]
 def vector(loss):
  grads=torch.autograd.grad(loss,params,allow_unused=True,retain_graph=True)
  return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(params,grads)])
 masks={group:torch.cat([torch.full((p.numel(),),n.startswith(group),dtype=torch.bool) for n,p in zip(names,params)]) for group in ('representation','dynamics','policy','value','reward')}
 rng=np.random.default_rng(a.seed);rows=[]
 for index,raw in enumerate(s['replay']):
  block={k:v.numpy() for k,v in raw.items()}
  batch=replay_batch(block,rng,a.batch,s['arguments']['context'],s['arguments']['unroll'],'cpu')
  obs,actions,rewards,values,policies,valid,value_valid,reward_valid=batch;steps=actions.shape[1]
  state,policy,value=m.initial(obs);losses={k:obs.new_zeros(a.batch) for k in ('policy','value','reward')}
  for t in range(steps+1):
   losses['policy']+=-(policies[:,t]*policy.log_softmax(-1)).sum(-1)*valid[:,t]/(steps+1)
   losses['value']+=-(support_targets(values[:,t],m.config.support)*value.log_softmax(-1)).sum(-1)*value_valid[:,t]/(steps+1)
   if t:losses['reward']+=-(support_targets(rewards[:,t-1],m.config.support)*reward.log_softmax(-1)).sum(-1)*reward_valid[:,t-1]/(steps+1)
   if t<steps:state,reward,policy,value=m.recurrent(scale_gradient(state,.5),actions[:,t])
  original=m.loss(*batch);total=sum(losses.values());assert torch.allclose(total,original,atol=2e-6,rtol=2e-6)
  grads={k:vector(v.mean()) for k,v in losses.items()};expected=vector(original.mean());summed=sum(grads.values())
  assert torch.allclose(summed,expected,atol=2e-6,rtol=2e-4)
  groups={}
  for name,mask in masks.items():
   vectors={k:v[mask] for k,v in grads.items()};norms={k:float(v.norm()) for k,v in vectors.items()};cos={}
   for x,y in [('policy','value'),('policy','reward'),('value','reward')]:
    den=norms[x]*norms[y];cos[x+'_'+y]=float(torch.dot(vectors[x],vectors[y])/den) if den>1e-20 else None
   groups[name]=dict(norms=norms,cosines=cos)
  rows.append(dict(block=index,samples=a.batch,losses={k:float(v.mean().detach()) for k,v in losses.items()},loss_error=float((total-original).abs().max().detach()),gradient_max_error=float((summed-expected).abs().max()),reward_positive=int(((rewards>0)&(reward_valid>0)).sum()),reward_valid=int(reward_valid.sum()),value_positive=int(((values>0)&(value_valid>0)).sum()),value_valid=int(value_valid.sum()),groups=groups))
 for name,v in m.state_dict().items():assert torch.equal(v,s['model'][name]),name
 report=dict(schema=1,seed=a.seed,sampled_with_replacement=True,checkpoint_games=int(s.get('games',2048)),optimizer_updates=0,new_games=0,weights_unchanged=True,source_files=s['source_files'],blocks=rows,limitations='Raw batch gradients before Adam, clipping and weight decay; retained replay and one checkpoint only. Magnitude/alignment is not strength or causal evidence.')
 a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(blocks=len(rows),samples=len(rows)*a.batch,weights_unchanged=True)))
if __name__=='__main__':main()
