"""Frozen repeated-search target decomposition; no environment games or training."""
import argparse,json,sys,hashlib
from pathlib import Path
import numpy as np,torch

def decompose(targets,prior):
 mean=targets.mean(0);h=-(targets*targets.clamp_min(1e-30).log()).sum(-1).mean();hm=-(mean*mean.clamp_min(1e-30).log()).sum()
 expected=(targets*(targets.clamp_min(1e-30).log()-prior.clamp_min(1e-30).log())).sum(-1).mean();kl=(mean*(mean.clamp_min(1e-30).log()-prior.clamp_min(1e-30).log())).sum();js=hm-h
 assert torch.allclose(expected,js+kl,atol=2e-6,rtol=2e-6)
 return dict(mean_target_entropy=float(h),averaged_target_entropy=float(hm),mean_sample_kl_to_prior=float(expected),sampling_js=float(js),averaged_target_kl_to_prior=float(kl),decomposition_error=float(abs(expected-js-kl)))

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--state',type=Path,required=True);p.add_argument('--catalog',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--noise-fraction',type=float,default=None);p.add_argument('--states',type=int,default=4);p.add_argument('--draws',type=int,default=32);a=p.parse_args();assert 1<=a.states<=64 and 2<=a.draws<=256;sys.path.insert(0,str(a.source))
 from model import MuZero,Config
 from arena.train_selfplay import replay_batch
 from batched_search_exploration import search_batch
 torch.set_num_threads(2);s=torch.load(a.state,map_location='cpu',weights_only=True);e=next(e for e in json.loads(a.catalog.read_text())['models'] if e['id']=='M2048');config=dict(e['config']);config.update({} if a.noise_fraction is None else {'exploration_fraction':a.noise_fraction});assert 0<=config['exploration_fraction']<=1;m=MuZero(Config(**config));m.load_state_dict(s['model']);m.eval()
 for name in ['model.py','arena/train_selfplay.py','arena/history.py','batched_search_exploration.py']:
  assert hashlib.sha256((a.source/name).read_bytes()).hexdigest()==s['source_files'][name]
 b={k:v.numpy() for k,v in s['replay'][-1].items()};x=replay_batch(b,np.random.default_rng(170002),a.states,s['arguments']['context'],s['arguments']['unroll'],'cpu')[0];del s,b
 results=[];torch.manual_seed(170003)
 with torch.no_grad():
  for i in range(a.states):
   obs=x[i:i+1].repeat(a.draws,1);_,logits,_=m.initial(obs);prior=logits.softmax(-1)
   noise=torch.distributions.Dirichlet(torch.full_like(prior,m.config.dirichlet_alpha)).sample();roots=torch.multinomial((1-m.config.exploration_fraction)*prior+m.config.exploration_fraction*noise,8,replacement=False)
   policies,_=search_batch(m,obs,root_noise=noise,root_candidates=roots)
   assert torch.allclose(policies.sum(-1),torch.ones(a.draws));results.append(dict(state_index=i,draws=a.draws,**decompose(policies,prior[0])))
 report=dict(exploration_fraction=m.config.exploration_fraction,changed_search_setting=a.noise_fraction is not None,seed=170003,states=a.states,draws_per_state=a.draws,new_games=0,optimizer_updates=0,scope='Finite Monte Carlo repeated noisy search on fixed saved histories; not generalization or irreducible-noise proof',results=results)
 a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
