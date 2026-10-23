"""Frozen equal-expansion target estimator diagnostic, not arena strength evaluation."""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch


def aggregate(policies, groups):
    if policies.ndim != 2 or groups < 1 or len(policies) % groups:
        raise ValueError('invalid grouping')
    if not torch.isfinite(policies).all() or (policies < 0).any():
        raise ValueError('invalid probability')
    if not torch.allclose(policies.sum(-1), torch.ones(len(policies)), atol=1e-6):
        raise ValueError('unnormalized policy')
    return policies.reshape(-1, groups, policies.shape[-1]).mean(1)


def metrics(estimates, reference):
    return dict(mean_squared_distance=float(((estimates-reference)**2).sum(-1).mean()),
                entropy=float(-(estimates*estimates.clamp_min(1e-30).log()).sum(-1).mean()),
                support=float((estimates > 0).sum(-1).float().mean()))


def main():
    p=argparse.ArgumentParser()
    for name in ('source','state','catalog','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.source))
    from model import MuZero, Config
    from arena.train_selfplay import replay_batch
    from batched_search_exploration import search_batch
    torch.set_num_threads(2)
    state=torch.load(str(a.state),map_location='cpu',weights_only=True,mmap=True)
    names=['model.py','arena/train_selfplay.py','arena/history.py','batched_search_exploration.py']
    bindings={n:hashlib.sha256((a.source/n).read_bytes()).hexdigest() for n in names}
    assert all(v==state['source_files'][n] for n,v in bindings.items())
    e=next(e for e in json.loads(a.catalog.read_text())['models'] if e['id']=='M2048')
    models={}
    for simulations in (8,32):
        c=dict(e['config']);c['simulations']=simulations
        m=MuZero(Config(**c));m.load_state_dict(state['model']);m.eval();models[simulations]=m
    rng=np.random.default_rng(180001);histories=[]
    for block in state['replay']:
        b={k:v.numpy() for k,v in block.items()}
        histories.append(replay_batch(b,rng,4,state['arguments']['context'],state['arguments']['unroll'],'cpu')[0])
    del state,b
    def draws(obs,count,simulations,roots,seed):
        torch.manual_seed(seed);m=models[simulations];x=obs.repeat(count,1)
        start=time.perf_counter()
        _,logits,_=m.initial(x);prior=logits.softmax(-1)
        noise=torch.distributions.Dirichlet(torch.full_like(prior,m.config.dirichlet_alpha)).sample()
        candidates=torch.multinomial((1-m.config.exploration_fraction)*prior+m.config.exploration_fraction*noise,roots,replacement=False)
        policies,_=search_batch(m,x,root_noise=noise,root_candidates=candidates)
        return policies,time.perf_counter()-start
    results=[]
    with torch.no_grad():
        for block,x in enumerate(histories):
            for i in range(len(x)):
                index=block*4+i;seed=180002+index*3;obs=x[i:i+1]
                reference,ref_time=draws(obs,32,32,8,seed)
                standard,std_time=draws(obs,8,32,8,seed+1)
                split,split_time=draws(obs,32,8,2,seed+2)
                standard=aggregate(standard,1);split=aggregate(split,4);mean=reference.mean(0)
                row=dict(block=block,history=i,seed=seed,reference_seconds=ref_time,
                         standard=dict(seconds=std_time,expansions=256,**metrics(standard,mean)),
                         split=dict(seconds=split_time,expansions=256,**metrics(split,mean)))
                row['mse_delta']=row['split']['mean_squared_distance']-row['standard']['mean_squared_distance']
                results.append(row)
                a.output.write_text(json.dumps(dict(complete=False,results=results),indent=2)+'\n')
    report=dict(complete=True,new_games=0,optimizer_updates=0,history_seed=180001,source_bindings=bindings,
                settings=dict(standard=[1,32,8],split=[4,8,2],reference_draws=32,estimates=8),
                scope='Development teacher-agreement proxy on saved histories. Changed tree depth/root settings; equal expansions do not imply equal time or strength.',results=results)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
