"""Train independent MuZero on the exact website arena, with explicit work counts."""
import argparse
from collections import deque
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model import Config,MuZero
from arena.rootweighted import model_class
from batched_search_exploration import search_batch
from arena.environment import Pool
from arena.history import DeviceHistory
from arena.returns import episode_returns
from arena.provenance import capture_source
from arena.state_history import save_milestone


class Search:
    def __init__(self,model,count,graph=False,agents=2,root_candidates=0):
        self.indices=torch.tensor([g*11+a for g in range(count//11) for a in range(agents)],device=next(model.parameters()).device)
        self.candidates=torch.zeros((len(self.indices),root_candidates),dtype=torch.long,device=self.indices.device) if root_candidates else None
        self.total=count
        count=len(self.indices)
        self.model=model;self.graph=None;self.entropies=[]
        self.input=torch.zeros(count,model.config.observations,device=next(model.parameters()).device)
        self.noise=torch.full((count,model.config.actions),1/model.config.actions,device=self.input.device)
        if graph:
            if self.input.device.type!='cuda':raise ValueError('CUDA graph requires CUDA')
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):search_batch(model,self.input,root_noise=self.noise,static=True,root_candidates=self.candidates)
            torch.cuda.current_stream().wait_stream(stream)
            self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.result=search_batch(model,self.input,root_noise=self.noise,static=True,root_candidates=self.candidates)

    @torch.no_grad()
    def __call__(self,inputs):
        self.input.copy_(inputs[self.indices])
        self.noise.copy_(torch.distributions.Dirichlet(torch.full_like(self.noise,self.model.config.dirichlet_alpha)).sample())
        if self.candidates is not None:
            _,logits,_=self.model.initial(self.input)
            prior=(1-self.model.config.exploration_fraction)*logits.softmax(-1)+self.model.config.exploration_fraction*self.noise
            self.candidates.copy_(torch.multinomial(prior,self.candidates.shape[1],replacement=False))
        if self.graph:
            self.graph.replay();policies,values=self.result
        else:policies,values=search_batch(self.model,self.input,root_noise=self.noise,root_candidates=self.candidates)
        actions=torch.multinomial(policies,1).squeeze(-1)
        full_actions=np.zeros(self.total,dtype=np.int64);full_policies=np.zeros((self.total,self.model.config.actions),dtype=np.float32);full_values=np.zeros(self.total,dtype=np.float32)
        ids=self.indices.cpu().numpy();full_actions[ids]=actions.cpu().numpy();full_policies[ids]=policies.cpu().numpy();full_values[ids]=values.cpu().numpy()
        self.entropies.append(float(-(policies*policies.clamp_min(1e-30).log()).sum(-1).mean()))
        return full_actions,full_policies,full_values


def history(observations,actions,times,agents,context):
    # One observation history and previous-action history per selected agent.
    indices=times[:,None]-np.arange(context-1,-1,-1)[None]
    mask=indices>=0;safe=np.maximum(indices,0)
    obs=observations[safe,agents[:,None]]*mask[...,None]
    ai=times[:,None]-np.arange(context-1,0,-1)[None]
    previous=actions[np.maximum(ai,0),agents[:,None]]*(ai>=0)/153.
    return np.concatenate((obs.reshape(len(times),-1),previous),-1).astype(np.float32)


def collect(pool,model,seeds,limit,context,searcher,device_history=False,return_mode="nstep",timeout_bootstrap="continue"):
    state=pool.reset(seeds);games=len(seeds);count=games*11
    obs=np.zeros((limit+1,count,64),np.float32)
    actions=np.zeros((limit,count),np.int64)
    rewards=np.zeros((limit,count),np.float32)
    policies=np.zeros((limit+1,count,153),np.float32)
    values=np.zeros((limit+1,count),np.float32)
    valid=np.zeros((limit+1,count),bool)
    terminal=np.zeros((limit,count),bool)
    action_counts=np.zeros(153,np.int64);hits=0;movement=0.
    finished=np.zeros(games,bool);joint_steps=0;timings={'search':0.,'environment':0.,'history':0.}
    device=next(model.parameters()).device
    cache=DeviceHistory(count,context,device) if device_history else None
    for t in range(limit):
        obs[t]=state['observation'].reshape(count,64)
        valid[t]=(state['alive']&~finished[:,None]).reshape(-1)
        joint_steps+=int((~finished).sum())
        start=time.perf_counter()
        x=cache.observe(obs[t]) if cache else torch.from_numpy(history(obs,actions,np.full(count,t),np.arange(count),context)).to(device)
        timings['history']+=time.perf_counter()-start;start=time.perf_counter()
        act,pol,value=searcher(x)
        if cache:cache.acted(act)
        timings['search']+=time.perf_counter()-start
        action_counts+=np.bincount(act[valid[t]],minlength=153)
        actions[t],policies[t],values[t]=act,pol,value
        commands=act.reshape(games,11).copy();commands[finished]=-1
        start=time.perf_counter();following=pool.step(commands);timings['environment']+=time.perf_counter()-start
        rewards[t]=following['reward'].reshape(-1)*valid[t]
        hits+=float(np.maximum(0,state['observation'][:,:,2]-following['observation'][:,:,2]).sum()*3)
        movement+=float(np.linalg.norm((following['observation'][:,:,:2]-state['observation'][:,:,:2])*np.array([1000,460]),axis=-1)[valid[t].reshape(games,11)].sum())
        terminal[t]=(~following['alive']|following['finished'][:,None]).reshape(-1)
        finished|=following['finished'];state=following
        if finished.all():break
    length=t+1
    obs[length]=state['observation'].reshape(count,64)
    # Only time-limit survivors bootstrap; use actual final observations.
    if timeout_bootstrap not in ("continue","zero"):raise ValueError("Invalid timeout bootstrap mode")
    if not finished.all() and timeout_bootstrap=="continue":
        x=history(obs,actions,np.full(count,length),np.arange(count),context)
        with torch.no_grad():
            from model import scalar
            _,_,value=model.initial(torch.from_numpy(x).to(device))
            values[length]=scalar(value).cpu().numpy()*(state['alive']&~finished[:,None]).reshape(-1)
    targets=np.zeros_like(values)
    for t in range(length):
        weight=np.ones(count,np.float32);end=min(t+10,length)
        for j in range(t,end):
            targets[t]+=weight*rewards[j];weight*=model.config.discount*(~terminal[j])
        targets[t]+=weight*values[end]
    actual_returns=episode_returns(rewards[:length],terminal[:length],values[length],model.config.discount)
    if return_mode=='episode':targets[:length]=actual_returns[:length]
    completed=valid[:length]&np.repeat(finished,11)[None]
    search_return_mse=float(np.mean((values[:length][completed]-actual_returns[:length][completed])**2)) if completed.any() else None
    absorbing=np.zeros_like(valid)
    absorbing[1:length+1]=np.maximum.accumulate(terminal[:length],axis=0)
    indices=np.argwhere(valid[:length])
    block=dict(observation=obs[:length+1],actions=actions[:length],rewards=rewards[:length],policies=policies[:length+1],targets=targets[:length+1],valid=valid[:length+1],absorbing=absorbing[:length+1],indices=indices)
    stats=dict(collection_steps=length,padded_search_roots=length*len(searcher.indices),action_counts=action_counts.tolist(),hits=round(hits),movement_pixels=movement,completed_search_return_mse=search_return_mse,games=games,natural_terminal=int(finished.sum()),cutoffs=int((~finished).sum()),joint_steps=joint_steps,active_transitions=int(valid.sum()),**timings)
    return block,stats


def replay_batch(block,rng,batch,context,unroll,device):
    picks=block['indices'][rng.integers(len(block['indices']),size=batch)]
    times,agents=picks[:,0],picks[:,1];length=len(block['actions'])
    x=history(block['observation'],block['actions'],times,agents,context)
    positions=times[:,None]+np.arange(unroll+1)
    safe=positions.clip(max=length)
    v=block['targets'][safe,agents[:,None]]
    p=block['policies'][safe,agents[:,None]]
    valid=block['valid'][safe,agents[:,None]]&(positions<length)
    absorbing=block['absorbing'][safe,agents[:,None]]
    value_valid=valid|absorbing
    reward_valid=valid[:,:-1]|absorbing[:,:-1]
    positions=positions[:,:-1];safe=positions.clip(max=length-1)
    a=block['actions'][safe,agents[:,None]]
    a=np.where(absorbing[:,:-1],rng.integers(153,size=a.shape),a)
    r=block['rewards'][safe,agents[:,None]]*(positions<length)
    return tuple(torch.from_numpy(np.ascontiguousarray(z)).to(device) for z in (x,a,r,v,p,valid.astype(np.float32),value_valid.astype(np.float32),reward_valid.astype(np.float32)))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--loss-contract',choices=['baseline','root5-mass-v1'],default='baseline')
    p.add_argument('--site',required=True);p.add_argument('--output',required=True)
    p.add_argument('--resume-state',type=Path)
    p.add_argument('--games',type=int,default=10000);p.add_argument('--parallel',type=int,default=128)
    p.add_argument('--replay-blocks',type=int,default=3,choices=range(1,13))
    p.add_argument('--workers',type=int,default=8);p.add_argument('--hidden',type=int,default=384)
    p.add_argument('--context',type=int,default=64);p.add_argument('--simulations',type=int,default=8)
    p.add_argument('--batch',type=int,default=32);p.add_argument('--unroll',type=int,default=5)
    p.add_argument('--return-mode',choices=['nstep','episode'],default='nstep')
    p.add_argument('--timeout-bootstrap',choices=['continue','zero'],default='continue',help='zero uses the finite collection horizon; default preserves historical continuing targets')
    p.add_argument('--root-candidates',type=int,default=0)
    p.add_argument('--cadence',choices=['legacy','homepage'],default='homepage')
    p.add_argument('--agents',type=int,default=2,choices=range(2,11));p.add_argument('--policy-floor',type=float,default=0.0)
    p.add_argument('--noise-fraction',type=float,default=.25)
    p.add_argument('--discount',type=float,default=.99,choices=[.99,1.0])
    p.add_argument('--limit',type=int,default=256);p.add_argument('--updates-per-game',type=int,default=1)
    p.add_argument('--seed',type=int,default=28000000);p.add_argument('--max-seconds',type=float,default=600)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--device-history',action='store_true');p.add_argument('--cuda-graph',action='store_true');p.add_argument('--tf32',action='store_true')
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(a.seed);rng=np.random.default_rng(a.seed)
    torch.backends.cuda.matmul.allow_tf32=a.tf32
    model=model_class(a.loss_contract)(Config(observations=64*a.context+a.context-1,actions=153,hidden=a.hidden,simulations=a.simulations,discount=a.discount,support=10,exploration_fraction=a.noise_fraction)).to(a.device)
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4,weight_decay=1e-4)
    restored=None
    if a.resume_state:
        restored=torch.load(a.resume_state,map_location='cpu',weights_only=True)
        for key,value in vars(a).items():
            if key not in ('site','output','resume_state','games','max_seconds') and restored['arguments'][key]!=value:
                raise ValueError('Resume setting changed: '+key)
        model.load_state_dict(restored['model']);optimizer.load_state_dict(restored['optimizer'])
    site=Path(a.site); source={str(x.relative_to(site)):hashlib.sha256(x.read_bytes()).hexdigest() for x in [site/'src/sandboxCombat.ts',site/'src/sandboxMotion.ts',site/'src/lana/observation.ts']}
    provenance=capture_source(Path(__file__).resolve().parents[1])
    if restored and (restored['source_files']!=provenance['files'] or restored['torch_version']!=str(torch.__version__)):
        raise ValueError('Resume requires identical source and PyTorch version')
    arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    manifest=dict(arguments=arguments,config=asdict(model.config),parameters=sum(x.numel() for x in model.parameters()),torch=str(torch.__version__),python=platform.python_version(),environment=source,source_commit=provenance['commit'],source_provenance=provenance,gpu=torch.cuda.get_device_name() if a.device=='cuda' else None)
    if restored and restored['environment']!=source:raise ValueError('Resume environment changed')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    pool=Pool(a.site,a.workers,agents=a.agents,cadence=a.cadence);replay=deque(maxlen=a.replay_blocks);started=time.perf_counter();totals=dict(games=0,natural_terminal=0,cutoffs=0,joint_steps=0,active_transitions=0,optimizer_updates=0,training_examples=0)
    try:
        initial=restored['initial'] if restored else {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        np.savez_compressed(out/'initial.npz',**{k:v.numpy() for k,v in initial.items()})
        searcher=Search(model,11*a.parallel,a.cuda_graph,a.agents,a.root_candidates);setup=time.perf_counter()-started
        if restored:
            totals.update(restored['totals'])
            replay.extend({k:v.numpy() for k,v in b.items()} for b in restored['replay'])
            rng.bit_generator.state=restored['numpy_rng'];torch.set_rng_state(restored['torch_rng'])
            if a.device=='cuda':torch.cuda.set_rng_state_all(restored['cuda_rng'])
        last_saved = None
        def save_state():
            nonlocal last_saved
            if last_saved == (totals["games"], totals["optimizer_updates"]):
                return
            state=dict(arguments=arguments,environment=source,source_files=provenance['files'],torch_version=str(torch.__version__),initial=initial,model=model.state_dict(),optimizer=optimizer.state_dict(),totals=totals,replay=[{k:torch.from_numpy(v) for k,v in b.items()} for b in replay],numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if a.device=='cuda' else [])
            save_milestone(state, out, torch.save)
            last_saved = (totals['games'], totals['optimizer_updates'])
        for offset in range(totals['games'],a.games,a.parallel):
            if time.perf_counter()-started>=a.max_seconds:break
            count=min(a.parallel,a.games-offset)
            if count!=a.parallel:searcher=Search(model,11*count,a.cuda_graph,a.agents,a.root_candidates)
            block,stats=collect(pool,model,range(a.seed+offset,a.seed+offset+count),a.limit,a.context,searcher,a.device_history,a.return_mode,a.timeout_bootstrap)
            np.savez_compressed(out/f'trace-{offset}.npz',**{k:v[:,:22] for k,v in block.items() if k!='indices'})
            replay.append(block);updates=0;train_start=time.perf_counter()
            for _ in range(count*a.updates_per_game):
                chosen=replay[int(rng.integers(len(replay)))]
                tensors=replay_batch(chosen,rng,a.batch,a.context,a.unroll,a.device)
                if a.policy_floor:
                    tensors=list(tensors);tensors[4]=(1-a.policy_floor)*tensors[4]+a.policy_floor/model.config.actions
                loss=model.loss(*tensors).mean()
                optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),100.,error_if_nonfinite=True);optimizer.step();updates+=1
            train_time=time.perf_counter()-train_start
            for key in ('games','natural_terminal','cutoffs','joint_steps','active_transitions'):totals[key]+=stats[key]
            totals['optimizer_updates']+=updates;totals['training_examples']+=updates*a.batch
            elapsed=time.perf_counter()-started
            np.savez_compressed(out/f'checkpoint-{totals["games"]}.npz',**{k:v.detach().cpu().numpy() for k,v in model.state_dict().items()})
            row=dict(**totals,search_entropy=float(np.mean(searcher.entropies)),elapsed_seconds=elapsed,setup_seconds=setup,batch_seconds=stats,training_seconds=train_time,loss=float(loss.detach()) if updates else None,games_per_second=totals['games']/elapsed)
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True);searcher.entropies.clear()
            if totals['games']%512==0:save_state()
        save_state()
        save_start=time.perf_counter()
        np.savez_compressed(out/'weights.npz',**{k:v.detach().cpu().numpy() for k,v in model.state_dict().items()})
        elapsed=time.perf_counter()-started
        report=dict(**totals,elapsed_seconds=elapsed,checkpoint_seconds=time.perf_counter()-save_start,requested_run_completed=totals['games']==a.games,target_10000_under_600_met=totals['games']>=10000 and elapsed<=600,checkpoint_sha256=hashlib.sha256((out/'weights.npz').read_bytes()).hexdigest(),claim='Training throughput only; no policy quality or research reproduction claim')
        (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    finally:pool.close()

if __name__=='__main__':main()
