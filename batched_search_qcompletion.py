"""Tensor-backed MCTS across independent roots; same model and backup convention.

One neural expansion per simulation for the whole batch. Trees never share state.
No per-root Python objects or GPU-to-CPU copies inside the search.
"""
import math
import torch
from model import scalar


@torch.no_grad()
def search_batch(model, observations, *, explore=False, legal=None, root_noise=None, static=False, root_candidates=None, root_temperature=1.0):
    c = model.config
    batch, device = len(observations), observations.device
    rows = torch.arange(batch, device=device)
    count = c.simulations+1
    hidden = observations.new_zeros(batch,count,c.hidden)
    priors = observations.new_zeros(batch,count,c.actions)
    reward = observations.new_zeros(batch,count)
    value_sum = observations.new_zeros(batch,count)
    visits = torch.zeros(batch,count,device=device,dtype=torch.long)
    children = torch.zeros(batch,count,c.actions,device=device,dtype=torch.long)
    hidden[:,0], policy, _ = model.initial(observations)
    if legal is not None:
        if not legal.any(-1).all(): raise ValueError('Empty legal action mask')
        policy = policy.masked_fill(~legal.bool(), -torch.inf)
    # root_temperature>1 flattens the ROOT prior only (self-play exploration flag); 1.0 is the exact historical path.
    priors[:,0] = policy.softmax(-1) if root_temperature==1.0 else (policy/root_temperature).softmax(-1)
    if explore or root_noise is not None:
        concentration = torch.full_like(priors[:,0],c.dirichlet_alpha)
        noise = torch.distributions.Dirichlet(concentration).sample() if root_noise is None else root_noise
        if legal is not None:
            noise *= legal; noise /= noise.sum(-1,keepdim=True)
        priors[:,0] = (1-c.exploration_fraction)*priors[:,0]+c.exploration_fraction*noise
    low = observations.new_full((batch,),float('inf'))
    high = -low
    for simulation in range(1,count):
        node = torch.zeros(batch,device=device,dtype=torch.long)
        active = torch.ones(batch,device=device,dtype=torch.bool)
        parents = torch.zeros_like(node)
        selected = torch.zeros_like(node)
        path_nodes, path_active = [], []
        # A tree has at most simulation-1 expanded edges before this expansion.
        for depth in range(simulation):
            path_nodes.append(node.clone());path_active.append(active.clone())
            next_ids = children[rows,node]
            nv = torch.where(next_ids>0,visits.gather(1,next_ids),torch.zeros_like(next_ids))
            q = reward.gather(1,next_ids)+c.discount*value_sum.gather(1,next_ids)/nv.clamp_min(1)
            q = torch.where(nv>0,q,torch.zeros_like(q))
            q = torch.where((high>low)[:,None],(q-low[:,None])/(high-low).clamp_min(1e-8)[:,None],q)
            q = torch.where(nv>0,q,torch.zeros_like(q))
            n = visits[rows,node]
            coef = torch.log((n+c.pb_c_base+1)/c.pb_c_base)+c.pb_c_init
            scores = q+coef[:,None]*torch.sqrt(n.float()+1)[:,None]*priors[rows,node]/(nv+1)
            if depth==0 and legal is not None: scores = scores.masked_fill(~legal,-torch.inf)
            action = scores.argmax(-1)
            if depth==0 and root_candidates is not None and simulation<=root_candidates.shape[1]:
                action=root_candidates[:,simulation-1]
            child = next_ids[rows,action]
            found = active & (child==0)
            parents = torch.where(found,node,parents)
            selected = torch.where(found,action,selected)
            active = active & ~found
            node = torch.where(active,child,node)
            if not static and not active.any(): break
        h,r,p,v = model.recurrent(hidden[rows,parents],selected)
        hidden[:,simulation] = h
        reward[:,simulation] = scalar(r)
        priors[:,simulation] = p.softmax(-1)
        children[rows,parents,selected] = torch.full_like(selected,simulation)
        backed = scalar(v)
        visits[:,simulation] = 1
        value_sum[:,simulation] = backed
        q = reward[:,simulation]+c.discount*backed
        low,high = torch.minimum(low,q),torch.maximum(high,q)
        backed = q
        for node,mask in zip(reversed(path_nodes),reversed(path_active)):
            old = value_sum[rows,node]
            value_sum[rows,node] = old+backed*mask
            visits[rows,node] += mask.long()
            q = reward[rows,node]+c.discount*value_sum[rows,node]/visits[rows,node].clamp_min(1)
            low = torch.where(mask,torch.minimum(low,q),low)
            high = torch.where(mask,torch.maximum(high,q),high)
            backed = torch.where(mask,reward[rows,node]+c.discount*backed,backed)
    root_children = children[:,0]
    child_visits = visits.gather(1,root_children)
    child_visits = torch.where(root_children>0,child_visits,torch.zeros_like(child_visits))
    return child_visits.float()/c.simulations,value_sum[:,0]/visits[:,0]
