"""Experimental inference adapter. No training integration or strength claim."""
import copy
from dataclasses import replace
import torch


@torch.no_grad()
def split_search(model, observations, search_fn, *, groups=4, simulations=8,
                 roots=2, root_noise=None, root_candidates=None):
    if groups < 1 or simulations < 1 or not 1 <= roots <= simulations:
        raise ValueError('invalid search budget')
    if groups * simulations != model.config.simulations:
        raise ValueError('total expansions must match caller budget')
    if observations.ndim != 2 or not len(observations):
        raise ValueError('nonempty observation matrix required')
    # Shallow module view shares read-only parameters, never changes caller config.
    view=copy.copy(model)
    view.config=replace(model.config,simulations=simulations)
    batch=len(observations);actions=model.config.actions
    x=observations.repeat_interleave(groups,dim=0)
    if root_noise is None:
        concentration=x.new_full((batch*groups,actions),model.config.dirichlet_alpha)
        noise=torch.distributions.Dirichlet(concentration).sample()
    else:
        if root_noise.shape != (batch,groups,actions):raise ValueError('noise shape')
        noise=root_noise.reshape(batch*groups,actions)
    if not torch.isfinite(noise).all() or (noise<0).any() or not torch.allclose(noise.sum(-1),noise.new_ones(len(noise)),atol=1e-6):
        raise ValueError('noise probabilities')
    if root_candidates is None:
        _,logits,_=view.initial(x)
        mass=(1-model.config.exploration_fraction)*logits.softmax(-1)+model.config.exploration_fraction*noise
        candidates=torch.multinomial(mass,roots,replacement=False)
    else:
        if root_candidates.shape != (batch,groups,roots):raise ValueError('candidate shape')
        candidates=root_candidates.reshape(batch*groups,roots)
        if candidates.dtype != torch.long or (candidates<0).any() or (candidates>=actions).any():raise ValueError('candidate actions')
        if roots>1 and (candidates.sort(-1).values.diff(dim=-1)==0).any():raise ValueError('duplicate roots')
    policies,values=search_fn(view,x,root_noise=noise,root_candidates=candidates)
    if policies.shape!=(batch*groups,actions) or values.shape!=(batch*groups,):raise ValueError('backend shape')
    if not torch.isfinite(policies).all() or (policies<0).any() or not torch.allclose(policies.sum(-1),policies.new_ones(len(policies)),atol=1e-6):raise ValueError('backend policy')
    if not torch.isfinite(values).all():raise ValueError('backend value')
    return policies.reshape(batch,groups,actions).mean(1),values.reshape(batch,groups).mean(1)
