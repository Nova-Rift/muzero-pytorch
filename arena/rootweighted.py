"""Single-factor root policy/value redistribution; architecture/search unchanged.

For each binary head mask, retain total coefficient mass valid_count/(K+1)
while assigning root:each recurrent term 5:1. Reward and hidden gradients retain
original MuZero semantics. This is an experimental objective, not a strength claim.
"""
import torch
from model import MuZero, scale_gradient, support_targets


def coefficients(mask):
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError('Binary masks required')
    mass = mask.sum(1, keepdim=True)
    weighted = mask * 1.0
    weighted[:, 0] *= 5
    return weighted * mass / (mask.shape[1] * weighted.sum(1, keepdim=True).clamp_min(1))


class RootWeightedMuZero(MuZero):
    def loss(self, observations, actions, rewards, values, policies, valid=None,
             value_valid=None, reward_valid=None):
        steps = actions.shape[1]
        if valid is None:
            valid = torch.ones_like(values)
        if value_valid is None:
            value_valid = valid
        if reward_valid is None:
            reward_valid = valid[:, :-1]
        pc, vc = coefficients(valid), coefficients(value_valid)
        if not torch.all((reward_valid == 0) | (reward_valid == 1)):
            raise ValueError('Binary reward masks required')
        state, policy, value = self.initial(observations)
        total = observations.new_zeros(observations.shape[0])
        for t in range(steps + 1):
            vloss = -(support_targets(values[:, t], self.config.support) * value.log_softmax(-1)).sum(-1)
            ploss = -(policies[:, t] * policy.log_softmax(-1)).sum(-1)
            term = ploss * pc[:, t] + vloss * vc[:, t]
            if t:
                rloss = -(support_targets(rewards[:, t-1], self.config.support) * reward.log_softmax(-1)).sum(-1)
                term = term + rloss * reward_valid[:, t-1] / (steps + 1)
            total = total + term
            if t < steps:
                state, reward, policy, value = self.recurrent(scale_gradient(state, .5), actions[:, t])
        return total


def model_class(contract):
    if contract == 'baseline':
        return MuZero
    if contract == 'root5-mass-v1':
        return RootWeightedMuZero
    raise ValueError('Unknown loss contract: ' + contract)
