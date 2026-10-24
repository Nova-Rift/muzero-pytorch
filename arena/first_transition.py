"""Original objective with only the first recurrent hidden gradient unattenuated."""
import torch
from model import MuZero, scale_gradient, support_targets
from arena.rootweighted import model_class as existing_model_class


class FirstTransitionMuZero(MuZero):
    def loss(self, observations, actions, rewards, values, policies, valid=None, value_valid=None, reward_valid=None):
        """B,K actions/rewards; B,K+1 value/policy targets, incl root.

        valid masks policy targets; rewards[:,k] is earned by actions[:,k].
        value_valid/reward_valid can include absorbing terminal extensions, while
        excluding unavailable transitions after a time-limit cutoff.
        Targets must already distinguish terminal zero-bootstrap from truncation.
        Returns per-sample losses for prioritized replay importance weighting.
        """
        steps = actions.shape[1]
        if valid is None:
            valid = torch.ones_like(values)
        if value_valid is None: value_valid = valid
        if reward_valid is None: reward_valid = valid[:, :-1]
        state, policy, value = self.initial(observations)
        total = observations.new_zeros(observations.shape[0])
        for t in range(steps + 1):
            vloss = -(support_targets(values[:, t], self.config.support) * value.log_softmax(-1)).sum(-1)
            ploss = -(policies[:, t] * policy.log_softmax(-1)).sum(-1)
            term = vloss * value_valid[:, t] + ploss * valid[:, t]
            if t:
                rloss = -(support_targets(rewards[:, t-1], self.config.support) * reward.log_softmax(-1)).sum(-1)
                term = term + rloss * reward_valid[:, t-1]
            total = total + term / (steps + 1)
            if t < steps:
                state, reward, policy, value = self.recurrent(scale_gradient(state, 1.0 if t == 0 else .5), actions[:, t])
        return total



def model_class(contract):
    if contract == 'first-transition1-v1':
        return FirstTransitionMuZero
    return existing_model_class(contract)
