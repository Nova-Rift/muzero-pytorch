"""Independent vector-observation MuZero: h (encode), g (dynamics), f (predict).

Paper: https://arxiv.org/abs/1911.08265 . No third-party model code.
Search never receives an environment. This is a single-player reward convention.
"""
from dataclasses import dataclass, field
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Config:
    observations: int = 8
    actions: int = 2
    hidden: int = 128
    support: int = 20
    discount: float = 0.997
    simulations: int = 50
    pb_c_base: float = 19652.
    pb_c_init: float = 1.25
    dirichlet_alpha: float = 0.3
    exploration_fraction: float = 0.25

    def __post_init__(self):
        if min(self.observations, self.actions, self.hidden, self.support, self.simulations) < 1:
            raise ValueError('Positive dimensions and simulation count required')
        if not 0 <= self.discount <= 1:
            raise ValueError('discount must be in [0,1]')


def transform(x):
    return x.sign() * (torch.sqrt(x.abs() + 1) - 1) + .001 * x


def inverse_transform(x):
    return x.sign() * (((torch.sqrt(1 + .004 * (x.abs() + 1.001)) - 1) / .002)**2 - 1)


def support_targets(x, size):
    x = transform(x).clamp(-size, size) + size
    low = x.floor().long()
    high = (low + 1).clamp(max=2 * size)
    fraction = x - low
    target = x.new_zeros(*x.shape, 2 * size + 1)
    target.scatter_add_(-1, low[..., None], (1 - fraction)[..., None])
    target.scatter_add_(-1, high[..., None], fraction[..., None])
    return target


def scalar(logits):
    size = (logits.shape[-1] - 1) // 2
    support = torch.arange(-size, size + 1, device=logits.device, dtype=logits.dtype)
    return inverse_transform((logits.softmax(-1) * support).sum(-1))


def scale_gradient(x, scale):
    return x * scale + x.detach() * (1 - scale)


def mlp(inputs, width, outputs):
    return nn.Sequential(nn.Linear(inputs, width), nn.ReLU(), nn.Linear(width, outputs))


class MuZero(nn.Module):
    def __init__(self, config=Config()):
        super().__init__()
        self.config = c = config
        self.representation = mlp(c.observations, c.hidden, c.hidden)
        self.dynamics = mlp(c.hidden + c.actions, c.hidden, c.hidden)
        self.reward = mlp(c.hidden, c.hidden, 2 * c.support + 1)
        self.policy = mlp(c.hidden, c.hidden, c.actions)
        self.value = mlp(c.hidden, c.hidden, 2 * c.support + 1)

    @staticmethod
    def normalize(state):
        low, high = state.amin(-1, keepdim=True), state.amax(-1, keepdim=True)
        return (state - low) / (high - low).clamp_min(1e-5)

    def initial(self, observations):
        state = self.normalize(self.representation(observations))
        return state, self.policy(state), self.value(state)

    def recurrent(self, state, actions):
        onehot = F.one_hot(actions.long(), self.config.actions).to(state.dtype)
        raw = self.dynamics(torch.cat((state, onehot), -1))
        reward = self.reward(raw)
        state = self.normalize(raw)
        return state, reward, self.policy(state), self.value(state)

    def loss(self, observations, actions, rewards, values, policies, valid=None):
        """B,K actions/rewards; B,K+1 value/policy targets, incl root.

        valid masks padded states; rewards[:,k] is earned by actions[:,k].
        Targets must already distinguish terminal zero-bootstrap from truncation.
        Returns per-sample losses for prioritized replay importance weighting.
        """
        steps = actions.shape[1]
        if valid is None:
            valid = torch.ones_like(values)
        state, policy, value = self.initial(observations)
        total = observations.new_zeros(observations.shape[0])
        for t in range(steps + 1):
            vloss = -(support_targets(values[:, t], self.config.support) * value.log_softmax(-1)).sum(-1)
            ploss = -(policies[:, t] * policy.log_softmax(-1)).sum(-1)
            term = (vloss + ploss) * valid[:, t]
            if t:
                rloss = -(support_targets(rewards[:, t-1], self.config.support) * reward.log_softmax(-1)).sum(-1)
                term = term + rloss * valid[:, t-1]
            total = total + term / (steps + 1)
            if t < steps:
                state, reward, policy, value = self.recurrent(scale_gradient(state, .5), actions[:, t])
        return total


@dataclass
class Node:
    prior: float
    reward: float = 0.
    visits: int = 0
    value_sum: float = 0.
    state: object = None
    children: dict = field(default_factory=dict)

    @property
    def value(self):
        return self.value_sum / self.visits if self.visits else 0.


@torch.no_grad()
def search(model, observation, *, legal=None, explore=False):
    """One root, persistent within this call; returns visit policy and root value.

    Legal masks apply at the real root only: MuZero does not know future rules.
    Noise and sampling use PyTorch's RNG (seed it for reproducible experiments).
    """
    c = model.config
    state, logits, value = model.initial(observation[None])
    legal = torch.ones(c.actions, dtype=torch.bool, device=observation.device) if legal is None else legal.bool()
    if not legal.any():
        raise ValueError('Root has no legal actions')
    root = Node(1., state=state)

    def expand(node, logits, mask):
        probs = logits[0].masked_fill(~mask, -torch.inf).softmax(-1)
        node.children = {a: Node(float(probs[a])) for a in range(c.actions) if mask[a]}

    expand(root, logits, legal)
    if explore:
        noise = torch.distributions.Dirichlet(observation.new_full((len(root.children),), c.dirichlet_alpha)).sample()
        for child, n in zip(root.children.values(), noise):
            child.prior = (1-c.exploration_fraction)*child.prior + c.exploration_fraction*float(n)
    minimum, maximum = math.inf, -math.inf
    for _ in range(c.simulations):
        node, path = root, [root]
        while node.children:
            def score(item):
                _, child = item
                q = child.reward + c.discount * child.value if child.visits else 0.
                q = (q-minimum)/(maximum-minimum) if maximum > minimum else q
                coefficient = math.log((node.visits+c.pb_c_base+1)/c.pb_c_base)+c.pb_c_init
                return q + coefficient * math.sqrt(node.visits+1)/(child.visits+1)*child.prior
            action, child = max(node.children.items(), key=score)
            parent, node = node, child
            path.append(node)
        state, reward, logits, value = model.recurrent(parent.state, torch.tensor([action], device=observation.device))
        node.state, node.reward = state, float(scalar(reward)[0])
        expand(node, logits, torch.ones_like(legal))
        backed = float(scalar(value)[0])
        for visited in reversed(path):
            visited.visits += 1
            visited.value_sum += backed
            q = visited.reward + c.discount * visited.value
            minimum, maximum = min(minimum, q), max(maximum, q)
            backed = visited.reward + c.discount * backed
    visits = observation.new_zeros(c.actions)
    for action, child in root.children.items():
        visits[action] = child.visits
    return visits / visits.sum(), root.value


def nstep_returns(rewards, values, terminated, discount=.997, steps=10):
    """values has T+1 entries, including the actual final-observation bootstrap.

    Supply one episode, stopping on termination OR truncation. Truncation keeps
    final value; termination kills it. Never bootstrap from a reset observation.
    """
    targets = []
    for start in range(len(rewards)):
        result, weight = 0., 1.
        end = min(start + steps, len(rewards))
        for t in range(start, end):
            result += weight * rewards[t]
            weight *= discount * (not terminated[t])
        targets.append(result + weight * values[end])
    return targets
