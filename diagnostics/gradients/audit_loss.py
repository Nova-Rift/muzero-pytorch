"""CPU gradient-contract probe. Executes pinned local/Google loss source.

Google's TF operations are mapped to Torch, not a TensorFlow parity claim.
No downloaded third-party source is vendored; supply the SHA-bound learner file.
"""
import argparse
import ast
import collections
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
import torch

LOCAL_SHA = 'c546948594aa70456b9361ccd4e5acc82fa2790b3d3c9d069fb769f095888e17'
GOOGLE_SHA = 'c5b2fedee2bce4da4ef9bba42f4583099f81b8fccd3cce629a12b716b3ba62fe'
GOOGLE_REV = '5b09c22d73a9d35eb6c5d2a99b95677a45053466'
ROOT = Path(__file__).resolve().parents[2]


def checked_source(path, expected):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('Source SHA256 mismatch')
    return data.decode()


def local_module():
    checked_source(ROOT / 'model.py', LOCAL_SHA)
    spec = importlib.util.spec_from_file_location('gradient_local_model', ROOT / 'model.py')
    module = importlib.util.module_from_spec(spec)
    # dataclass annotations are concrete; register for future compatibility.
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def reference_loss(path):
    """Execute unchanged function AST bodies, with only annotations removed."""
    tree = ast.parse(checked_source(path, GOOGLE_SHA))
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in ('scale_gradient', 'compute_loss')]
    if len(funcs) != 2:
        raise ValueError('Missing reference functions')
    for fn in funcs:
        fn.returns = None
        for arg in fn.args.args:
            arg.annotation = None
    maximum = lambda x, y: torch.maximum(x, torch.as_tensor(y, dtype=x.dtype))
    tf = NS(stop_gradient=lambda x: x.detach(), ones_like=torch.ones_like,
            reshape=lambda x, shape: x.reshape(shape), reduce_sum=lambda x, axis=None: x.sum(dim=axis),
            reduce_mean=torch.mean, maximum=maximum, abs=torch.abs, squeeze=torch.squeeze,
            stack=lambda xs, axis: torch.stack(xs, dim=axis),
            argmax=lambda x, axis: x.argmax(dim=axis), cast=lambda x, dtype: x.to(dtype),
            float32=torch.float32,
            math=NS(divide=lambda x, y: x/y,
                    divide_no_nan=lambda x, y: torch.where(y != 0, x/y, torch.zeros_like(x))),
            nn=NS(softmax_cross_entropy_with_logits=lambda logits, labels:
                  -(labels * logits.log_softmax(-1)).sum(-1),
                  softmax=lambda x, axis: x.softmax(axis)),
            keras=NS(metrics=NS(categorical_accuracy=lambda labels, preds:
                     (labels.argmax(-1) == preds.argmax(-1)).to(labels.dtype))))
    scope = dict(tf=tf, collections=collections, core=NS(Prediction=lambda **kw: NS(**kw)), log_keys=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=funcs, type_ignores=[])),
                 '<pinned-google-learner-functions>', 'exec'), scope)
    return scope['compute_loss']


class Fixture:
    """Independent logits expose each term; identity hidden path exposes attenuation."""
    def __init__(self, model, batch=2, steps=5, hidden_probe=False):
        self.config = NS(support=1)
        self.leaves = {head: torch.zeros(batch, steps+1, width, dtype=torch.float64,
                                         requires_grad=True)
                       for head, width in [('value',3), ('reward',3), ('policy',2)]}
        self.root = torch.zeros(batch, 1, dtype=torch.float64, requires_grad=True)
        self.hidden_probe = hidden_probe
        self.t = 0
        self.hidden = []
        self.value_encoder = self.reward_encoder = NS(
            num_steps=3, encode=lambda x: model.support_targets(x, 1))

    def output(self, state):
        state.retain_grad()
        self.hidden.append(state)
        v, r, p = [self.leaves[h][:, self.t] for h in ('value','reward','policy')]
        if self.hidden_probe:
            p = p + torch.cat((state, state * 0), -1)
        zeros = torch.zeros(state.shape[0], 1, dtype=state.dtype)
        return NS(hidden_state=state, value_logits=v, reward_logits=r,
                  policy_logits=p, value=zeros, reward=zeros)

    def initial_inference(self, obs):
        self.t = 0
        return self.output(self.root)

    def recurrent_inference(self, state, action):
        self.t += 1
        return self.output(state + 0)

    def initial(self, obs):
        o = self.initial_inference(obs)
        return o.hidden_state, o.policy_logits, o.value_logits

    def recurrent(self, state, action):
        o = self.recurrent_inference(state, action)
        return o.hidden_state, o.reward_logits, o.policy_logits, o.value_logits


def probe(model, ref, contract, masks, hidden_probe=False):
    b, n = masks['value'].shape
    f = Fixture(model, b, n-1, hidden_probe)
    obs = torch.zeros(b, 1, dtype=torch.float64)
    actions = torch.zeros(b, n-1, dtype=torch.int64)
    scalar_targets = torch.zeros(b, n, dtype=torch.float64)
    policy_targets = torch.zeros(b, n, 2, dtype=torch.float64)
    policy_targets[:, :, 1] = 1
    if contract == 'local':
        loss = model.MuZero.loss(f, obs, actions, scalar_targets[:,1:], scalar_targets,
                                policy_targets, masks['policy'], masks['value'], masks['reward'][:,1:]).sum()
    else:
        config = NS(policy_loss_entropy_regularizer=0., reward_loss_scaling=1.,
                    policy_loss_scaling=1., weight_decay=0., debug=False)
        distribution = NS(entropy=lambda logits: -(logits.softmax(-1) * logits.log_softmax(-1)).sum(-1))
        loss, _ = ref(config, distribution, f, torch.ones(b, dtype=torch.float64),
                      obs, actions, masks['value'], masks['reward'], masks['policy'],
                      scalar_targets, scalar_targets, policy_targets)
        loss = loss * b  # Google batch mean -> local batch sum, preserving time weights.
    loss.backward()
    coefficients = {head: (leaf.grad[:,:,0] / (0.5 if head == 'policy' else 1/3)).tolist()
                    for head, leaf in f.leaves.items()}
    return dict(loss_forward=float(loss.detach()), coefficients=coefficients,
                root_hidden_gradient=None if f.root.grad is None else f.root.grad[:,0].tolist(),
                hidden_gradients=[None if x.grad is None else x.grad[:,0].tolist() for x in f.hidden])


def run(reference):
    torch.set_num_threads(1)
    model, ref = local_module(), reference_loss(reference)
    full = {h: torch.ones(2, 6, dtype=torch.float64) for h in ('policy','value','reward')}
    full['reward'][:,0] = 0  # local has no initial reward loss
    sparse = {h: m.clone() for h,m in full.items()}
    sparse['policy'][0, 2:] = 0
    sparse['policy'][1, 1:] = 0
    sparse['value'][0, 3:] = 0
    sparse['value'][1, 1:] = 0
    sparse['reward'][0, 4:] = 0
    sparse['reward'][1, 1:] = 0
    result = dict(local_model_sha256=LOCAL_SHA, google_learner_sha256=GOOGLE_SHA,
                  google_revision=GOOGLE_REV, torch_version=torch.__version__, cases={})
    for name, masks in [('all_valid', full), ('component_masks', sparse)]:
        result['cases'][name] = dict(masks={h:m.tolist() for h,m in masks.items()},
            **{c:probe(model,ref,c,masks) for c in ('local','reference')})
    result['hidden_isolated'] = []
    for t in range(6):
        masks = {h:torch.zeros_like(m) for h,m in full.items()}
        masks['policy'][:,t] = 1
        result['hidden_isolated'].append(dict(target_step=t,
            **{c:probe(model,ref,c,masks,True) for c in ('local','reference')}))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-learner', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(run(args.reference_learner), indent=2)+'\n')
