"""Frozen source-bound comparison; changes only unvisited Q completion in memory."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

NORMALIZE = '            q = torch.where((high>low)[:,None],(q-low[:,None])/(high-low).clamp_min(1e-8)[:,None],q)'


def loaded_search(source, reference):
    text = source.read_text()
    assert text.count(NORMALIZE) == 1, 'Source contract changed; review required'
    if reference:
        text = text.replace(NORMALIZE, NORMALIZE + '\n            q = torch.where(nv>0,q,torch.zeros_like(q))')
    namespace = {}
    exec(compile(text, str(source), 'exec'), namespace)
    return namespace['search_batch']


def algebra_fixture(source):
    text = source.read_text()
    begin = text.index('            q = torch.where(nv>0,q,torch.zeros_like(q))')
    end = text.index('\n', text.index(NORMALIZE))
    statements = '\n'.join(line.strip() for line in text[begin:end].splitlines())
    rows = []
    for low, high in [(0.2,0.4),(-0.4,-0.2),(-0.2,0.2),(0.,0.4),(0.2,0.2)]:
        ns = dict(torch=torch, q=torch.tensor([[0.,high]]), nv=torch.tensor([[0,1]]),
                  low=torch.tensor([low]), high=torch.tensor([high]))
        exec(statements, ns)
        actual = ns['q']
        reference = torch.where(ns['nv']>0, actual, torch.zeros_like(actual))
        assert actual[0,1] == reference[0,1]
        rows.append(dict(low=low, high=high, current=actual.tolist(), reference=reference.tolist()))
    return rows


def run(source_root, recovered):
    sys.path.insert(0, str(source_root))
    from model import Config, MuZero
    from arena.train_selfplay import history
    torch.set_num_threads(2)
    torch.manual_seed(23001)
    config = Config(observations=4159,actions=153,hidden=448,support=10,
                    discount=1.,simulations=32)
    model = MuZero(config).eval()
    weights = recovered/'checkpoint-640.npz'
    with np.load(weights) as data:
        model.load_state_dict({key:torch.from_numpy(data[key].copy()) for key in data.files})
    trace = recovered/'trace-512.npz'
    with np.load(trace) as data:
        # Predeclared evenly spaced valid root histories from development training trace.
        available = np.argwhere(data['valid'][:-1,:2])
        selected = available[np.linspace(0,len(available)-1,32,dtype=int)]
        x = torch.from_numpy(history(data['observation'],data['actions'],selected[:,0],selected[:,1],64))
    with torch.no_grad():
        _, logits, _ = model.initial(x)
        candidates = torch.multinomial(logits.softmax(-1),8,replacement=False)
    source = source_root/'batched_search_exploration.py'
    runs = {}
    for label, reference in [('current',False),('reference_completion',True)]:
        start = time.monotonic()
        policy, value = loaded_search(source,reference)(model,x,root_candidates=candidates)
        runs[label] = dict(policy=policy.tolist(),value=value.tolist(),seconds=time.monotonic()-start)
    old = np.array(runs['current']['policy']);new = np.array(runs['reference_completion']['policy'])
    return dict(scope='Frozen CPU diagnostic, no games, updates, or strength claim',
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
                roots=selected.tolist(),root_candidates=candidates.tolist(),seed=23001,
                simulations=32,noise=False,fixture=algebra_fixture(source),runs=runs,
                changed_policies=int(np.any(old!=new,axis=1).sum()),
                changed_argmax=int((old.argmax(-1)!=new.argmax(-1)).sum()),
                mean_total_variation=float(np.abs(old-new).sum(-1).mean()/2),
                mean_support_current=float((old>0).sum(-1).mean()),
                mean_support_reference=float((new>0).sum(-1).mean()))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--recovered',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=run(a.source.resolve(),a.recovered.resolve())
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['changed_policies','changed_argmax','mean_total_variation','mean_support_current','mean_support_reference']}))
