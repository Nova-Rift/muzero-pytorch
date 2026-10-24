"""Read-only CPU gradient probe; never trains, clips, or changes stored state."""
import argparse, copy, gzip, hashlib, json, shutil, signal, sys, tempfile
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model import Config, MuZero, scale_gradient, support_targets
from arena.train_selfplay import replay_batch
ROOT = Path(__file__).resolve().parents[2]
GROUPS = ('representation', 'dynamics', 'policy', 'value', 'reward')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''): h.update(b)
    return h.hexdigest()

def tensor_sha(t):
    t = t.detach().cpu().contiguous()
    return hashlib.sha256(str((str(t.dtype), list(t.shape))).encode()+t.numpy().tobytes()).hexdigest()

def coefficients(mask, ratio=1.):
    if not torch.all((mask == 0) | (mask == 1)): raise ValueError('Binary masks required')
    mass = mask.sum(1, keepdim=True)
    weighted = mask.clone(); weighted[:, 0] *= ratio
    return weighted * mass / (mask.shape[1] * weighted.sum(1, keepdim=True).clamp_min(1))

def terms(model, batch, ratio=1.):
    x, actions, rewards, values, policies, valid, vv, rv = batch
    pc, vc = coefficients(valid, ratio), coefficients(vv, ratio)
    if not torch.all((rv == 0) | (rv == 1)): raise ValueError('Binary reward masks required')
    state, policy, value = model.initial(x)
    result = []
    for t in range(actions.shape[1]+1):
        row = {'policy': -(policies[:, t]*policy.log_softmax(-1)).sum(-1),
               'value': -(support_targets(values[:, t], model.config.support)*value.log_softmax(-1)).sum(-1)}
        weighted = row['policy']*pc[:, t] + row['value']*vc[:, t]
        masks = {'policy': valid[:, t], 'value': vv[:, t]}
        if t:
            row['reward'] = -(support_targets(rewards[:, t-1], model.config.support)*reward.log_softmax(-1)).sum(-1)
            weighted = weighted + row['reward']*rv[:, t-1]/(actions.shape[1]+1)
            masks['reward'] = rv[:, t-1]
        if ratio == 1.:
            weighted = row['value']*vv[:, t] + row['policy']*valid[:, t]
            if t: weighted = weighted + row['reward']*rv[:, t-1]
            weighted = weighted/(actions.shape[1]+1)
        result.append((weighted, row, masks))
        if t < actions.shape[1]:
            state, reward, policy, value = model.recurrent(scale_gradient(state, .5), actions[:, t])
    return result

def gradients(loss, model):
    if not torch.isfinite(loss).all(): raise ValueError("Nonfinite loss")
    named = list(model.named_parameters())
    if any(n.split(".")[0] not in GROUPS for n,_ in named): raise ValueError("Uncovered parameter group")
    grads = torch.autograd.grad(loss, [p for _, p in named], allow_unused=True, retain_graph=True)
    if any(g is not None and not torch.isfinite(g).all() for g in grads): raise ValueError("Nonfinite gradient")
    return {n: torch.zeros_like(p) if g is None else g for (n,p),g in zip(named,grads)}

def norms(g):
    out = {group: sum(float(v.double().square().sum()) for n,v in g.items() if n.split('.')[0] == group)**.5 for group in GROUPS}
    out['total'] = sum(x*x for x in out.values())**.5
    out['clip100_ratio'] = out['total']/100
    return out

def cosine(a,b):
    dot = sum(float((a[n].double()*b[n].double()).sum()) for n in a)
    den = norms(a)['total']*norms(b)['total']
    return dot/den if den else None

def measure(model, batch):
    original = model.loss(*batch).mean(); og = gradients(original, model)
    out = {'original': {'loss': float(original), 'gradients': norms(og)}}
    for label, ratio in [('reconstructed',1.),('root5',5.)]:
        ts = terms(model,batch,ratio)
        loss = sum(t[0] for t in ts).mean(); g = gradients(loss,model)
        root = gradients(ts[0][0].mean(),model)
        recurrent = gradients(sum(t[0] for t in ts[1:]).mean(),model)
        depths = [{h: {'valid_count': int(m[h].sum()), 'masked_loss_mean': float((v*m[h]).mean()), 'valid_loss_mean': float((v*m[h]).sum()/m[h].sum()) if m[h].sum() else None} for h,v in row.items()} for _,row,m in ts]
        out[label] = {'loss':float(loss),'gradients':norms(g),'root_gradients':norms(root),'recurrent_gradients':norms(recurrent),'root_recurrent_cosine':cosine(root,recurrent),'root_recurrent_group_cosines':{h:cosine({n:v for n,v in root.items() if n.split('.')[0]==h},{n:v for n,v in recurrent.items() if n.split('.')[0]==h}) for h in GROUPS},'per_depth':depths}
        if label == 'reconstructed':
            error = max(float((g[n]-og[n]).abs().max()) for n in g)
            out[label]['max_gradient_error'] = error
            torch.testing.assert_close(loss,original,rtol=2e-6,atol=2e-6)
            for n in g: torch.testing.assert_close(g[n],og[n],rtol=2e-5,atol=2e-6)
    if any(p.grad is not None for p in model.parameters()): raise AssertionError('Parameter grad mutation')
    return out

class RecordingRNG:
    def __init__(self,rng): self.rng=rng; self.calls=[]
    def integers(self,*args,**kwargs):
        result=self.rng.integers(*args,**kwargs); self.calls.append(np.asarray(result).copy()); return result

def run(a):
    torch.set_num_threads(1)
    if not (1 <= a.batch <= 256 and 1 <= a.count <= 4): raise ValueError('Bounded batch/count required')
    a.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(a.manifest.read_text())['states'][a.state.name.removesuffix('.gz')]
    if sha(a.state)!=manifest['compressed_sha256']: raise ValueError('Compressed hash mismatch')
    if shutil.disk_usage(a.output).free < manifest['raw_bytes'] + 750*1024**2: raise ValueError('Insufficient decode space plus 750MiB reserve')
    with tempfile.TemporaryDirectory(prefix='gradient-raw-',dir=a.output) as tmp:
        raw=Path(tmp)/'state.pt'; h=hashlib.sha256(); size=0
        with gzip.open(a.state,'rb') as src, raw.open('wb') as dst:
            for b in iter(lambda:src.read(8*1024**2),b''): dst.write(b); h.update(b); size+=len(b)
        if size!=manifest['raw_bytes'] or h.hexdigest()!=manifest['raw_sha256']: raise ValueError('Raw mismatch')
        state=torch.load(raw,map_location='cpu',weights_only=True)
    source=json.loads(a.source_manifest.read_text())['files']
    if state['source_files']!=source: raise ValueError('Saved source binding mismatch')
    for name in ('model.py','arena/train_selfplay.py'):
        if sha(ROOT/name)!=source[name]: raise ValueError('Diagnostic baseline source drift: '+name)
    args=state['arguments']
    if args['unroll']!=5 or args['policy_floor']!=0: raise ValueError('Unsupported training contract')
    rng=np.random.default_rng(a.seed); batches=[]; selection=[]
    for i in range(a.count):
        bi=i%len(state['replay']); block={k:v.numpy() for k,v in state['replay'][bi].items()}
        before=copy.deepcopy(rng.bit_generator.state); recorder=RecordingRNG(rng)
        batch=replay_batch(block,recorder,a.batch,args['context'],args['unroll'],'cpu')
        batches.append(batch)
        selection.append({'block':bi,'rng_before':before,'rng_after':copy.deepcopy(rng.bit_generator.state),'index_positions':recorder.calls[0].tolist(),'time_agent_indices':block['indices'][recorder.calls[0]].tolist(),'tensor_sha256':[tensor_sha(t) for t in batch]})
    frozen={'seed':a.seed,'batch':a.batch,'count':a.count,'block_selection':'round-robin retained order; descriptive stratified probe, not uniform replay estimate','state':manifest,'source_files':source,'selections':selection,'diagnostic_sha256':sha(__file__),'weights':a.weights,'torch':torch.__version__}
    (a.output/'frozen.json').write_text(json.dumps(frozen,indent=2)+'\n')
    model=MuZero(Config(observations=64*args['context']+args['context']-1,actions=153,hidden=args['hidden'],support=10,simulations=args['simulations'],discount=args['discount'],exploration_fraction=args['noise_fraction']))
    model.load_state_dict(state[a.weights]); model.eval()
    before={k:tensor_sha(v) for k,v in model.state_dict().items()}
    del state
    results=[measure(model,b) for b in batches]
    assert before=={k:tensor_sha(v) for k,v in model.state_dict().items()}
    (a.output/'results.json').write_text(json.dumps({'frozen_sha256':sha(a.output/'frozen.json'),'weights_sha256':before,'results':results,'scope':'Descriptive CPU gradients only; no strength or optimization claim'},indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--state',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--source-manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch',type=int,default=16);p.add_argument('--count',type=int,default=3);p.add_argument('--seed',type=int,required=True);p.add_argument('--weights',choices=['model','initial'],default='model');p.add_argument('--max-seconds',type=int,default=180)
    a=p.parse_args()
    if not 1<=a.max_seconds<=600: p.error('max-seconds must be 1..600')
    def timeout(*_): raise TimeoutError('Diagnostic deadline reached')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(a.max_seconds)
    run(a)
