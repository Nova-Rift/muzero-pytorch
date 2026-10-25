"""No-update, no-environment initialization gate; run in the frozen CUDA runtime.

This tool imports the immutable trainer's model factory, but never calls its main,
collector, optimizer, or search. It freezes a NEW contract, never edits a template.
"""
import argparse
import ast
from dataclasses import asdict
import importlib
import io
from types import SimpleNamespace
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import replication_contract as rc
from evaluate_endpoint import bind_source

RUNTIME = {'torch':'2.8.0+cu128', 'python':'3.12.3', 'parameters':3226691}


def now():
    return datetime.now(timezone.utc).isoformat()


def timestamp(value):
    t=datetime.fromisoformat(value)
    rc.require(t.tzinfo is not None, 'Timezone required')
    return t.timestamp()


def validate_template(template):
    value=copy.deepcopy(template)
    rc.require(value['initial_sha256'] is None, 'Initialization requires unbound template')
    value['initial_sha256']='0'*64
    rc.validate_contract(value)


def validate_artifact(contract, receipt, initial, template_sha256=None):
    rc.validate_contract(contract)
    rc.require(receipt.get('schema')==1 and receipt.get('kind')=='no-update-initialization-v1', 'Wrong initialization receipt')
    rc.require(receipt.get('runtime')==RUNTIME, 'Initialization runtime differs')
    rc.require(receipt.get('source')==contract['source'] and receipt.get('config')==contract['config'] and
               receipt.get('recipe')==contract['recipe'], 'Initialization source/config/recipe differs')
    rc.require(receipt.get('training_seed')==rc.TRAINING_SEED, 'Initialization seed differs')
    rc.require(rc.exact(receipt.get('counters'),dict(games=0,optimizer_updates=0,training_examples=0)), 'Initialization performed training')
    rc.require(receipt.get('initial_sha256')==contract['initial_sha256']==rc.digest(initial), 'Initialization bytes differ')
    rc.require(receipt.get('contract_sha256')==hashlib.sha256((json.dumps(contract,indent=2)+'\n').encode()).hexdigest(), 'Frozen contract differs')
    rc.require(timestamp(receipt['started_at'])<=timestamp(receipt['completed_at'])<=datetime.now(timezone.utc).timestamp()+5, 'Invalid initialization timestamps')
    if template_sha256 is not None:rc.require(receipt.get('template_sha256')==template_sha256,'Template binding differs')
    return receipt


# Reviewed statements from frozen train_loss.py main, before optimizer/Pool/search.
# Only these AST-equal statements may execute; main itself is never invoked.
PREFIX = """torch.set_num_threads(1);torch.manual_seed(a.seed);rng=np.random.default_rng(a.seed)
torch.backends.cuda.matmul.allow_tf32=a.tf32
model=model_class(a.loss_contract)(Config(observations=64*a.context+a.context-1,actions=153,hidden=a.hidden,simulations=a.simulations,discount=a.discount,support=10,exploration_fraction=a.noise_fraction)).to(a.device)
"""
EXPORT = """initial=restored['initial'] if restored else {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
np.savez_compressed(out/'initial.npz',**{k:v.numpy() for k,v in initial.items()})
"""


def audited_statements(source):
    tree=ast.parse((source/'arena/train_loss.py').read_text())
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    expected=ast.parse(PREFIX).body
    matches=[main.body[i:i+len(expected)] for i in range(len(main.body))
             if [ast.dump(n) for n in main.body[i:i+len(expected)]]==[ast.dump(n) for n in expected]]
    rc.require(len(matches)==1, 'Frozen trainer initialization sequence differs')
    export=ast.parse(EXPORT).body
    matches_export=[n.body[:2] for n in main.body if isinstance(n,ast.Try)
                    and [ast.dump(x) for x in n.body[:2]]==[ast.dump(x) for x in export]]
    rc.require(len(matches_export)==1, 'Frozen trainer initial export sequence differs')
    return ast.Module(body=matches[0],type_ignores=[])


def frozen_factory(source):
    """Reject cached imports from another checkout instead of silently mixing code."""
    source=source.resolve()
    for name,module in list(sys.modules.items()):
        if name=='model' or name=='arena' or name.startswith('arena.'):
            path=getattr(module,'__file__',None)
            if path is not None:
                rc.require(Path(path).resolve().is_relative_to(source), 'Foreign cached trainer module: '+name)
            else:
                locations=list(getattr(module,'__path__',[]))
                rc.require(locations and all(Path(p).resolve().is_relative_to(source) for p in locations),
                           'Foreign cached trainer namespace: '+name)
    sys.path.insert(0,str(source))
    model=importlib.import_module('model')
    factory=importlib.import_module('arena.rootweighted')
    rc.require(Path(model.__file__).resolve()==source/'model.py' and
               Path(factory.__file__).resolve()==source/'arena/rootweighted.py', 'Factory import escaped frozen source')
    return model.Config,factory.model_class


def initialize_and_check(source, device):
    """CPU is diagnostic only. Production caller enforces frozen CUDA runtime.

    Compare our constructor with the exact audited trainer AST prefix, including
    post-construction RNG state. Never construct optimizer, Pool or Search.
    """
    import torch
    import numpy as np
    prefix=audited_statements(source)
    Config,model_class=frozen_factory(source)
    def snapshot(model,rng):
        return ({k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                torch.get_rng_state().clone(),copy.deepcopy(rng.bit_generator.state),
                [v.clone() for v in torch.cuda.get_rng_state_all()] if device=='cuda' else [])
    torch.set_num_threads(1);torch.manual_seed(rc.TRAINING_SEED)
    rng=np.random.default_rng(rc.TRAINING_SEED)
    torch.backends.cuda.matmul.allow_tf32=rc.RECIPE['tf32']
    model=model_class(rc.RECIPE['loss_contract'])(Config(**rc.CONFIG)).to(device)
    rc.require(asdict(model.config)==rc.CONFIG, 'Model config differs')
    rc.require(sum(p.numel() for p in model.parameters())==RUNTIME['parameters'],'Parameter count differs')
    state,cpu_rng,numpy_rng,cuda_rng=snapshot(model,rng)
    scope=dict(torch=torch,np=np,Config=Config,model_class=model_class,
               a=SimpleNamespace(**{**rc.RECIPE,'device':device}))
    exec(compile(prefix,str(source/'arena/train_loss.py'),'exec'),scope)
    reference,ref_cpu,ref_numpy,ref_cuda=snapshot(scope['model'],scope['rng'])
    rc.require(list(state)==list(reference), 'Initial tensor keys/order differ')
    for key,value in state.items():
        other=reference[key]
        rc.require(value.dtype==other.dtype and value.shape==other.shape and
                   value.numpy().tobytes()==other.numpy().tobytes(), 'Initial tensor differs: '+key)
    rc.require(torch.equal(cpu_rng,ref_cpu) and numpy_rng==ref_numpy and
               len(cuda_rng)==len(ref_cuda) and all(torch.equal(a,b) for a,b in zip(cuda_rng,ref_cuda)),
               'Post-initialization RNG differs')
    def encoded(values):
        stream=io.BytesIO();np.savez_compressed(stream,**{k:v.numpy() for k,v in values.items()})
        return stream.getvalue()
    payload=encoded(state)
    rc.require(payload==encoded(reference), 'Initial NPZ export bytes differ')
    return payload,dict(schema=1,kind='frozen-trainer-initial-prefix-parity-v1',
        device=device,torch=str(torch.__version__),python=platform.python_version(),
        source_commit=rc.SOURCE_COMMIT,generator_sha256=rc.digest(Path(__file__)),
        trainer_sha256=rc.digest(source/'arena/train_loss.py'),
        initial_sha256=hashlib.sha256(payload).hexdigest(),tensor_count=len(state),
        tensor_bytes_equal=True,npz_bytes_equal=True,rng_states_equal=True,
        trainer_main_invoked=False,actual_trainer_export_verified=False,
        limitation='Exact audited prefix only; compare actual trainer initial.npz before accepting endpoints.')


def generate(template_path, expected_template, source, output):
    rc.require(rc.digest(template_path)==expected_template, 'Predeclared template digest differs')
    template=json.loads(template_path.read_text());validate_template(template)
    bind_source({'source_provenance':template['source'],'source_commit':rc.SOURCE_COMMIT},source,('replication',))
    started=now()
    import torch
    rc.require(str(torch.__version__)==RUNTIME['torch'] and platform.python_version()==RUNTIME['python'], 'Frozen CUDA/Python runtime required')
    rc.require(torch.cuda.is_available(), 'CUDA unavailable; CPU hash cannot freeze this gate')
    payload,parity=initialize_and_check(source,'cuda')
    output.mkdir(parents=True,exist_ok=False)
    (output/'initial.npz').write_bytes(payload)
    (output/'initial-parity.json').write_text(json.dumps(parity,indent=2)+'\n')
    contract=copy.deepcopy(template);contract['initial_sha256']=rc.digest(output/'initial.npz')
    rc.validate_contract(contract)
    (output/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    receipt=dict(schema=1,kind='no-update-initialization-v1',started_at=started,completed_at=now(),
                 template_sha256=expected_template,contract_sha256=rc.digest(output/'contract.json'),
                 initial_sha256=contract['initial_sha256'],source=template['source'],config=rc.CONFIG,
                 recipe=rc.RECIPE,training_seed=rc.TRAINING_SEED,runtime=RUNTIME,
                 counters=dict(games=0,optimizer_updates=0,training_examples=0),
                 generator_sha256=rc.digest(Path(__file__)),gpu=torch.cuda.get_device_name())
    validate_artifact(contract,receipt,output/'initial.npz',expected_template)
    (output/'initialization.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('template','source','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--template-sha256',required=True);a=p.parse_args()
    generate(a.template,a.template_sha256,a.source,a.output)

if __name__=='__main__':main()
