"""No-update, no-environment initialization gate; run in the frozen CUDA runtime.

This tool imports the immutable trainer's model factory, but never calls its main,
collector, optimizer, or search. It freezes a NEW contract, never edits a template.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
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


def generate(template_path, expected_template, source, output):
    rc.require(rc.digest(template_path)==expected_template, 'Predeclared template digest differs')
    template=json.loads(template_path.read_text());validate_template(template)
    bind_source({'source_provenance':template['source'],'source_commit':rc.SOURCE_COMMIT},source,('replication',))
    started=now()
    # Source path wins over the adapter checkout for all trainer dependencies.
    sys.path.insert(0,str(source.resolve()))
    import torch
    import numpy as np
    from arena.train_loss import model_class, Config
    rc.require(str(torch.__version__)==RUNTIME['torch'] and platform.python_version()==RUNTIME['python'], 'Frozen CUDA/Python runtime required')
    rc.require(torch.cuda.is_available(), 'CUDA unavailable; CPU hash cannot freeze this gate')
    torch.set_num_threads(1);torch.manual_seed(rc.TRAINING_SEED)
    torch.backends.cuda.matmul.allow_tf32=True
    model=model_class('baseline')(Config(**rc.CONFIG)).to('cuda')
    rc.require(sum(p.numel() for p in model.parameters())==RUNTIME['parameters'],'Parameter count differs')
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'initial.npz',**{k:v.numpy() for k,v in state.items()})
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
