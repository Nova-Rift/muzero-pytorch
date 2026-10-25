"""Fresh-seed collection-only prerequisite and in-process first-update guard.

Never changes the frozen trainer. Collection-only exits through its finally block.
Receipts are evidence, not admission, selection, or playing-strength decisions.
"""
import argparse
import importlib
import inspect
import json
import re
from pathlib import Path
import sys
import numpy as np
import replication_contract as rc
import replication_initialization as init
from evaluate_endpoint import bind_source

TIMINGS = {'search', 'environment', 'history'}
RECEIPT = 'replication-collection.json'
FILES = ('initial.npz', 'manifest.json', 'full-block.npz', 'collection-stats.json')


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def is_digest(value):
    return isinstance(value,str) and re.fullmatch(r'[0-9a-f]{64}',value) is not None


def pinned(path, sha):
    rc.require(is_digest(sha) and rc.digest(path)==sha, 'Predeclared input hash differs: '+Path(path).name)
    return json.loads(Path(path).read_text())


def validate_manifest(directory, contract, source_receipt, side, initialization, require_output_path=False):
    rc.validate_contract(contract)
    rc.require(side in ('paired','candidate'), 'Unknown replication side')
    rc.require(source_receipt.get('dirty') is False and rc.exact(
        {k:source_receipt.get(k) for k in ('commit','files')}, contract['source']), 'Source receipt differs')
    m=json.loads((directory/'manifest.json').read_text());a=m.get('arguments',{})
    recipe={**rc.RECIPE, **(rc.VARIANTS[contract['variant']] if side=='candidate' else {})}
    rc.require(set(a)==set(recipe)|rc.OPERATIONAL and rc.exact(
        {k:v for k,v in a.items() if k not in rc.OPERATIONAL},recipe), 'Training recipe/seed/intervention differs')
    rc.require(all(isinstance(a[k],str) and a[k] for k in ('site','output')) and
        (not require_output_path or Path(a['output']).resolve()==directory.resolve()) and type(a['max_seconds']) in (int,float) and
        0<a['max_seconds']<=850, 'Operational arguments differ')
    rc.require(m.get('source_commit')==rc.SOURCE_COMMIT and rc.exact(m.get('source_provenance'),source_receipt), 'Trainer source differs')
    rc.require(rc.exact(m.get('config'),rc.CONFIG) and rc.exact(m.get('environment'),rc.ENVIRONMENT), 'Config/environment differs')
    rc.require(rc.exact({k:m.get(k) for k in init.RUNTIME},init.RUNTIME), 'Trainer runtime differs')
    rc.require(isinstance(m.get('gpu'),str) and m['gpu'] and m['gpu']==initialization.get('gpu'), 'Initialization GPU differs')
    rc.require(rc.digest(directory/'initial.npz')==contract['initial_sha256'], 'Actual trainer initial export differs')
    return m


def bindings(initialization_dir, contract_sha, initialization_sha, review_path, review_sha, source_receipt):
    contract=pinned(initialization_dir/'contract.json',contract_sha)
    receipt=pinned(initialization_dir/'initialization.json',initialization_sha)
    init.validate_artifact(contract,receipt,initialization_dir/'initial.npz')
    rc.require(receipt.get('generator_sha256')==rc.digest(Path(init.__file__)), 'Initializer implementation differs')
    parity=json.loads((initialization_dir/'initial-parity.json').read_text())
    rc.require(parity.get('schema')==1 and parity.get('kind')=='frozen-trainer-initial-prefix-parity-v1' and
        parity.get('trainer_main_invoked') is False and parity.get('actual_trainer_export_verified') is False and
        type(parity.get('tensor_count')) is int and parity['tensor_count']==20 and parity.get('device')=='cuda' and parity.get('source_commit')==rc.SOURCE_COMMIT and
        parity.get('generator_sha256')==receipt['generator_sha256'] and
        parity.get('trainer_sha256')==contract['source']['files']['arena/train_loss.py'] and
        parity.get('initial_sha256')==contract['initial_sha256'] and
        parity.get('torch')==init.RUNTIME['torch'] and parity.get('python')==init.RUNTIME['python'] and
        all(parity.get(k) is True for k in ('tensor_bytes_equal','npz_bytes_equal','rng_states_equal')),
        'Frozen CUDA prefix parity missing or different')
    review=pinned(review_path,review_sha)
    expected=dict(schema=1,kind='replication-initialization-review-v1',accepted=True,
        contract_sha256=contract_sha,initialization_sha256=initialization_sha,
        parity_sha256=rc.digest(initialization_dir/'initial-parity.json'))
    rc.require(set(review)==set(expected)|{'completed_at'} and all(rc.exact(review[k],v) for k,v in expected.items()), 'Initialization review bindings differ')
    rc.require(init.timestamp(receipt['completed_at'])<=init.timestamp(review['completed_at'])<=init.timestamp(init.now()), 'Review must follow initialization and precede trainer')
    rc.require(source_receipt.get('dirty') is False and rc.exact({k:source_receipt.get(k) for k in ('commit','files')},contract['source']), 'Frozen source receipt differs')
    return contract,receipt,dict(contract_sha256=contract_sha,initialization_sha256=initialization_sha,
        initialization_review_sha256=review_sha,initialization_review_completed_at=review['completed_at'])


def validate_block(block, stats):
    """Require the complete frozen collector schema, not the old two-game trace."""
    rc.require(type(stats.get('games')) is int and stats['games']==128, 'Full128 games required')
    rc.require(set(stats)=={'collection_steps','padded_search_roots','action_counts','hits','movement_pixels','completed_search_return_mse','games','natural_terminal','cutoffs','joint_steps','active_transitions'}|TIMINGS, 'Collector aggregate fields differ')
    n=stats.get('collection_steps');rc.require(type(n) is int and 1<=n<=256,'Invalid collection length')
    shapes={'observation':((n+1,1408,64),'float32'), 'actions':((n,1408),'int64'),
        'rewards':((n,1408),'float32'), 'policies':((n+1,1408,153),'float32'),
        'targets':((n+1,1408),'float32'), 'valid':((n+1,1408),'bool'),
        'absorbing':((n+1,1408),'bool')}
    rc.require(set(block)==set(shapes)|{'indices'},'Incomplete collector arrays')
    for key,(shape,dtype) in shapes.items():
        v=block[key];rc.require(isinstance(v,np.ndarray) and v.shape==shape and str(v.dtype)==dtype and np.isfinite(v).all(),'Collector array differs: '+key)
    indices=block['indices'];expected=np.argwhere(block['valid'][:n])
    rc.require(indices.dtype==expected.dtype and indices.shape==expected.shape and np.array_equal(indices,expected),'Collector replay indices differ')
    rc.require(all(type(stats[k]) is int and stats[k]>=0 for k in ('natural_terminal','cutoffs','active_transitions','joint_steps','padded_search_roots','hits')), 'Invalid collection counters')
    rc.require(np.all((block['actions']>=0)&(block['actions']<153)), 'Invalid action index')
    rc.require(stats['natural_terminal']+stats['cutoffs']==128 and stats['active_transitions']==int(block['valid'].sum()),'Collector aggregates inconsistent')
    rc.require(all(type(v) is int and v>=0 for v in stats['action_counts']), 'Invalid action counts')
    rc.require(len(stats['action_counts'])==153 and sum(stats['action_counts'])==int(block['valid'][:n].sum()), 'Action aggregates inconsistent')
    return {k:v for k,v in stats.items() if k not in TIMINGS}


def equal_blocks(actual, expected):
    rc.require(set(actual)==set(expected), 'Full block keys differ')
    for key,x in actual.items():
        y=expected[key]
        rc.require(x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes(), 'Full128 collection differs: '+key)


def reference(directory, sha, contract, source_receipt, side, initialization, bound):
    rc.require(isinstance(directory,Path) and isinstance(sha,str), 'Pinned reference directory and receipt required')
    r=pinned(directory/RECEIPT,sha)
    rc.require(r.get('kind')=='replication-collection-v1' and type(r.get('schema')) is int and r['schema']==1 and
        r.get('mode')=='reference' and r.get('side')==side and r.get('variant')==contract['variant'] and
        r.get('complete') is True and type(r.get('optimizer_updates')) is int and r['optimizer_updates']==0 and
        r.get('checked_before_first_collect') is True and r.get('checked_before_first_update') is True and
        r.get('guard_sha256')==rc.digest(Path(__file__)) and rc.exact(r.get('bindings'),bound), 'Collection reference binding differs')
    rc.require(init.timestamp(bound['initialization_review_completed_at'])<=init.timestamp(r['started_at'])<=init.timestamp(r['completed_at'])<=init.timestamp(init.now()), 'Collection chronology differs')
    paired = side=='candidate' and contract['variant']=='MIR'
    rc.require(rc.exact(r.get('comparisons'),{'paired':dict(full_128_trajectory_parity=True,non_timing_aggregates_equal=True)} if paired else {}), 'Reference parity semantics differ')
    rc.require(r.get('paired_parity')==parity_semantics(side,contract), 'Reference policy semantics differ')
    rc.require(r.get('own_reference_sha256') is None and (is_digest(r.get('paired_reference_sha256')) if side=='candidate' else r.get('paired_reference_sha256') is None), 'Reference chain binding differs')
    rc.require(set(r.get('files',{}))==set(FILES),'Incomplete collection receipt')
    for name in FILES:rc.require(rc.digest(directory/name)==r['files'][name],'Collection evidence changed: '+name)
    validate_manifest(directory,contract,source_receipt,side,initialization)
    with np.load(directory/'full-block.npz',allow_pickle=False) as z:
        validate_block(dict(z),json.loads((directory/'collection-stats.json').read_text()))
    return r


def parity_semantics(side,contract):
    return ('required-exact' if contract['variant']=='MIR' else 'inapplicable-relative-aim-changes-policy') if side=='candidate' else 'paired-reference'


class CollectionComplete(Exception):
    """Only a verified reference can exit before the trainer's first update."""


def install_gate(trainer, source, output, contract, source_receipt, side, initialization, bound,
                 mode, own_reference=None, own_sha=None, paired_reference=None, paired_sha=None):
    rc.require(mode in ('reference','train'), 'Unknown guard mode')
    rc.require(not output.exists(), 'Fresh trainer output required')
    bind_source({'source_provenance':source_receipt,'source_commit':rc.SOURCE_COMMIT},source,('replication',))
    if mode=='train':reference(own_reference,own_sha,contract,source_receipt,side,initialization,bound)
    if side=='candidate':reference(paired_reference,paired_sha,contract,source_receipt,'paired',initialization,bound)
    started=init.now()
    rc.require(init.timestamp(bound['initialization_review_completed_at'])<=init.timestamp(started), 'Initialization review must precede trainer')
    original=trainer.collect;signature=inspect.signature(original);checked=[]
    def collect(*args,**kwargs):
        if checked:return original(*args,**kwargs)
        validate_manifest(output,contract,source_receipt,side,initialization,require_output_path=True)
        call=signature.bind(*args,**kwargs);call.apply_defaults();a=call.arguments
        rc.require(list(a['seeds'])==list(range(rc.TRAINING_SEED,rc.TRAINING_SEED+128)), 'Actual collection seeds differ')
        recipe={**rc.RECIPE,**(rc.VARIANTS[contract['variant']] if side=='candidate' else {})}
        rc.require(all(rc.exact(a[k],recipe[k]) for k in ('limit','context','device_history','return_mode','timeout_bootstrap','aim_decoding')), 'Actual collection flags differ')
        block,stats=original(*args,**kwargs);aggregates=validate_block(block,stats)
        comparisons={}
        for label,directory in ([('own',own_reference)] if mode=='train' else [])+(
                [('paired',paired_reference)] if side=='candidate' and contract['variant']=='MIR' else []):
            with np.load(directory/'full-block.npz',allow_pickle=False) as z:equal_blocks(block,z)
            other=json.loads((directory/'collection-stats.json').read_text())
            rc.require(rc.exact(aggregates,{k:v for k,v in other.items() if k not in TIMINGS}), 'Non-timing aggregates differ')
            comparisons[label]=dict(full_128_trajectory_parity=True,non_timing_aggregates_equal=True)
        np.savez_compressed(output/'full-block.npz',**block);write(output/'collection-stats.json',stats)
        result=dict(schema=1,kind='replication-collection-v1',mode=mode,side=side,variant=contract['variant'],
            complete=True,optimizer_updates=0,started_at=started,completed_at=init.now(),checked_before_first_collect=True,checked_before_first_update=True,
            guard_sha256=rc.digest(Path(__file__)),bindings=bound,comparisons=comparisons,
            paired_parity=parity_semantics(side,contract),
            own_reference_sha256=own_sha,paired_reference_sha256=paired_sha,
            files={name:rc.digest(output/name) for name in FILES})
        write(output/RECEIPT,result);checked.append(result)
        if mode=='reference':raise CollectionComplete()
        return block,stats
    trainer.collect=collect
    return checked


def trainer_command(contract,side,site,output,seconds):
    rc.validate_contract(contract);rc.require(side in ('paired','candidate'),'Unknown side')
    rc.require(0<seconds<=850,'Trainer deadline outside bound')
    recipe={**rc.RECIPE,**(rc.VARIANTS[contract['variant']] if side=='candidate' else {}),
        'site':str(site),'output':str(output),'max_seconds':seconds}
    result=[]
    for key,value in recipe.items():
        if value is None or value is False:continue
        result.append('--'+key.replace('_','-'))
        if value is not True:result.append(str(value))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','site','output','initialization-dir','source-receipt','initialization-review'):
        p.add_argument('--'+key,type=Path,required=True)
    for key in ('contract','initialization','source-receipt','initialization-review'):p.add_argument('--'+key+'-sha256',required=True)
    p.add_argument('--side',choices=('paired','candidate'),required=True);p.add_argument('--mode',choices=('reference','train'),required=True)
    for key in ('own-reference','paired-reference'):
        p.add_argument('--'+key,type=Path);p.add_argument('--'+key+'-sha256')
    p.add_argument('--max-seconds',type=float,required=True);a=p.parse_args()
    source_receipt=pinned(a.source_receipt,a.source_receipt_sha256)
    contract,initialization,bound=bindings(a.initialization_dir,a.contract_sha256,a.initialization_sha256,
        a.initialization_review,a.initialization_review_sha256,source_receipt)
    source=a.source.resolve();bind_source({'source_provenance':source_receipt,'source_commit':rc.SOURCE_COMMIT},source,('replication',))
    for name,sha in rc.ENVIRONMENT.items():rc.require(rc.digest(a.site/name)==sha,'Environment source differs')
    init.frozen_factory(source)
    trainer=importlib.import_module('arena.train_loss')
    rc.require(Path(trainer.__file__).resolve()==source/'arena/train_loss.py','Wrong trainer import')
    checked=install_gate(trainer,source,a.output,contract,source_receipt,a.side,initialization,bound,a.mode,
        a.own_reference,a.own_reference_sha256,a.paired_reference,a.paired_reference_sha256)
    sys.argv=[str(trainer.__file__),*trainer_command(contract,a.side,a.site,a.output,a.max_seconds)]
    try:trainer.main()
    except CollectionComplete:
        rc.require(a.mode=='reference' and len(checked)==1,'Unverified collection-only exit')
    rc.require(len(checked)==1,'No successful collection gate')

if __name__=='__main__':main()
