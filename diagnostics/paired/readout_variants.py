"""Frozen EGO/MIR development audit. Inputs are admitted contracts, never inferred permissions.
No training, confirmation execution, promotion, or ranking is performed.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'retention'))
from evaluate_endpoint import (validate_source_contract, compare_arguments,
                               OPERATIONAL_ARGUMENTS, declared_record)
from readout import audit_bout, require, digest, load
from readout_3arm import paired, gate, OPP_SHA, SAMPLING

SEEDS = {'E1024': 95010000, 'F1024': 95020000, 'OwnInitial': 95030000}
VARIANTS = {'EGO2048': ('aim_decoding', 'relative', 'absolute'),
            'MIR2048': ('mirror_augment', True, False)}


def training(directory, base, contract, name):
    m, r = load(directory/'manifest.json'), load(directory/'report.json')
    key, candidate, original = VARIANTS[name]
    require(contract['arguments'] == {key: {'candidate': candidate, 'paired': original}}, 'wrong declared variant')
    validate_source_contract(m, base, (key,), contract)
    compare_arguments(m['arguments'], base['arguments'], OPERATIONAL_ARGUMENTS, (key,))
    for field in ('config','environment','torch','python','parameters'):
        require(m.get(field) == base.get(field) and m.get(field) is not None, 'recipe differs: '+field)
    for record in (m,base):
        require(record['source_provenance'].get('dirty') is False and record['source_provenance']['commit']==record['source_commit'], 'dirty/mismatched source provenance')
    endpoint(directory, m)
    return m, r


def endpoint(directory, m):
    r=load(directory/'report.json')
    require(r.get('requested_run_completed') is True, 'timeout/incomplete training')
    for key in ('games','optimizer_updates'):
        require(type(r.get(key)) is int and r[key]==2048, 'wrong endpoint '+key)
    require(r.get('checkpoint_sha256') == digest(directory/'weights.npz'), 'wrong final weights')
    require(m['arguments']['games']==2048 and m['arguments']['updates_per_game']==1, 'wrong training count')


def bouts(root, name, directory, manifest, base_directory, source, opponents, contract=None):
    out={}
    for opponent, seed in SEEDS.items():
        folder=root/f'{name}-{opponent}'
        catalog=load(folder/'catalog.json'); protocol=load(folder/'results/protocol.json')
        require(protocol['catalog_sha256']==digest(folder/'catalog.json'), 'catalog hash differs')
        require(protocol['models']==[{k:v for k,v in m.items() if k!='checkpoint'} for m in catalog['models']], 'catalog models differ')
        require(protocol['seed']==seed and protocol['sampling']==SAMPLING, 'wrong frozen seed/sampling')
        require(protocol['games']==64 and protocol['limit']==256, 'wrong games/limit')
        for key,file in [('source_sha256','tournament/run.py'),('controller_sha256','tournament/controllers.py'),('lana_batch_adapter_sha256','tournament/lana_batch.py')]:
            require(protocol[key]==source['files'][file], 'wrong evaluator source: '+key)
        require(protocol['environment']==manifest['environment'], 'wrong environment')
        own,opp=protocol['models']
        expected=dict(id=name, family='independent_muzero', sha256=digest(directory/'weights.npz'), config=manifest['config'],root_candidates=8,trained=True)
        if manifest['arguments'].get('aim_decoding','absolute')=='relative':expected['aim_decoding']='relative'
        require(own==expected, 'wrong arm weights/config/declared flags')
        require(opp=={k:v for k,v in opponents[opponent].items() if k!='checkpoint'}, 'opponent differs from frozen catalog input')
        require(opp['id']==opponent and opp['sha256']==(OPP_SHA[opponent] if opponent in OPP_SHA else digest(base_directory/'initial.npz')), 'wrong frozen opponent')
        require(opp.get('aim_decoding','absolute')=='absolute', 'opponent aim changed')
        if opponent in OPP_SHA:require(opp['config']['discount']==.99, 'wrong opponent discount')
        if contract:
            require(catalog.get('source_contract')==contract, 'catalog source contract differs')
            for key,value in declared_record(manifest,load(base_directory/'manifest.json'),tuple(contract['arguments'])).items():
                require(catalog.get(key)==value, 'catalog declaration differs: '+key)
        else:
            require(set(catalog)=={'models'}, 'undeclared baseline catalog metadata')
        out[opponent]=audit_bout(folder/'results',protocol)
        out[opponent]['directory']=str(folder/'results')
    return out


def analyze(root, name, training_dir, baseline_root, baseline_dir, contract, source, opponents):
    require(name in VARIANTS,'unsupported arm')
    base=load(baseline_dir/'manifest.json'); endpoint(baseline_dir,base)
    m,_=training(training_dir,base,contract,name)
    require(source.get('dirty') is False and source['commit']==contract['candidate']['commit'] and source['files']==contract['candidate']['files'], 'wrong evaluator source receipt')
    require(digest(training_dir/'initial.npz')==digest(baseline_dir/'initial.npz'), 'initial weights differ')
    b=bouts(baseline_root,'Baseline2048',baseline_dir,base,baseline_dir,source,opponents)
    c=bouts(root,name,training_dir,m,baseline_dir,source,opponents,contract)
    common=None
    for records in (b,c):
        for opponent,bout in records.items():
            p=bout['protocol'];signature={k:v for k,v in p.items() if k not in ('models','seed','catalog_sha256')}
            require(common is None or common==signature,'evaluator/environment/runtime differs')
            common=signature
            require(p['models'][1]==b[opponent]['protocol']['models'][1],'opponent source/config differs')
    per=paired(c,b,95050000)
    return dict(schema=1,arm=name,source_contract=contract,seeds=SEEDS,bootstrap_seed=95050000,
                per_opponent=per,gate=gate(per),own_initial=dict(baseline=b['OwnInitial']['counts'],arm=c['OwnInitial']['counts'],input_sha256=dict(baseline=b['OwnInitial']['input_sha256'],arm=c['OwnInitial']['input_sha256'])),
                scope='Development only; one training seed. Confirmation and independent training replication remain conditional. No promotion or ladder result.',
                catalog_sha256={str(root_/f'{name_}-{o}'/'catalog.json'):digest(root_/f'{name_}-{o}'/'catalog.json') for root_,name_ in ((root,name),(baseline_root,'Baseline2048')) for o in SEEDS},
                input_sha256={str(p):digest(p) for d in (training_dir,baseline_dir) for p in (d/'manifest.json',d/'report.json',d/'weights.npz',d/'initial.npz')})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','training','baseline-root','baseline-training','contract','source-receipt','opponents','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--arm',choices=VARIANTS,required=True)
    a=p.parse_args()
    try:
        result=analyze(a.root,a.arm,a.training,a.baseline_root,a.baseline_training,load(a.contract),load(a.source_receipt),{o:load(a.opponents/(o+'.json')) for o in SEEDS})
        result['contract_sha256']=digest(a.contract);result['source_receipt_sha256']=digest(a.source_receipt)
        a.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result['gate']))
    except (ValueError,KeyError,TypeError,OSError) as e:
        print('READOUT FAILED: '+str(e),file=sys.stderr);return 2
    return 0

if __name__=='__main__':sys.exit(main())
