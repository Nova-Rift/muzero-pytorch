"""Separate, fail-closed independent-seed restart contract. No job admission or training.

The coordinator freezes a concrete contract before either full training run. A template
without the independently established initialization SHA is intentionally invalid.
Historical evaluate_endpoint restart semantics are unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

from evaluate_endpoint import bind_source

MODE = 'independent-seed-restart-v1'
TRAINING_SEED = 73142017
SOURCE_COMMIT = '14fece1209067004f4c3936a6171bc40da55577d'
SOURCE_FILES_SHA256 = '5804ba906db20321884110b91467a4ed059d4ed4f0727c8717841bd84543d65e'
OPERATIONAL = frozenset(('site', 'output', 'max_seconds'))
RECIPE = dict(loss_contract='baseline', resume_state=None, games=2048, parallel=128,
              replay_blocks=3, workers=4, hidden=448, context=64, simulations=32,
              batch=256, unroll=5, return_mode='episode', timeout_bootstrap='zero',
              root_candidates=8, cadence='homepage', agents=2, policy_floor=0.0,
              noise_fraction=0.25, discount=1.0, limit=256, updates_per_game=1,
              seed=TRAINING_SEED, device='cuda', device_history=True, cuda_graph=True,
              tf32=True, aim_decoding='absolute', mirror_augment=False)
CONFIG = dict(observations=4159, actions=153, hidden=448, support=10, discount=1.0,
              simulations=32, pb_c_base=19652.0, pb_c_init=1.25,
              dirichlet_alpha=0.3, exploration_fraction=0.25)
ENVIRONMENT = {'src/sandboxCombat.ts':'8041163035515c13d1645ada2df72bdf482408dc9f467f220b5429da43e8685a',
               'src/sandboxMotion.ts':'bff13f8b38d4d59b40271f60fb1fda57ae5c1a752c5763cbe2a9fb383910811c',
               'src/lana/observation.ts':'1867ab292e924e5708eac81924efc4d1fe7200f294e4d20de804d0955d2a8947'}
VARIANTS = {'EGO': {'aim_decoding':'relative'}, 'MIR': {'mirror_augment':True}}


def require(value, message):
    if not value:
        raise ValueError(message)


def exact(actual, expected):
    # JSON canonical comparison distinguishes bool/int/float, including nested fields.
    return json.dumps(actual, sort_keys=True, allow_nan=False) == json.dumps(expected, sort_keys=True, allow_nan=False)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_contract(contract):
    require(isinstance(contract, dict) and set(contract) == {
        'schema','mode','variant','training_seed','source','initial_sha256','recipe','config','environment'},
        'Invalid replication contract fields')
    require(type(contract['schema']) is int and contract['schema']==1 and contract['mode']==MODE,
            'Explicit independent-seed mode required')
    require(contract['variant'] in VARIANTS, 'Only one EGO or MIR intervention is admitted')
    require(type(contract['training_seed']) is int and contract['training_seed']==TRAINING_SEED,
            'Different or unaudited replication seed')
    require(exact(contract['recipe'], RECIPE), 'Frozen replication recipe differs')
    require(exact(contract['config'], CONFIG), 'Frozen model config differs')
    require(exact(contract['environment'], ENVIRONMENT), 'Frozen reward/environment source differs')
    require(isinstance(contract['initial_sha256'],str) and
            re.fullmatch(r'[0-9a-f]{64}',contract['initial_sha256']), 'Unbound initial weights')
    source=contract['source']
    require(isinstance(source,dict) and set(source)=={'commit','files'} and
            source['commit']==SOURCE_COMMIT and isinstance(source['files'],dict) and source['files'],
            'Replication requires the exact reviewed trainer commit')
    for name, sha in source['files'].items():
        require(isinstance(name,str) and name and not Path(name).is_absolute() and
                '..' not in Path(name).parts and str(Path(name))==name and isinstance(sha,str) and
                re.fullmatch(r'[0-9a-f]{64}',sha), 'Invalid source path/hash')
    require(hashlib.sha256(json.dumps(source['files'],sort_keys=True,separators=(',',':')).encode()).hexdigest()==SOURCE_FILES_SHA256,
            'Full frozen source map differs')
    return contract


def validate_pair(candidate, paired, contract, source_receipt):
    """Check both final endpoints against a frozen complete source receipt and exact recipe.

    source_receipt must be the separately pinned full capture_source result for SOURCE_COMMIT,
    not manufactured from the candidate manifest. The CLI requires its predeclared SHA256.
    """
    validate_contract(contract)
    require(source_receipt.get('dirty') is False and
            exact({'commit':source_receipt.get('commit'),'files':source_receipt.get('files')},contract['source']),
            'Full source receipt differs from contract')
    require(Path(candidate).resolve()!=Path(paired).resolve(), 'Self pairing is forbidden')
    result={}
    for side, directory in (('candidate',candidate),('paired',paired)):
        directory=Path(directory)
        m=json.loads((directory/'manifest.json').read_text());r=json.loads((directory/'report.json').read_text())
        expected={**RECIPE, **(VARIANTS[contract['variant']] if side=='candidate' else {})}
        args=m.get('arguments',{})
        require(set(args)==set(RECIPE)|OPERATIONAL, 'Missing or unrelated training argument')
        require(exact({k:v for k,v in args.items() if k not in OPERATIONAL},expected),
                'Seed, flags, objective or recipe differs: '+side)
        require(all(isinstance(args[k],str) and args[k] for k in ('site','output')) and
                type(args['max_seconds']) in (int,float) and 0<args['max_seconds']<=960,
                'Invalid operational arguments')
        require(m.get('source_commit')==SOURCE_COMMIT and
                m.get('source_provenance',{}).get('dirty') is False and
                m['source_provenance'].get('commit')==SOURCE_COMMIT and
                exact(m['source_provenance'].get('files'),contract['source']['files']),
                'Source provenance differs: '+side)
        require(exact(m.get('config'),CONFIG) and exact(m.get('environment'),ENVIRONMENT),
                'Config or environment differs: '+side)
        require(m.get('torch')=='2.8.0+cu128' and m.get('python')=='3.12.3' and
                type(m.get('parameters')) is int and m['parameters']==3226691,
                'Frozen runtime or parameter count differs')
        require(digest(directory/'initial.npz')==contract['initial_sha256'], 'Initial weights differ: '+side)
        require(r.get('requested_run_completed') is True, 'Incomplete endpoint: '+side)
        for key,value in (('games',2048),('optimizer_updates',2048),('training_examples',2048*256)):
            require(type(r.get(key)) is int and r[key]==value, 'Endpoint counter differs: '+key)
        require(r.get('checkpoint_sha256')==digest(directory/'weights.npz'), 'Final weights differ: '+side)
        result[side]={'manifest':m,'report':r,'weights_sha256':digest(directory/'weights.npz')}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('candidate','paired','contract','source-receipt','source'):
        p.add_argument('--'+key,type=Path,required=True)
    for key in ('contract-sha256','source-receipt-sha256'):
        p.add_argument('--'+key,required=True)
    a=p.parse_args()
    require(digest(a.contract)==a.contract_sha256,'Predeclared contract digest differs')
    require(digest(a.source_receipt)==a.source_receipt_sha256,'Predeclared source receipt digest differs')
    result=validate_pair(a.candidate,a.paired,json.loads(a.contract.read_text()),json.loads(a.source_receipt.read_text()))
    bind_source(result['candidate']['manifest'],a.source,('replication',))
    print(json.dumps({'verified':True,'mode':MODE,'training_seed':TRAINING_SEED,
                      'contract_sha256':a.contract_sha256,'source_receipt_sha256':a.source_receipt_sha256,
                      'scope':'Final pair provenance and endpoint contract only; no scientific pass or admission'}))


if __name__=='__main__':
    main()
