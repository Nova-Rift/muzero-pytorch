"""Fail-closed independent-seed readout; CPU only, never selects/admit/publishes an arm.

Input is a SHA256-pinned request JSON. Each file binding is {path, sha256}; training
and evaluation directories must be recovered with their original catalogs intact.
The trusted evaluation_environment binding contains exactly gpu and legacy_sources
from the frozen site/runtime. Readout verifies evidence, not CUDA execution itself.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys

import replication_contract as rc
import replication_initialization as init
import evaluate_replication as ev
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'paired'))
from readout import audit_bout
from readout_3arm import paired

BOOTSTRAP_SEED = 121510000
AUDIT_HELPERS = {
    'readout.py': 'a7e48be1f6a2c08006273181047f9f4f07903da7969c75d6bf4387b2cefe161d',
    'readout_3arm.py': '57e6bde3ce075ad6d036bbbb5c48e124ed7ee92e195bdd8ef78b01438dd33f4d',
}


def verify_helpers():
    for name, sha in AUDIT_HELPERS.items():
        rc.require(rc.digest(Path(__file__).resolve().parents[1]/'paired'/name)==sha,
                   'Frozen audit helper differs: '+name)


def load(path):
    return json.loads(Path(path).read_text())


def bound(binding):
    rc.require(set(binding) == {'path', 'sha256'}, 'Invalid file binding')
    path = Path(binding['path'])
    rc.require(rc.digest(path) == binding['sha256'], 'Pinned file differs: ' + str(path))
    return path


def wilson_lower(wins):
    z = 1.96
    p = wins / 64
    return (p + z*z/128 - z*math.sqrt(p*(1-p)/64 + z*z/(4*64*64))) / (1+z*z/64)


def protocol(catalog, catalog_sha, phase, opponent, contract, environment):
    files = contract['source']['files']
    return dict(catalog_sha256=catalog_sha,
                models=[{k:v for k,v in m.items() if k != 'checkpoint'} for m in catalog['models']],
                games=64, parallel=64, lana_microbatch=32, workers=4, limit=256,
                seed=ev.SEEDS[phase][opponent], device='cuda',
                source_sha256=files['tournament/run.py'], controller_sha256=files['tournament/controllers.py'],
                lana_batch_adapter_sha256=files['tournament/lana_batch.py'], environment=rc.ENVIRONMENT,
                cadence='five .05s frames per decision', sampling='independent PCG64 per game and seat; no root noise',
                scope='two active agents, fixed spawn geometry; seeds vary policy sampling, not worlds',
                runtime={'torch':init.RUNTIME['torch'], 'gpu':environment['gpu']},
                legacy_sources=environment['legacy_sources'])


def audit_evaluation(directory, side, phase, opponent, endpoints, contract, initialization,
                     environment, bindings, opponent_entry):
    directory = Path(directory)
    cat = load(directory/'catalog.json')
    rc.require(set(cat) == {'models','replication_contract','side','phase','initialization'}, 'Catalog fields differ')
    rc.require(rc.exact(cat['replication_contract'],contract) and rc.exact(cat['initialization'],initialization)
               and cat['side']==side and cat['phase']==phase, 'Catalog contract/initialization/role differs')
    label = 'Replication'+(contract['variant'] if side=='candidate' else 'Baseline')+'2048'
    m = endpoints[side]['manifest']
    expected_model = dict(id=label, family='independent_muzero', sha256=endpoints[side]['weights_sha256'],
                          config=m['config'], root_candidates=8, trained=True)
    if m['arguments']['aim_decoding']=='relative': expected_model['aim_decoding']='relative'
    rc.require(len(cat['models'])==2, 'Catalog needs exactly two models')
    # Paths are operational and may point to the original remote filesystem; bytes
    # are verified from the separately bound recovered endpoints/opponents.
    actual_model = cat['models'][0]
    rc.require(isinstance(actual_model.get('checkpoint'),str) and actual_model['checkpoint'], 'Missing catalog checkpoint path')
    rc.require(rc.exact({k:v for k,v in actual_model.items() if k!='checkpoint'},expected_model), 'Catalog endpoint differs')
    rc.require(rc.exact(cat['models'][1],opponent_entry), 'Catalog frozen opponent differs')
    receipt = load(directory/'evaluation-receipt.json')
    rc.require(set(receipt)=={'schema','started_at','completed_at','catalog_sha256','contract_sha256',
                             'initialization_sha256','source_receipt_sha256','opponent_sha256',
                             'evaluator_sha256','result_sha256'}, 'Evaluation receipt fields differ')
    rc.require(type(receipt['schema']) is int and receipt['schema']==1, 'Receipt schema differs')
    for key in ('contract','initialization','source_receipt'):
        rc.require(receipt[key+'_sha256']==bindings[key]['sha256'], 'Receipt '+key+' differs')
    rc.require(receipt['opponent_sha256']==bindings['opponents'][opponent]['entry']['sha256'], 'Opponent receipt differs')
    rc.require(receipt['evaluator_sha256']==rc.digest(Path(ev.__file__)), 'Evaluator adapter source differs')
    rc.require(receipt['catalog_sha256']==rc.digest(directory/'catalog.json'), 'Catalog receipt differs')
    rc.require(init.timestamp(initialization['completed_at']) <= init.timestamp(receipt['started_at']) <=
               init.timestamp(receipt['completed_at']) <= datetime.now(timezone.utc).timestamp()+5,
               'Evaluation timestamp ordering differs')
    results = directory/'results'
    expected_files = {'protocol.json','summary.json',label+'__'+opponent+'-0.json'}
    rc.require({p.name for p in results.iterdir()}==expected_files, 'Missing/extra result files')
    rc.require(rc.exact(receipt['result_sha256'],{p.name:rc.digest(p) for p in results.iterdir()}), 'Result receipt differs')
    expected = protocol(cat, receipt['catalog_sha256'], phase, opponent, contract, environment)
    rc.require(rc.exact(load(results/'protocol.json'),expected), 'Exact evaluator protocol differs')
    bout = audit_bout(results, expected)
    bout['directory']=str(results)
    bout['evidence_sha256']={p.name:rc.digest(p) for p in (directory/'catalog.json',directory/'evaluation-receipt.json')}
    return bout


def analyze(request):
    verify_helpers()
    rc.require(set(request)=={'schema','phase','candidate','paired','contract','source_receipt','initialization',
                             'initial','template','evaluation_environment','opponents','bouts'}, 'Request fields differ')
    rc.require(type(request['schema']) is int and request['schema']==1 and request['phase'] in ev.SEEDS, 'Invalid request phase/schema')
    paths={key:bound(request[key]) for key in ('contract','source_receipt','initialization','initial','template','evaluation_environment')}
    contract=load(paths['contract']); source=load(paths['source_receipt']); initialization=load(paths['initialization'])
    template=load(paths['template']); init.validate_template(template)
    rc.require(rc.exact({**template,'initial_sha256':contract['initial_sha256']},contract), 'Template differs from frozen contract')
    init.validate_artifact(contract,initialization,paths['initial'],request['template']['sha256'])
    rc.require(type(initialization.get('schema')) is int and rc.exact(initialization.get('runtime'),init.RUNTIME),
               'Initialization schema/runtime types differ')
    rc.require(initialization.get('generator_sha256')==rc.digest(Path(init.__file__)), 'Initialization generator source differs')
    environment=load(paths['evaluation_environment'])
    rc.require(set(environment)=={'gpu','legacy_sources'} and isinstance(environment['gpu'],str) and environment['gpu'], 'Invalid runtime binding')
    rc.require(initialization.get('gpu')==environment['gpu'], 'Initialization GPU differs')
    legacy=environment['legacy_sources']
    rc.require(isinstance(legacy,dict) and legacy and all(isinstance(k,str) and k and not Path(k).is_absolute() and
        '..' not in Path(k).parts and isinstance(v,str) and re.fullmatch('[0-9a-f]{64}',v) for k,v in legacy.items()), 'Invalid legacy source binding')
    endpoints=rc.validate_pair(Path(request['candidate']),Path(request['paired']),contract,source)
    for endpoint in endpoints.values():
        rc.require(endpoint['manifest'].get('gpu')==environment['gpu'], 'Training GPU differs')
    phase=request['phase']; opponents=ev.SEEDS[phase]
    rc.require(set(request['opponents'])==set(opponents), 'Opponent set differs')
    entries={}
    for name,binding in request['opponents'].items():
        rc.require(set(binding)=={'entry','checkpoint'}, 'Opponent binding fields differ')
        entry=load(bound(binding['entry'])); weight=bound(binding['checkpoint'])
        rc.require(entry.get('id')==name and entry.get('sha256')==rc.digest(weight)==ev.OPP_SHA.get(name,contract['initial_sha256']), 'Opponent identity/bytes differ')
        # Reuse the adapter's full controller validation against restored bytes,
        # preserving its original path for exact catalog matching afterwards.
        ev.catalog(Path(request['candidate']),Path(request['paired']),'candidate',
                   {**entry,'checkpoint':str(weight)},phase,contract,source,initialization,paths['initial'])
        entries[name]=entry
    expected={(s,o) for s in ('candidate','paired') for o in opponents}
    indexed={}
    for entry in request['bouts']:
        rc.require(set(entry)=={'side','opponent','directory'}, 'Bout binding fields differ')
        key=(entry['side'],entry['opponent'])
        rc.require(key in expected and key not in indexed, 'Unknown/duplicate bout')
        indexed[key]=audit_evaluation(entry['directory'],key[0],phase,key[1],endpoints,contract,initialization,
                                      environment,request,entries[key[1]])
    rc.require(set(indexed)==expected, 'Missing bout')
    candidate={o:indexed['candidate',o] for o in opponents}; baseline={o:indexed['paired',o] for o in opponents}
    per={o:dict(candidate=candidate[o]['counts'],paired=baseline[o]['counts'],
                wins_minus_paired=candidate[o]['counts']['wins']-baseline[o]['counts']['wins'],
                denominator=64) for o in opponents}
    if phase=='development':
        checks={o:per[o]['wins_minus_paired']>0 for o in opponents}
        rule='Strictly more wins than fresh matched baseline on EACH E1024/F1024/OwnInitial; denominator64 including draws/cutoffs.'
        intervals=paired(candidate,baseline,BOOTSTRAP_SEED)
    else:
        checks={o:dict(wilson_lower_gt_half=wilson_lower(per[o]['candidate']['wins'])>.5,
                       timeout_fraction_le_point2=per[o]['candidate']['cutoffs']/64<=.2) for o in opponents}
        for o in opponents:per[o]['candidate_wilson95_lower']=wilson_lower(per[o]['candidate']['wins'])
        rule='EACH E1024/F1024 candidate Wilson95 lower>0.5 AND cutoffs/64<=0.20; baseline differences descriptive.'
        intervals=None
    passed=all(checks.values()) if phase=='development' else all(all(c.values()) for c in checks.values())
    return dict(schema=1,phase=phase,variant=contract['variant'],training_seed=rc.TRAINING_SEED,
                gate=dict(passed=passed,checks=checks,rule=rule),per_opponent=per,
                development_paired_bootstrap=intervals,
                uncertainty=dict(method='Paired adjacent two-game balanced-seat block bootstrap; 32 blocks, 10000 draws, linear percentile95',
                                 seed=BOOTSTRAP_SEED,opponent_order=['E1024','F1024'],
                                 scope='Development only; conditional on fixed weights, not training-seed variance') if intervals else None,
                input_sha256={k:v['sha256'] for k,v in request.items() if isinstance(v,dict) and 'sha256' in v},
                endpoint_sha256={s:{f:rc.digest(Path(request[s])/f) for f in ('manifest.json','report.json','initial.npz','weights.npz')} for s in endpoints},
                bout_evidence={s+'-'+o:dict(raw=indexed[s,o]['input_sha256'],adapter=indexed[s,o]['evidence_sha256']) for s,o in sorted(expected)},
                scope='One independent training seed pair, fixed opponents and geometry; no promotion, arm selection, admission or training-seed variance estimate. Receipt timestamps bind initialization before evaluation, not training chronology.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('request',type=Path);p.add_argument('--request-sha256',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();bound(dict(path=str(a.request),sha256=a.request_sha256))
    result=analyze(load(a.request));result['request_sha256']=a.request_sha256
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')

if __name__=='__main__':main()
