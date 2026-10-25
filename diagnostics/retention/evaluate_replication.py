"""Dedicated fresh-seed evaluator. No allocation or conditional arm selection."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import replication_contract as rc
from replication_initialization import validate_artifact, now, timestamp
from evaluate_endpoint import bind_source

SEEDS={'development':{'E1024':121110000,'F1024':121210000,'OwnInitial':121310000},
       'confirmation':{'E1024':123110000,'F1024':123210000}}
OPP_SHA={'E1024':'18c18151b820636ef3bf67ea52dbecd3cb47582d9ea6bf63ede36d6c2adb1532',
         'F1024':'4de08ce08adef8344d6cf8707e22a74896c355008ffff18d3b10ce21ec1dcdb7'}


def model_entry(directory,manifest,label):
    entry=dict(id=label,family='independent_muzero',checkpoint=str(directory/'weights.npz'),
               sha256=rc.digest(directory/'weights.npz'),config=manifest['config'],root_candidates=8,trained=True)
    if manifest['arguments']['aim_decoding']=='relative':entry['aim_decoding']='relative'
    return entry


def catalog(candidate, baseline, side, opponent, phase, contract, source_receipt, initialization, initial):
    pair=rc.validate_pair(candidate,baseline,contract,source_receipt)
    validate_artifact(contract,initialization,initial)
    rc.require(side in ('candidate','paired') and phase in SEEDS,'Unknown side/phase')
    identity=opponent.get('id');rc.require(identity in SEEDS[phase],'Unknown opponent')
    rc.require(opponent.get('family')=='independent_muzero' and opponent.get('root_candidates')==8 and
               opponent.get('aim_decoding','absolute')=='absolute','Opponent controller differs')
    expected=OPP_SHA.get(identity,contract['initial_sha256'])
    rc.require(opponent.get('sha256')==expected==rc.digest(opponent['checkpoint']),'Frozen opponent differs')
    if identity=='OwnInitial':
        rc.require(rc.exact(opponent.get('config'),rc.CONFIG) and opponent.get('trained') is False,'OwnInitial config differs')
    else:
        rc.require(opponent.get('config',{}).get('discount')==.99 and opponent.get('trained') is True,'Opponent config differs')
    label='Replication'+(contract['variant'] if side=='candidate' else 'Baseline')+'2048'
    directory=candidate if side=='candidate' else baseline
    return dict(models=[model_entry(directory,pair[side]['manifest'],label),opponent],
                replication_contract=contract,side=side,phase=phase,initialization=initialization)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('candidate','baseline','contract','source-receipt','initialization','initial','opponent','source','site','output'):
        p.add_argument('--'+key,type=Path,required=True)
    for key in ('contract','source-receipt','initialization','opponent'):p.add_argument('--'+key+'-sha256',required=True)
    p.add_argument('--side',choices=('candidate','paired'),required=True)
    p.add_argument('--phase',choices=SEEDS,required=True)
    a=p.parse_args();loaded={}
    for key in ('contract','source_receipt','initialization','opponent'):
        path=getattr(a,key);rc.require(rc.digest(path)==getattr(a,key+'_sha256'),'Predeclared '+key+' differs')
        loaded[key]=json.loads(path.read_text())
    c=catalog(a.candidate,a.baseline,a.side,loaded['opponent'],a.phase,loaded['contract'],loaded['source_receipt'],loaded['initialization'],a.initial)
    bind_source({'source_commit':rc.SOURCE_COMMIT,'source_provenance':loaded['source_receipt']},a.source,('replication',))
    for name,sha in rc.ENVIRONMENT.items():rc.require(rc.digest(a.site/name)==sha,'Environment differs')
    a.output.mkdir(parents=True,exist_ok=False);path=a.output/'catalog.json';path.write_text(json.dumps(c,indent=2)+'\n')
    started=now()
    subprocess.run([sys.executable,str(a.source/'tournament/run.py'),'--catalog',str(path),'--site',str(a.site),
                    '--output',str(a.output/'results'),'--device','cuda','--games','64','--parallel','64',
                    '--workers','4','--limit','256','--seed',str(SEEDS[a.phase][loaded['opponent']['id']])],check=True,timeout=150)
    receipt=dict(schema=1,started_at=started,completed_at=now(),catalog_sha256=rc.digest(path),
                 contract_sha256=a.contract_sha256,initialization_sha256=a.initialization_sha256,
                 source_receipt_sha256=a.source_receipt_sha256,opponent_sha256=a.opponent_sha256,
                 evaluator_sha256=rc.digest(Path(__file__)),
                 result_sha256={p.name:rc.digest(p) for p in sorted((a.output/'results').glob('*.json'))})
    (a.output/'evaluation-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')

if __name__=='__main__':main()
