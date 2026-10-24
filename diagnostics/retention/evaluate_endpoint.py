"""Evaluate one frozen final endpoint with a common original-search controller."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def validate_continuation(manifest, paired, source):
    """Bind the new endpoint to its frozen training source and original run."""
    if source is None:
        raise ValueError('Continuation requires frozen training source')
    original=json.loads((paired/'manifest.json').read_text())
    for field in ('config', 'environment', 'torch', 'python', 'source_commit'):
        if not manifest.get(field) or manifest[field]!=original.get(field):
            raise ValueError('Continuation differs: '+field)
    for record in (manifest, original):
        provenance=record.get('source_provenance', {})
        if (provenance.get('dirty') is not False or
                provenance.get('commit')!=record['source_commit'] or not provenance.get('files')):
            raise ValueError('Missing or dirty source provenance')
    files=manifest['source_provenance']['files']
    if files!=original['source_provenance']['files']:
        raise ValueError('Continuation source files differ')
    operational={'site','output','resume_state','games','max_seconds'}
    arguments=manifest.get('arguments', {})
    old_arguments=original.get('arguments', {})
    if (not arguments or not old_arguments or
            {k:v for k,v in arguments.items() if k not in operational}!=
            {k:v for k,v in old_arguments.items() if k not in operational} or
            arguments.get('games')!=4096 or arguments.get('updates_per_game')!=1 or
            not arguments.get('resume_state') or old_arguments.get('games')!=2048):
        raise ValueError('Continuation training arguments differ')
    source=Path(source).resolve()
    for name, expected in files.items():
        path=(source/name).resolve()
        if not path.is_relative_to(source) or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('Frozen training source changed: '+name)
    receipt=source/'.source-provenance.json'
    if receipt.exists():
        actual=json.loads(receipt.read_text())
        if actual.get('dirty') is not False or actual.get('files')!=files:
            raise ValueError('Frozen source receipt differs')
        commit=actual.get('commit')
    else:
        commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if commit!=manifest['source_commit']:
        raise ValueError('Frozen source commit differs')


def catalog(training,paired,opponent,label,expected_endpoint_games=2048,source=None):
    report=json.loads((training/'report.json').read_text())
    if type(expected_endpoint_games) is not int or expected_endpoint_games not in (2048,4096):
        raise ValueError('Supported endpoints are 2048 and 4096')
    if (report['requested_run_completed'] is not True or
            type(report['games']) is not int or report['games']!=expected_endpoint_games or
            type(report['optimizer_updates']) is not int or report['optimizer_updates']!=expected_endpoint_games):
        raise ValueError(f'Frozen{expected_endpoint_games} endpoint incomplete')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if sha(training/'initial.npz')!=sha(paired/'initial.npz'):
        raise ValueError('Paired initialization differs')
    manifest=json.loads((training/'manifest.json').read_text())
    if expected_endpoint_games==4096:
        validate_continuation(manifest,paired,source)
    weights=training/'weights.npz'
    if sha(weights)!=report['checkpoint_sha256']:raise ValueError('Final checkpoint differs from report')
    if sha(Path(opponent['checkpoint']))!=opponent['sha256']:raise ValueError('Frozen opponent changed')
    return {'models':[dict(id=label,family='independent_muzero',checkpoint=str(weights),sha256=sha(weights),
                           config=manifest['config'],root_candidates=8,trained=True),opponent]}


def main():
    p=argparse.ArgumentParser();p.add_argument('--training',type=Path,required=True)
    p.add_argument('--paired-training',type=Path,required=True);p.add_argument('--opponent',type=Path,required=True)
    p.add_argument('--label',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--site',type=Path,required=True)
    p.add_argument('--expected-endpoint-games',type=int,choices=(2048,4096),default=2048)
    p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    frozen=catalog(a.training,a.paired_training,json.loads(a.opponent.read_text()),a.label,a.expected_endpoint_games,a.source)
    path=a.output/'catalog.json';path.write_text(json.dumps(frozen,indent=2)+'\n')
    subprocess.run([sys.executable,str(a.source/'tournament/run.py'),'--catalog',str(path),
                    '--site',str(a.site),'--output',str(a.output/'results'),'--device','cuda',
                    '--games','64','--parallel','64','--workers','4','--limit','256','--seed',str(a.seed)],
                   timeout=150,check=True)


if __name__=='__main__':main()
