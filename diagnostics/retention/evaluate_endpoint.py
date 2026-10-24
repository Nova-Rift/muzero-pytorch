"""Evaluate one frozen final endpoint with a common original-search controller."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def catalog(training,paired,opponent,label):
    report=json.loads((training/'report.json').read_text())
    if not report['requested_run_completed'] or report['games']!=2048 or report['optimizer_updates']!=2048:
        raise ValueError('Frozen2048 endpoint incomplete')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if sha(training/'initial.npz')!=sha(paired/'initial.npz'):
        raise ValueError('Paired initialization differs')
    manifest=json.loads((training/'manifest.json').read_text())
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
    p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    frozen=catalog(a.training,a.paired_training,json.loads(a.opponent.read_text()),a.label)
    path=a.output/'catalog.json';path.write_text(json.dumps(frozen,indent=2)+'\n')
    subprocess.run([sys.executable,str(a.source/'tournament/run.py'),'--catalog',str(path),
                    '--site',str(a.site),'--output',str(a.output/'results'),'--device','cuda',
                    '--games','64','--parallel','64','--workers','4','--limit','256','--seed',str(a.seed)],
                   timeout=150,check=True)


if __name__=='__main__':main()
