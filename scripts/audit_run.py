"""Check saved work counts, inference weights, and the measured speed claim."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model import Config,MuZero,scalar


def audit(path,expected_games=None):
    report=json.loads((path/'report.json').read_text())
    manifest=json.loads((path/'manifest.json').read_text())
    rows=[json.loads(line) for line in (path/'metrics.jsonl').read_text().splitlines()]
    args=manifest['arguments']
    if expected_games is not None:assert report['games']==expected_games
    assert report['games']==report['natural_terminal']+report['cutoffs']
    assert report['optimizer_updates']==report['games']*args['updates_per_game']
    assert report['training_examples']==report['optimizer_updates']*args['batch']
    for field in ['games','natural_terminal','cutoffs','joint_steps','active_transitions','optimizer_updates','training_examples']:
        assert rows[-1][field]==report[field],field
        assert all(a[field]<=b[field] for a,b in zip(rows,rows[1:])),field
    assert sum(row['batch_seconds']['games'] for row in rows)==report['games']
    for key in ['natural_terminal','cutoffs','joint_steps','active_transitions']:
        assert sum(row['batch_seconds'][key] for row in rows)==report[key],key
    weights=path/'weights.npz'
    assert hashlib.sha256(weights.read_bytes()).hexdigest()==report['checkpoint_sha256']
    torch.set_num_threads(1)
    model=MuZero(Config(**manifest['config']))
    with np.load(weights,allow_pickle=False) as data:
        assert all(np.isfinite(data[k]).all() for k in data.files)
        model.load_state_dict({k:torch.from_numpy(data[k].copy()) for k in data.files},strict=True)
    assert sum(p.numel() for p in model.parameters())==manifest['parameters']
    with torch.no_grad():
        state,policy,value=model.initial(torch.zeros(2,model.config.observations))
        following=model.recurrent(state,torch.tensor([0,1]))
        assert all(torch.isfinite(x).all() for x in (state,policy,value,*following))
        predicted=scalar(value).tolist()
    process_file=path/'process-timing.json'
    process=json.loads(process_file.read_text()) if process_file.exists() else None
    if process:
        assert process['exit_code']==0
        assert process['process_wall_seconds']>=report['elapsed_seconds']
    actual_claim=report['games']>=10000 and report['elapsed_seconds']<=600
    assert report['target_10000_under_600_met']==actual_claim
    return dict(verified=True,games=report['games'],natural_terminal=report['natural_terminal'],cutoffs=report['cutoffs'],optimizer_updates=report['optimizer_updates'],parameters=manifest['parameters'],training_seconds=report['elapsed_seconds'],process_wall_seconds=process['process_wall_seconds'] if process else None,full_process_target_met=bool(process and report['games']>=10000 and process['process_wall_seconds']<=600),checkpoint_sha256=report['checkpoint_sha256'],zero_input_value_sanity=predicted,claim='Integrity and throughput audit only; no policy quality assessment')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('path',type=Path);parser.add_argument('--expected-games',type=int)
    args=parser.parse_args();print(json.dumps(audit(args.path,args.expected_games),indent=2))
