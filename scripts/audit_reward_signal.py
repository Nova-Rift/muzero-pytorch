"""Read-only reward/value calibration on preserved self-play trace transitions.

These are training-derived diagnostic data, never a held-out policy score.
Includes every valid transition from each saved two-game trace (not all games).
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import Config, MuZero, scalar
from arena.train_selfplay import history

def metrics(pred, target):
    return {'count':len(pred), 'prediction_mean':float(np.mean(pred)),
            'target_mean':float(np.mean(target)), 'mse':float(np.mean((pred-target)**2)),
            'constant_mean_mse':float(np.var(target)),
            'prediction_std':float(np.std(pred)), 'prediction_min':float(np.min(pred)),
            'prediction_max':float(np.max(pred))} if len(pred) else {'count':0}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Preserve previous audit output')
    torch.set_num_threads(1)
    meta=json.loads((args.run/'manifest.json').read_text())
    traces=sorted(args.run.glob('trace-*.npz'),key=lambda p:int(p.stem.split('-')[1]))
    report={'scope':'training-derived saved two-game traces; no new games, updates, or final seeds',
            'trace_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in traces},'checkpoints':{}}
    for name in ['initial.npz','checkpoint-512.npz','checkpoint-1024.npz','checkpoint-2048.npz']:
        model=MuZero(Config(**meta['config'])).eval()
        with np.load(args.run/name,allow_pickle=False) as z:
            model.load_state_dict({k:torch.from_numpy(z[k]) for k in z.files},strict=True)
        rows=[]
        for file in traces:
            with np.load(file,allow_pickle=False) as z:
                b={k:z[k] for k in z.files}
            picks=np.argwhere(b['valid'][:-1])
            for start in range(0,len(picks),128):
                t,a=picks[start:start+128].T
                x=history(b['observation'],b['actions'],t,a,meta['arguments']['context'])
                with torch.no_grad():
                    h,p,v=model.initial(torch.from_numpy(x))
                    _,r,_,nv=model.recurrent(h,torch.from_numpy(b['actions'][t,a]))
                    probability=p.softmax(-1)
                    entropy=-(probability*probability.clamp_min(1e-30).log()).sum(-1)
                    rows.append(np.stack([scalar(v).numpy(),b['targets'][t,a],scalar(r).numpy(),b['rewards'][t,a],scalar(nv).numpy(),entropy.numpy()],axis=-1))
        data=np.concatenate(rows)
        win=data[:,3]>0
        report['checkpoints'][name]={
            'sha256':hashlib.sha256((args.run/name).read_bytes()).hexdigest(),
            'root_value':metrics(data[:,0],data[:,1]),
            'reward':metrics(data[:,2],data[:,3]),
            'winning_transition_reward':metrics(data[win,2],data[win,3]),
            'other_transition_reward':metrics(data[~win,2],data[~win,3]),
            'policy_entropy_mean':float(data[:,5].mean()),
            'winning_reward_above_half':int((data[win,2]>.5).sum()),
            'nonwinning_reward_above_half':int((data[~win,2]>.5).sum())}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checkpoints'],indent=2))

if __name__=='__main__':main()
