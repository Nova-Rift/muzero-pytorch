"""Bounded real trainer split-resume check, including optimizer/replay/RNG."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import torch
from compressed_checkpoint import compress, restore


def compare(a,b):
    counts={'tensors':0,'values':0}
    def equal(x,y,path):
        if isinstance(x,torch.Tensor):
            if not isinstance(y,torch.Tensor) or not torch.equal(x,y):raise ValueError('Tensor differs: '+path)
            counts['tensors']+=1
        elif isinstance(x,dict):
            if x.keys()!=y.keys():raise ValueError('Keys differ: '+path)
            for k in x:equal(x[k],y[k],path+'.'+str(k))
        elif isinstance(x,(list,tuple)):
            if len(x)!=len(y):raise ValueError('Lengths differ: '+path)
            for i,(xx,yy) in enumerate(zip(x,y)):equal(xx,yy,path+'.'+str(i))
        else:
            if x!=y:raise ValueError('Value differs: '+path)
            counts['values']+=1
    if a.keys()!=b.keys():raise ValueError('State keys differ')
    for k in a:
        if k!='arguments':equal(a[k],b[k],k)
    for k in a['arguments']:
        if k not in ('output','resume_state'):equal(a['arguments'][k],b['arguments'][k],'arguments.'+k)
    return counts


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],required=True)
    p.add_argument('--entrypoint',choices=['train_selfplay.py','train_qcompletion.py','train_loss.py','train_rootweighted.py','train_first_transition_fixed.py'],default='train_qcompletion.py')
    p.add_argument('--seconds',type=int,default=120);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic();deadline=started+a.seconds
    large=a.device=='cuda'
    base=[sys.executable,str(a.source/'arena'/a.entrypoint),'--site',str(a.site),'--parallel','2','--workers','1',
          '--hidden',str(448 if large else 16),'--context',str(64 if large else 2),
          '--simulations',str(32 if large else 2),'--batch','8','--unroll','5','--limit','8','--agents','2',
          '--return-mode','episode','--timeout-bootstrap','zero','--root-candidates',str(8 if large else 2),
          '--discount','1','--seed','23002','--device',a.device,'--max-seconds','60']
    if large:base+=['--device-history','--cuda-graph','--tf32']
    for name,games in [('full',4),('partial',2),('resumed',4)]:
        command=base+['--games',str(games),'--output',str(a.output/name)]
        if name=='resumed':
            state=a.output/'partial/resume-state.pt'
            raw_sha=hashlib.sha256(state.read_bytes()).hexdigest()
            encoded=a.output/'partial-transport.pt.gz';restored=a.output/'partial-restored.pt'
            transport=compress(state,encoded,raw_sha)
            restore(encoded,restored,raw_sha,transport['raw_bytes'])
            command+=['--resume-state',str(restored)]
        remaining=deadline-time.monotonic()
        if remaining<=0:raise TimeoutError('Shared resume gate deadline')
        with (a.output/(name+'.log')).open('w') as log:
            subprocess.run(command,timeout=remaining,check=True,stdout=log,stderr=subprocess.STDOUT)
    states=[torch.load(a.output/name/'resume-state.pt',map_location='cpu',weights_only=True) for name in ['full','resumed']]
    result=dict(verified=True,device=a.device,entrypoint=a.entrypoint,transport=transport,torch_version=str(torch.__version__),counts=compare(*states),
                elapsed_seconds=time.monotonic()-started,games=4,updates=4,
                scope='Tiny real trainer split-resume with complete state equality; no full-workload scaling or strength claim')
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
