"""External checkpoint encoder. Trainer source remains immutable while running."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from compressed_checkpoint import compress


def atomic(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as f:
        json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=int,required=True);p.add_argument('--max-encoded-bytes',type=int,required=True)
    p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    command=a.command[1:] if a.command[:1]==['--'] else a.command
    if not command or a.output.exists():raise ValueError('Fresh output and trainer command required')
    process=subprocess.Popen(command)  # Inherit supervisor process group.
    start=time.monotonic();records={};encoded_bytes=0
    try:
        while True:
            if time.monotonic()-start>a.seconds:raise TimeoutError('Training/encoding deadline')
            state_dir=a.output/'full-state'
            for state in sorted(state_dir.glob('games-*-updates-*.pt')):
                if state.name in records:continue
                # Immutable name exists only after the trainer atomically saves it.
                digest=hashlib.sha256()
                with state.open('rb') as f:
                    while block:=f.read(1048576):digest.update(block)
                target=a.output/'compressed';target.mkdir(exist_ok=True)
                record=compress(state,target/(state.name+'.gz'),digest.hexdigest())
                encoded_bytes+=record['compressed_bytes']
                record['raw_name']=state.name;record['saved_at_unix']=state.stat().st_mtime
                records[state.name]=record
                atomic(target/'manifest.json',dict(states=records,encoded_bytes=encoded_bytes))
                if encoded_bytes>a.max_encoded_bytes:raise ValueError('Encoded recovery storage allowance exceeded')
            code=process.poll()
            if code is not None:
                if code:raise subprocess.CalledProcessError(code,command)
                report=json.loads((a.output/'report.json').read_text())
                if not report['requested_run_completed']:raise ValueError('Training stopped before frozen endpoint')
                # A process may finish while compression runs; scan again if its
                # final immutable milestone arrived after the earlier listing.
                if any(s.name not in records for s in state_dir.glob('games-*-updates-*.pt')):continue
                if not records:raise ValueError('No recovered full-state milestone')
                break
            time.sleep(1)
        atomic(a.output/'encoding-result.json',dict(complete=True,states=len(records),encoded_bytes=encoded_bytes,
                                                   elapsed_seconds=time.monotonic()-start,pod_retained=True))
    finally:
        if process.poll() is None:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)


if __name__=='__main__':main()
