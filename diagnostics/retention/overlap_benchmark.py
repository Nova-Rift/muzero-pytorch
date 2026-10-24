"""Bounded scratch serial/overlap measurement. Never enables supervisor slots itself."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(plan):
    return hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def memory_sample():
    raw = subprocess.run(['nvidia-smi', '--query-gpu=uuid,memory.total,memory.used', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True, check=True, timeout=2).stdout
    rows = [x.strip().split(',') for x in raw.strip().splitlines()]
    if len(rows) != 1:
        raise ValueError('Benchmark requires exactly one visible GPU')
    uuid, total, used = rows[0]
    return dict(uuid=uuid.strip(), total_bytes=int(total)*1048576, used_bytes=int(used)*1048576)


def terminate(groups):
    # Every child starts its own group; never target parent/supervisor/scientific groups.
    for proc in groups:
        try: os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError: pass
    end = time.monotonic()+2
    while any(p.poll() is None for p in groups) and time.monotonic()<end:
        time.sleep(.02)
    for proc in groups:
        try: os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError: pass
    for proc in groups:
        try: proc.wait(timeout=2)
        except subprocess.TimeoutExpired: pass


def run_group(commands, directories, deadline, sampler=memory_sample):
    started = time.monotonic(); processes=[]; logs=[]; samples=[]
    try:
        for command, directory in zip(commands, directories):
            if time.monotonic()>=deadline: raise TimeoutError('Global benchmark deadline')
            directory.mkdir(parents=True, exist_ok=False)
            log=(directory/'process.log').open('w'); logs.append(log)
            argv=[str(directory) if x=='{output}' else x for x in command]
            processes.append(subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        while True:
            if time.monotonic()>=deadline: raise TimeoutError('Global benchmark deadline')
            sample=sampler(); sample['elapsed_seconds']=time.monotonic()-started; samples.append(sample)
            codes=[p.poll() for p in processes]
            if any(c is not None and c!=0 for c in codes): raise RuntimeError(f'Child failure: {codes}')
            if all(c==0 for c in codes): break
            time.sleep(.2)
        return dict(elapsed_seconds=time.monotonic()-started, samples=samples, exit_codes=codes)
    finally:
        terminate(processes)
        for log in logs: log.close()
        if directories and directories[0].exists():
            (directories[0]/'measurement.json').write_text(json.dumps({'elapsed_seconds':time.monotonic()-started,'samples':samples,'exit_codes':[p.poll() for p in processes]},indent=2)+'\n')


def inspect(directory, arm, plan):
    report=json.loads((directory/'report.json').read_text())
    manifest=json.loads((directory/'manifest.json').read_text())
    if not report['requested_run_completed'] or report['games']!=128 or report['optimizer_updates']!=128:
        raise ValueError('Incomplete exact work')
    if manifest['source_commit']!=arm['source_commit'] or manifest['torch']!=plan['torch'] or manifest['gpu']!=plan['gpu']:
        raise ValueError('Runtime/source mismatch')
    if manifest['config']!=arm['config']: raise ValueError('Model config mismatch')
    actual={k:v for k,v in manifest['arguments'].items() if k!='output'}
    if actual!=arm['expected_arguments']: raise ValueError('Trainer arguments mismatch')
    if digest(directory/'weights.npz')!=report['checkpoint_sha256']: raise ValueError('Checkpoint mismatch')
    return dict(games=128, optimizer_updates=128, initial_sha256=digest(directory/'initial.npz'),
                checkpoint_sha256=report['checkpoint_sha256'], manifest=manifest, report=report)


def assess(serial_seconds, concurrent_seconds, samples, same_outputs):
    if not samples or serial_seconds<=0 or concurrent_seconds<=0: raise ValueError('Missing measurements')
    total=min(s['total_bytes'] for s in samples); peak=max(s['used_bytes'] for s in samples)
    speedup=serial_seconds/concurrent_seconds; headroom=total-peak
    return dict(eligible=same_outputs and speedup>=1.1 and headroom>=max(.15*total,8*1024**3),
                max_parallel=2, sequential_work_per_second=256/serial_seconds,
                concurrent_work_per_second=256/concurrent_seconds, speedup=speedup,
                peak_memory_bytes=peak, available_memory_bytes=total, headroom_bytes=headroom,
                exact_serial_concurrent_checkpoints=same_outputs)


def execute(plan, output):
    if plan.get('approved') is not True or not plan.get('files'): raise ValueError('Unapproved or unbound plan')
    if plan['seconds']>240 or plan['seconds']<=0: raise ValueError('Maximum240 seconds')
    if len(plan['arms'])!=2: raise ValueError('Exactly two arms')
    if len(plan['target_plan_sha256'])!=64: raise ValueError('Target fingerprint required')
    for filename, sha in plan['files'].items():
        if digest(filename)!=sha: raise ValueError('Bound source/runtime file differs: '+filename)
    for arm in plan['arms']:
        if arm['command'].count('{output}')!=1: raise ValueError('One isolated output required')
    output.mkdir(parents=True,exist_ok=False)
    deadline=time.monotonic()+plan['seconds']; result=dict(schema=1, complete=False, eligible=False,
        benchmark_plan_sha256=fingerprint(plan), plan_sha256=plan['target_plan_sha256'], started_at=time.time(),
        full_plan_authorized=False,
        scope='128-game cold-start blocks only; no steady-state or full-run overlap authorization', groups=[])
    try:
        serial=[]; concurrent=[]
        for i,arm in enumerate(plan['arms']):
            d=output/f'serial-{i}'
            group=run_group([arm['command']],[d],deadline); result['groups'].append(group)
            serial.append(inspect(d,arm,plan))
        dirs=[output/f'concurrent-{i}' for i in range(2)]
        group=run_group([a['command'] for a in plan['arms']],dirs,deadline); result['groups'].append(group)
        concurrent=[inspect(d,a,plan) for d,a in zip(dirs,plan['arms'])]
        same=all(a['initial_sha256']==b['initial_sha256'] and a['checkpoint_sha256']==b['checkpoint_sha256'] for a,b in zip(serial,concurrent))
        result.update(assess(sum(g['elapsed_seconds'] for g in result['groups'][:2]),group['elapsed_seconds'],group['samples'],same))
        result.update(complete=True, serial=serial, concurrent=concurrent)
    except Exception as error:
        result['error']=str(error)
    finally:
        result['finished_at']=time.time(); (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    def stopped(signum,frame): raise RuntimeError('Benchmark interrupted')
    signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
    result=execute(json.loads(a.plan.read_text()),a.output)
    print(json.dumps({k:result.get(k) for k in ['complete','eligible','speedup','error']}))
    # An inconclusive measurement is evidence, not permission to overlap.

if __name__=='__main__':main()
