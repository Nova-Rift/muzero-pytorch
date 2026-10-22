"""Registered L orchestration; one deadline covers every child process group."""
import argparse,hashlib,json,os,signal,subprocess,time
from pathlib import Path

def select(rows):
    return max(rows,key=lambda x:(min(r['wins']/r['games'] for r in x['results']),sum(r['wins'] for r in x['results']),-sum(r['cutoffs'] for r in x['results']),-x['checkpoint']))

def dev_pass(row):
    return all(r['wins']/r['games']>.6 and r['cutoffs']/r['games']<=.2 for r in row['results'])

def wilson(w,n):
    z=1.959963984540054;p=w/n
    return (p+z*z/(2*n)-z*((p*(1-p)+z*z/(4*n))/n)**.5)/(1+z*z/n)

def main():
    p=argparse.ArgumentParser();p.add_argument('--site',required=True);p.add_argument('--output',required=True);p.add_argument('--weights',required=True);p.add_argument('--deadline-unix',type=float,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];out=Path(a.output);end=time.monotonic()+max(1,a.deadline_unix-time.time())
    progress=out.parent/(out.name+'-progress.json');exitfile=out.parent/(out.name+'-exit.json')
    if out.exists() or exitfile.exists():raise ValueError('Existing run; never overwrite')
    def status(**kw):
        tmp=progress.with_suffix('.tmp');tmp.write_text(json.dumps(kw));tmp.replace(progress)
    def run(cmd):
        # Require local owner-control review before each training/evaluation subprocess.
        token=str(time.time_ns())
        request=out.parent/(out.name+'-boundary.json')
        request.write_text(json.dumps({'token':token}))
        approval=out.parent/(out.name+'-approved.txt')
        while not approval.exists() or approval.read_text().strip()!=token:
            if time.monotonic()>=end:raise TimeoutError('Owner boundary deadline')
            time.sleep(1)
        print(json.dumps(cmd),flush=True)
        proc=subprocess.Popen(cmd,cwd=root,start_new_session=True)
        try:
            code=proc.wait(timeout=max(1,end-time.monotonic()))
            if code:raise RuntimeError('Child exit '+str(code))
        except BaseException:
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            raise
    result={'training_completed':False,'evaluation_completed':False}
    try:
        protocol=json.loads((root/'docs/automation/iteration-6-protocol.json').read_text())
        for name,h in protocol['source_sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h,name
        for name,h in protocol['environment_sha256'].items():assert hashlib.sha256((Path(a.site)/name).read_bytes()).hexdigest()==h,name
        for name,h in protocol['opponent_sha256'].items():assert hashlib.sha256((Path(a.weights)/('selfplay-'+name+'.npz')).read_bytes()).hexdigest()==h,name
        status(stage='train',title='Training L from scratch',evaluation_label='No new evaluation',wins=0,evaluation_games=0,timeouts=0)
        run(['python','arena/train_selfplay.py','--site',a.site,'--output',a.output,'--games','4096','--parallel','128','--workers','4','--hidden','448','--context','64','--simulations','32','--batch','256','--unroll','5','--return-mode','episode','--root-candidates','8','--cadence','homepage','--agents','2','--policy-floor','0','--noise-fraction','.25','--limit','256','--updates-per-game','1','--seed','51000000','--max-seconds','1200','--discount','1.0','--replay-blocks','3','--device','cuda','--device-history','--cuda-graph','--tf32'])
        result['training_completed']=json.loads((out/'report.json').read_text())['requested_run_completed']
        if not result['training_completed']:raise RuntimeError('Training incomplete; no continuation')
        import shutil
        for name in ['E1024','F1024']:shutil.copyfile(Path(a.weights)/('selfplay-'+name+'.npz'),out/(name+'.npz'))
        def evaluate(epoch,label,count,seeds):
            reports=[]
            for opponent,seed in zip(['initial','E1024','F1024'],seeds):
                status(stage='evaluate',title=f'Evaluating L{epoch} against {opponent}',evaluation_label=f'{label}: L{epoch} vs {opponent}',wins=0,evaluation_games=0,timeouts=0)
                path=out/f'{label}-{epoch}-{opponent}.json'
                live=out/f'{label}-{epoch}-{opponent}-live.json'
                status(stage='evaluate',title=f'Evaluating L{epoch} against {opponent}',evaluation_label=f'{label}: L{epoch} vs {opponent}',wins=0,evaluation_games=0,timeouts=0,live_file=live.name,matchup={'challenger':f'MuZero L · {epoch} games','opponent':f'MuZero {opponent}','opponent_wins':0,'draws':0})
                run(['python','scripts/evaluate_selfplay.py','--site',a.site,'--run',a.output,'--challenger',f'checkpoint-{epoch}.npz','--reference',opponent+'.npz','--reference-discount','1.0' if opponent=='initial' else '.99','--games',str(count),'--seed',str(seed),'--output',str(path),'--device','cuda','--progress',str(live)])
                r=json.loads(path.read_text());assert len(r['records'])==count and sum(x['seat']==0 for x in r['records'])==count//2
                reports.append(r)
                status(stage='evaluate',title=f'Evaluating L{epoch} against {opponent}',evaluation_label=f'{label}: L{epoch} vs {opponent}',wins=r['wins'],evaluation_games=r['games'],timeouts=r['cutoffs'],matchup={'challenger':f'MuZero L · {epoch} games','opponent':f'MuZero {opponent}','opponent_wins':r['losses'],'draws':r['draws']})
            return {'checkpoint':epoch,'results':reports}
        rows=[evaluate(e,'development',64,[82000000,82010000,82020000]) for e in [1024,2048,4096]]
        selected=select(rows);epoch=selected['checkpoint']
        frozen={'checkpoint':epoch,'sha256':hashlib.sha256((out/f'checkpoint-{epoch}.npz').read_bytes()).hexdigest(),'development_pass':dev_pass(selected),'selection':selected,'frozen_at':time.time()}
        (out/'selection.json').write_text(json.dumps(frozen,indent=2))
        result.update(evaluation_completed=True,selected=epoch,development_pass=frozen['development_pass'],confirmation='not run: development gate failed',promotion_eligible=False)
        if frozen['development_pass'] and end-time.monotonic()>=600:
            if end-time.monotonic()<420:raise TimeoutError('Insufficient time for complete confirmation and recovery; selection preserved')
            assert hashlib.sha256((out/f'checkpoint-{epoch}.npz').read_bytes()).hexdigest()==frozen['sha256']
            confirm=evaluate(epoch,'confirmation',200,[83000000,83010000,83020000])
            passed=all(wilson(r['wins'],r['games'])>.5 and r['cutoffs']/r['games']<=.2 for r in confirm['results'])
            result.update(confirmation='passed' if passed else 'failed',replication='required before promotion')
        if frozen['development_pass'] and 'replication' not in result:result['confirmation']='deferred: insufficient remaining time; selection frozen'
        result['status']='completed'
    except BaseException as e:
        result.update(status='failed',error=f'{type(e).__name__}: {e}');raise
    finally:exitfile.write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
