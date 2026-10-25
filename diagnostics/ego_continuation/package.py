"""Assemble a complete inert continuation plan and hash-bound payload recipe."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from run import digest, training_command, validate_manifest

REMOTE='/workspace/experiment';RUN='/workspace/runs/loop-c/ego4096'
SEEDS={'E1024':131110000,'F1024':131210000,'OwnInitial':131310000,'EGO2048':131410000}

def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')

def jobs(manifest):
    source=REMOTE+'/model';site=REMOTE+'/site';original=REMOTE+'/ego2048';diag=REMOTE+'/ego-continuation/run.py';retention=source+'/diagnostics/retention';out=RUN+'/EGO4096';rows=[]
    def add(name,stage,seconds,cmd):
        row=dict(id='ego4096-'+name,loop='loop-c',stage=stage,resource='gpu',cwd=RUN,depends_on=[rows[-1]['id']] if rows else [],command=['timeout',str(seconds)+'s']+cmd)
        rows.append(row)
    add('runtime','test',20,['python','-c',"import platform,torch; assert str(torch.__version__)=='2.8.0+cu128'; assert platform.python_version()=='3.12.3'; assert torch.cuda.is_available() and torch.cuda.device_count()==1; assert 'H100' in torch.cuda.get_device_name(); print('Exact EGO runtime verified')"])
    add('split-resume','test',120,['python',retention+'/resume_gate.py','--source',source,'--site',site,'--output',RUN+'/split-resume','--device','cuda','--entrypoint','train_loss.py','--updates-per-game','1','--loss-contract','baseline','--aim-decoding','relative','--seconds','110'])
    add('input','test',120,['python',diag,'input','--source',source,'--site',site,'--original',original,'--output',RUN+'/input-gate'])
    train=training_command(manifest,source,site,out,RUN+'/input-gate/resume.pt')
    add('train','train',660,['python',retention+'/train_compressed.py','--output',out,'--seconds','640','--max-encoded-bytes','600000000','--']+train)
    add('recovery','test',120,['python',diag,'post','--source',source,'--original',original,'--training',out,'--output',RUN+'/post-recovery.json'])
    for opponent in ('E1024','F1024','OwnInitial','EGO2048'):
        for label,training,games in [('EGO4096',out,4096)]+([] if opponent=='EGO2048' else [('EGO2048',original,2048)]):
            add(label+'-'+opponent,'evaluate',160,['python',retention+'/evaluate_endpoint.py','--training',training,'--paired-training',original,'--expected-endpoint-games',str(games),'--expected-endpoint-updates',str(games),'--opponent',REMOTE+'/ego-continuation/'+opponent+'.json','--label',label,'--output',RUN+'/'+label+'-'+opponent,'--source',source,'--site',site,'--seed',str(SEEDS[opponent])])
    return rows

def build(root,workspace):
    frozen=root/'.local/research-loop/iteration34-frozen-package';base=frozen/'staged/payload/experiment';original=frozen/'runner-ego/artifacts/loop-a/EGO2048';prepared=workspace/'prepared';prepared.mkdir(exist_ok=False)
    manifest=json.loads((original/'manifest.json').read_text());receipt=json.loads((base/'model/.source-provenance.json').read_text());validate_manifest(manifest,receipt)
    rows=jobs(manifest);write(prepared/'initial-jobs.json',rows)
    write(prepared/'plan-template.json',dict(schema=1,max_parallel_loops=4,gpu_count=1,close_when_idle=False,jobs=rows,deadline=None,session_deadline=None,admission_required='Coordinator binds live deadlines, runner identity, exact storage approval and quote before submission.'))
    for opponent in ('E1024','F1024'):
        write(prepared/(opponent+'.json'),json.loads((base/(opponent+'.json')).read_text()))
    for label,filename,trained in [('OwnInitial','initial.npz',False),('EGO2048','weights.npz',True)]:
        write(prepared/(label+'.json'),dict(id=label,family='independent_muzero',checkpoint=REMOTE+'/ego2048/'+filename,sha256=digest(original/filename),config=manifest['config'],root_candidates=8,trained=trained,aim_decoding='relative'))
    inputs=[]
    def add(local,target):
        inputs.append(dict(local=str(local.resolve()),target=target,bytes=local.stat().st_size,sha256=digest(local)))
    # Full frozen environment and training source; old baseline/unused diagnostics are omitted.
    for directory in ('model','site','weights'):
        for p in sorted((base/directory).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and '.git' not in p.parts:
                add(p,'experiment/'+str(p.relative_to(base)))
    for name in ('manifest.json','report.json','initial.npz','weights.npz','compressed/manifest.json','compressed/games-2048-updates-2048.pt.gz'):
        add(original/name,'experiment/ego2048/'+name)
    add(Path(__file__).with_name('run.py'),'experiment/ego-continuation/run.py')
    for name in ('E1024','F1024','OwnInitial','EGO2048'):add(prepared/(name+'.json'),'experiment/ego-continuation/'+name+'.json')
    for name in ('workload_supervisor.py','workload_reporting.py','lease_watchdog.py','transport_command.py'):add(root/'tools/research-loop'/name,name)
    write(prepared/'payload-inputs.json',dict(schema=1,files=inputs,total_bytes=sum(r['bytes'] for r in inputs),scope='Reference-only recipe: all sources rehashed; no checkpoint copied or decoded locally. Materialize only after storage admission.'))
    protocol=dict(schema=1,name='EGO2048 full-state unchanged continuation to4096',hypothesis='Original development EGO2048 beat E1024 in34/64 versus baseline15/64 and F1024 in61/64 versus baseline30/64. Additional unchanged training may improve E, but strength is unproven.',training=dict(mode='full-state exact-resume candidate',source_commit=manifest['source_commit'],start_games=2048,end_games=4096,new_updates=2048,seed=52731415,new_collection_seed_range=[52733463,52735510],unchanged_scientific_arguments={k:v for k,v in manifest['arguments'].items() if k not in {'site','output','resume_state','games','max_seconds'}}),development=dict(games_per_comparison=64,total_games=448,balanced_seats=True,seed_ranges={k:[v,v+63] for k,v in SEEDS.items()},matched='Both endpoints vs E1024/F1024 and own shared initialization on identical fresh seeds; direct continuation vs EGO2048 on separate seeds.',own_initial='Both policies use relative decoding, unlike old absolute-decoding baseline OwnInitial. This comparison is within EGO only.',selection='Compare per-seed matched win/cutoff outcomes and uncertainty; do not claim ladder improvement or publish from development alone.'),future_heldout=dict(admitted=False,reserved_seed_ranges={'E1024':[133110000,133110063],'F1024':[133210000,133210063]},rule='Never reuse existing confirmation outcomes to tune this continuation. Freeze any later heldout admission before reading it; paired EGO2048 control required.'),gates=['exact source/runtime','relative-aim CUDA split-resume equality','actual recovered2048 full-state identity','4096 completion','all four new checkpoint recovery states','dependent development evaluations'],publication=False)
    write(prepared/'protocol.json',protocol)
    write(prepared/'runtime-cost.json',dict(observed_training_seconds=314.9,observed_training_with_encoding_seconds=321.9,observed_eval_seconds_per64=[42,44],expected_compute_seconds=[750,1100],sum_job_timeout_seconds=sum(int(x['command'][1][:-1]) for x in rows),setup_transfer_bound_seconds=600,recovery_bound_seconds=300,cleanup_bound_seconds=120,contingency_20pct_seconds=636,whole_run_bound_seconds=3816,same_pod_bound_seconds=3096,whole_run_cost_formula='quoted_dollars_per_hour * 3816 / 3600; same-pod quoted_rate *3096/3600. Admit only with exact budget/renewal coverage.',example_rate_not_quote=3.99,example_max_cost=round(3.99*3816/3600,4),example_same_pod_max_cost=round(3.99*3096/3600,4),reservation_note='At indicative3.99/h conservative wall bounds exceed one$3segment; needs authorized automatic renewal and enough remaining daily budget. No reservation made.',uncertainties=['Continuation replay/state sizes may change;600MB compressed cap aborts safely.','H100 timings from original2048 do not guarantee continuation time.','Setup includes serialized upload;600s is bound, not measured transfer duration.','Recovery300s and cleanup120s preserve prior runner bounds;20pct contingency applied to all.','Fresh or reused pod needs lease renewal for worst case; exact current session/lease must fit.']))
    write(prepared/'storage-plan.json',dict(schema=1,iteration=34,approved=False,jobs_sha256=digest(prepared/'initial-jobs.json'),recovered_artifacts_bytes=1200000000,archive_growth_bytes=1200000000,scratch_peak_bytes=1500000000,other_outstanding_bytes=2000000000,headroom_bytes=4294967296,evidence=dict(input_payload_bytes=sum(r['bytes'] for r in inputs),source_inputs_already_resident=True,recovery='600MB compressed states +400MB16 weights/traces/eval +200MBsplitresume diagnostic; recover excludes raw*.pt; preserve remote evidence until hash-verified recovery',archive='One new full recovered copy, conservative despite dedup',scratch='1.5GB includes new payload staging+tar copies plus up to1.1GB sequential decoder scratch, not old resident checkpoint bytes',other_outstanding='2GB coordinator-bound outstanding: initializer64MiB recovery+64MiBarchive+64MiBscratch, EGO archive1.2GB, other sourcehelpers512MiB combined and small reports. Mutually exclusive with any new full replication training; fresh recost before both are admitted',remote_scratch_peak_bytes=17179869184,remote='Retain original decoded state +fournew raw states + compressed chain + splitresume + payload.16GiB estimate; verify actual capacity.',approval='Coordinator exact-job storage readiness/archive quota/other tasks/deadlines/rate not yet checked. No paid admission.')))
    return prepared

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--workspace',type=Path,required=True);a=p.parse_args();print(build(a.root.resolve(),a.workspace.resolve()))
if __name__=='__main__':main()
