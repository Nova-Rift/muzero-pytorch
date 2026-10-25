"""Exact conditional EGO argv templates; never execute, allocate or approve work."""
import argparse
import copy
import json
from pathlib import Path
import re
import replication_contract as rc
import replication_package as package

PINNED={'ego_template_sha256':'0c6dd8e21b5b21db69062761e10dba7224565a211191255c1071d88a91a23682',
        'source_receipt_sha256':'94fa57c06b8e98951ccbb824a0ba7b71b699cdb93562adfd569593ca67e65df3'}
TOKEN=re.compile(r'\{([a-zA-Z0-9_]+)\}')

def arg(command,key,value):
    command.extend(['--'+key,value])

def python(script):return ['{python}','{adapters}/'+script+'.py']

def collection(side,mode):
    c=python('replication_collection')
    for key,value in [('source','{source}'),('site','{site}'),('output','{'+side+('_reference}' if mode=='reference' else '_training}')),('initialization-dir','{initialization_dir}'),('source-receipt','{source_receipt}'),('initialization-review','{initialization_review}'),('side',side),('mode',mode),('max-seconds','300' if mode=='reference' else '850')]:arg(c,key,value)
    for key in ('contract','initialization','source-receipt','initialization-review'):arg(c,key+'-sha256','{'+key.replace('-','_')+'_sha256}')
    if mode=='train':
        arg(c,'own-reference','{'+side+'_reference}');arg(c,'own-reference-sha256','{'+side+'_reference_sha256}')
    if side=='candidate':
        arg(c,'paired-reference','{paired_reference}');arg(c,'paired-reference-sha256','{paired_reference_sha256}')
    if mode=='train':
        c=python('train_compressed')+['--output','{'+side+'_training}','--seconds','960','--max-encoded-bytes','600000000','--']+c
    return c

def specification():
    graph=package.validate_graph(package.graph())
    jobs=copy.deepcopy(graph['jobs']);by={j['id']:j for j in jobs}
    c=python('replication_initialization')
    for k,v in [('template','{ego_template}'),('template-sha256','{ego_template_sha256}'),('source','{source}'),('output','{initialization_dir}')]:arg(c,k,v)
    by['no-update-initialization']['commands']=[c]
    for side in ('paired','candidate'):
        c=python('resume_gate')+['--source','{source}','--site','{site}','--output','{'+side+'_resume}','--device','cuda','--entrypoint','train_loss.py','--updates-per-game','1','--loss-contract','baseline','--seconds','120']
        if side=='candidate':c+=['--aim-decoding','relative']
        by[side+'-resume']['commands']=[c]
        by[side+'-collection']['commands']=[collection(side,'reference')]
        by[side+'-fresh2048']['commands']=[collection(side,'train')]
    commands=[]
    for side in ('paired','candidate'):
        c=python('replication_recovery')
        for k,v in [('root','{'+side+'_training}'),('source-receipt','{source_receipt}'),('initialization-dir','{initialization_dir}'),('initialization-review','{initialization_review}'),('output','{'+side+'_recovery_output}'),('side',side)]:arg(c,k,v)
        for k in ('source-receipt','contract','initialization','initialization-review','collection','encoding'):
            token=(side+'_' if k in ('collection','encoding') else '')+k.replace('-','_')+'_sha256'
            arg(c,k+'-sha256','{'+token+'}')
        commands.append(c)
    c=python('replication_contract')
    for k,v in [('candidate','{candidate_training}'),('paired','{paired_training}'),('contract','{contract}'),('source-receipt','{source_receipt}'),('source','{source}'),('contract-sha256','{contract_sha256}'),('source-receipt-sha256','{source_receipt_sha256}')]:arg(c,k,v)
    commands.append(c);by['exact-pair-and-recovery']['commands']=commands
    for j in jobs:
        if j['stage']=='evaluate':
            c=python('evaluate_replication');op=j['opponent']
            for k,v in [('candidate','{candidate_training}'),('baseline','{paired_training}'),('contract','{contract}'),('source-receipt','{source_receipt}'),('initialization','{initialization}'),('initial','{initial}'),('opponent','{opponent_'+op+'}'),('source','{source}'),('site','{site}'),('output','{evaluation_root}/'+j['id']),('side',j['side']),('phase',j['phase'])]:arg(c,k,v)
            for k in ('contract','source-receipt','initialization','opponent'):
                arg(c,k+'-sha256','{'+('opponent_'+op if k=='opponent' else k.replace('-','_'))+'_sha256}')
            j['commands']=[c]
        if j['id'] in ('development-readout','confirmation-readout'):
            phase=j['id'].split('-')[0]
            j['commands']=[python('readout_replication')+['{'+phase+'_request}','--request-sha256','{'+phase+'_request_sha256}','--output','{'+phase+'_readout}']]
        j['external_review_required']='commands' not in j
    tokens=sorted({m for j in jobs for c in j.get('commands',[]) for s in c for m in TOKEN.findall(s)})
    return dict(schema=1,kind='conditional-ego-command-templates-v1',approved=False,executable=False,
        conditional_variant='EGO',selected_variant=None,initial_sha256=None,training_seed=rc.TRAINING_SEED,
        graph_sha256=__import__('hashlib').sha256(package.encode(graph)).hexdigest(),
        bindings={k:None for k in tokens},pinned_input_hashes=PINNED,jobs=jobs,
        policy='Templates only. Coordinator supplies evidence and admits each frontier separately. No shell, automatic review, execution, allocation or promotion. Initializer-only first; unknown future hashes cannot be filled speculatively.')

def validate(spec):
    rc.require(rc.exact(spec,specification()),'Command template differs from frozen topology/argv')
    return spec

def ancestors(spec,identity):
    by={j['id']:j for j in spec['jobs']};seen=set()
    def visit(i):
        for d in by[i]['depends_on']:
            if d not in seen:seen.add(d);visit(d)
    visit(identity);return seen

def render(spec,identity,bindings,reviewed):
    """Render argv only, never a supervisor admission. reviewed maps ancestors to receipts.

    Coordinator must verify actual receipt content/scientific decisions; a digest
    here is a transport binding, not an independent authorization implementation.
    """
    validate(spec);job=next(j for j in spec['jobs'] if j['id']==identity)
    rc.require(bool(job.get('commands')),'External coordinator review has no executable command')
    for parent in ancestors(spec,identity):
        rc.require(re.fullmatch('[0-9a-f]{64}',reviewed.get(parent,'') or '') is not None,'Missing reviewed prerequisite: '+parent)
    def replace(match):
        key=match[1];value=bindings.get(key)
        rc.require(isinstance(value,str) and value and '\x00' not in value and not TOKEN.search(value),'Unresolved runtime binding: '+key)
        if key in PINNED:rc.require(value==PINNED[key],'Frozen input hash differs: '+key)
        if key.endswith('_sha256'):rc.require(re.fullmatch('[0-9a-f]{64}',value) is not None,'Invalid hash binding: '+key)
        return value
    return [[TOKEN.sub(replace,s) for s in c] for c in job['commands']]

def initializer_plan(spec,bindings,reviewed,cwd,deadline,session_deadline):
    """Pure plan construction after coordinator admission; no launch or approval."""
    rc.require(Path(cwd).is_dir(),'Existing isolated initializer cwd required')
    rc.require(type(deadline) in (int,float) and type(session_deadline) in (int,float) and
               0<deadline<=session_deadline,'Invalid lease/session deadlines')
    command=render(spec,'no-update-initialization',bindings,reviewed)[0]
    command=[bindings['python'],'-c','import subprocess,sys; subprocess.run(sys.argv[1:],check=True,timeout=60)',*command]
    return dict(schema=1,max_parallel_loops=1,gpu_count=1,deadline=deadline,
        session_deadline=session_deadline,jobs=[dict(id='replication-no-update-initialization',
        loop='replication-initialization',stage='test',resource='gpu',cwd=str(Path(cwd).resolve()),
        depends_on=[],command=command)],
        external_prerequisite_receipts={k:reviewed[k] for k in sorted(ancestors(spec,'no-update-initialization'))})


def estimates(spec):
    value=package.estimates(package.graph())
    value['runtime']['qualified_observations']={'original_EGO_train_seconds':314.9,'original_EGO_encoder_wall_seconds':321.9,'original_EGO_dev_64_games_seconds':[42,44],'scope':'Original EGO only; encoder wall overlaps training, never add them. Baseline and new-seed replication unmeasured.'}
    value['runtime']['optimistic_non_admission_pair_plus_10_bouts_seconds']=2*321.9+10*44
    value['runtime']['unknowns']=['Baseline throughput','Fresh seed behavior','Full128 references and serialization','CUDA initialization/resume gates','Setup and payload transfer','Recovery/decode/archive/cleanup and bandwidth contention']
    value['storage']['payload_bytes']=None
    value['storage']['full128_reference_and_training_outputs_bytes']=None
    value['storage']['archive_quota_available_bytes']=None
    value['storage']['remote_peak_measured_bytes']=None
    value['storage']['admission_status']='BLOCKED until exact payload, collection, outstanding work, recovery/archive/scratch and free/quota evidence are bound to final commands; provisional bounds are not approvals.'
    return value

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);spec=validate(specification())
    (a.output/'command-templates.json').write_bytes(package.encode(spec))
    (a.output/'runtime-storage-requirements.json').write_bytes(package.encode(estimates(spec)))

if __name__=='__main__':main()
