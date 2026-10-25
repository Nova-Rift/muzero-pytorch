"""Stage a non-executable replication package. Never admit, select, or run jobs.

The graph is a specification, not a workload_supervisor plan. Unknown runtime
bindings and the dedicated collection guard intentionally remain unresolved.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

import replication_contract as rc
import replication_initialization as init
import evaluate_replication as evaluator
import readout_replication as readout

INPUTS = ('protocol.json', 'source-receipt.json', 'seed-audit.json',
          'contract-ego.template.json', 'contract-mir.template.json')
ADAPTERS = ('replication_package.py', 'replication_contract.py', 'replication_initialization.py',
            'evaluate_replication.py', 'readout_replication.py', 'evaluate_endpoint.py',
            'resume_gate.py', 'compressed_checkpoint.py', 'check_contract.py', 'verify_training_states.py',
            'train_compressed.py', 'verify_compressed_states.py')
SEED_AUDIT = '333ed93a68ff026c940a5daee6152c9108c8ca721c3530df263d06489d978c39'


def graph():
    jobs = []
    def add(identity, lane, stage, dependencies, seconds, requirement, **extra):
        jobs.append(dict(id=identity, loop=lane, stage=stage, depends_on=dependencies,
                         device_count=0 if stage=='review' else 1, timeout_seconds=seconds,
                         requirement=requirement, **extra))
    add('original-selection', 'selection', 'review', [], 0,
        'Coordinator verifies original EGO/MIR dev9501/9502/9503 and heldout9701/9702, then selects exactly one passing variant. No arm chosen here.')
    add('conditional-admission', 'selection', 'review', ['original-selection'], 0,
        'Fresh seed audit; published source; implemented/tested replication collection guard; exact job/storage/quote/budget/lease approval before CUDA allocation.')
    add('no-update-initialization', 'initialization', 'test', ['conditional-admission'], 60,
        'Run replication_initialization.py in Python3.12.3/Torch2.8.0+cu128 CUDA against frozen14fece source and selected null template. Preserve contract, initial.npz, receipt and prefix parity.')
    add('initialization-review', 'initialization', 'review', ['no-update-initialization'], 30,
        'Verify exact generator/template/source/contract hashes and no-update prefix parity; freeze output bindings before either trainer starts. CPU evidence is ineligible.')
    for side in ('paired', 'candidate'):
        add(side+'-runtime', side, 'test', ['initialization-review'], 20,
            'Check frozen Python/Torch/CUDA and exact GPU identity against initialization; bind environment/site/evaluator source.')
        add(side+'-resume', side, 'test', [side+'-runtime'], 120,
            'Frozen resume_gate train_loss baseline objective, updates/game1; selected intervention only on candidate. Existing seed23002 is a diagnostic, never replication training evidence.')
        deps=[side+'-resume'] + (['paired-collection'] if side=='candidate' else [])
        add(side+'-collection', side, 'test', deps, 300,
            'Dedicated source-bound fresh-seed collection-only guard REQUIRED. Validate actual trainer initial.npz before collection; zero updates; preserve full128-game first block/aggregates. MIR compares exact paired first block; EGO records changed-policy collection without a false equality claim.',
            implementation=None)
        add(side+'-fresh2048', side, 'train', [side+'-collection','initialization-review'], 980,
            'Fresh actual trainer invocation, same frozen initialization and seed73142017; in-process guard rejects initial mismatch before first collect/update and rechecks first block against own collection-only receipt. Encode/recover full-state milestones; no resume or training recipe changes.',
            recipe=copy.deepcopy(rc.RECIPE), intervention={} if side=='paired' else None,
            trainer_max_seconds=850, encoder_max_seconds=960, max_encoded_bytes=600000000)
    add('exact-pair-and-recovery', 'pair', 'test', ['paired-fresh2048','candidate-fresh2048'], 180,
        'Verify replication_contract, both actual initial exports, all endpoint counters, source and full-state encoded/raw hashes. Preserve supervisor chronology proving init-review before both trainer starts. Recover/decode checkpoint and optimizer/replay/RNG evidence.')
    for phase in ('development', 'confirmation'):
        gate='exact-pair-and-recovery' if phase=='development' else 'development-pass-review'
        evaluations=[]
        for side in ('paired','candidate'):
            previous=gate
            for opponent, seed in evaluator.SEEDS[phase].items():
                identity=f'{side}-{phase}-{opponent}'
                add(identity, side, 'evaluate', [previous], 160,
                    'Exact evaluate_replication adapter, source/catalog/initialization/environment/opponent bindings;64games alternating32/32seats. Preserve catalog, receipt, protocol, summary, raw games.',
                    side=side, opponent=opponent, phase=phase, seed=seed, games=64)
                previous=identity;evaluations.append(identity)
        add(phase+'-readout', 'readout', 'review', evaluations, 60,
            'Pinned readout_replication request with every input hash and evaluation environment. Process success is not scientific pass; inspect gate.passed explicitly.')
        if phase=='development':
            add('development-pass-review', 'readout', 'review', ['development-readout'], 30,
                'Require development gate.passed true (strictly more candidate wins on each E/F/OwnInitial) before ANY confirmation game. False is terminal rejection, not confirmation permission.')
        else:
            add('confirmation-review', 'readout', 'review', ['confirmation-readout'], 30,
                'Candidate Wilson95 lower bound>.5 and cutoffs/64<=.20 on EACH E/F. Baseline comparisons descriptive; all64 denominator. No automatic promotion.')
    return dict(schema=1, kind='conditional-replication-specification-v1', approved=False,
                executable=False, selected_variant=None, initial_sha256=None,
                trainer_commit=rc.SOURCE_COMMIT, seed=rc.TRAINING_SEED,
                gpu_overlap_assumed=False, execution_policy='serial one-device jobs; lane identities are logical, no extra pod implied',
                jobs=jobs)


def validate_graph(value):
    # Exact frozen design rejects bypasses, new seeds/arms, commands and altered gates.
    rc.require(rc.exact(value,graph()), 'Conditional graph differs from reviewed specification')
    seen=set()
    for job in value['jobs']:
        rc.require(job['id'] not in seen and set(job['depends_on']) <= seen, 'Duplicate or forward/cyclic job')
        seen.add(job['id'])
    return value


def estimates(value):
    work=sum(j['timeout_seconds'] for j in value['jobs'])
    setup_recovery_cleanup=1800
    total=int((work+setup_recovery_cleanup)*1.2)
    return dict(approved=False, planning_only=True, graph_sha256=hashlib.sha256(encode(value)).hexdigest(),
        runtime=dict(serial_job_bound_seconds=work, setup_transfer_recovery_cleanup_seconds=setup_recovery_cleanup,
                     contingency_fraction=.2,total_seconds=total,indicative_usd_per_hour=3.99,
                     indicative_cost_upper_usd=round(total/3600*3.99,4),
                     limitation='Unmeasured conservative planning bound, not a fresh quote, reservation or permission. May need renewable segments and session fit check.'),
        storage=dict(recovered_artifacts_bytes=3000000000,archive_growth_bytes=3000000000,
                     scratch_peak_bytes=3000000000,other_outstanding_bytes=None,headroom_bytes=4294967296,
                     remote_scratch_bytes_per_training_lane=17179869184,
                     encoded_full_state_limit_per_side=600000000,
                     raw_full_state_local_recovery=False,
                     basis='Per side:600MB encoded states+400MB weights/traces/evals+200MB resume+300MB collection evidence. One full additional archive copy. 3GB scratch includes two raw1.016GB restore probes plus package/recovery transient allowance. Retain raw full-state remotely inside16GiB/lane; do not recover every raw milestone locally. New full-block collector size requires measurement and revised bound if larger. Current/future outstanding work is unknown, never zero.'),
        admission='This is NOT an approved storage_readiness plan; bind exact executable jobs and measure inputs, collection output, archive quota, outstanding work, free space, remote raw peak and current policy immediately before admission.')


def encode(value):
    return (json.dumps(value,indent=2)+'\n').encode()


def stage(prerequisites, output):
    rc.require(not output.exists(),'Fresh package output required')
    for variant in ('ego','mir'):
        template=json.loads((prerequisites/f'contract-{variant}.template.json').read_text())
        init.validate_template(template)
        rc.require(template['variant']==variant.upper(),'Template variant differs')
    rc.require(rc.digest(prerequisites/'seed-audit.json')==SEED_AUDIT,'Frozen seed audit differs')
    receipt=json.loads((prerequisites/'source-receipt.json').read_text())
    template=json.loads((prerequisites/'contract-ego.template.json').read_text())
    rc.require(receipt.get('dirty') is False and
               rc.exact({k:receipt[k] for k in ('commit','files')},template['source']), 'Source receipt differs')
    protocol=json.loads((prerequisites/'protocol.json').read_text())
    rc.require(protocol['training_seed']==rc.TRAINING_SEED and protocol['selected_variant'] is None and
               protocol['seed_audit_sha256']==SEED_AUDIT,'Prerequisite protocol differs')
    for phase,opponents in evaluator.SEEDS.items():
        for name,seed in opponents.items():
            key=('dev_' if phase=='development' else 'confirmation_')+name
            rc.require(protocol['evaluation_ranges'][key]==[seed,seed+63],'Frozen evaluation range differs')
    readout.verify_helpers()
    value=validate_graph(graph())
    output.mkdir(parents=True)
    (output/'inputs').mkdir();(output/'diagnostics'/'retention').mkdir(parents=True);(output/'diagnostics'/'paired').mkdir()
    for name in INPUTS:shutil.copyfile(prerequisites/name,output/'inputs'/name)
    for name in ADAPTERS:shutil.copyfile(Path(__file__).parent/name,output/'diagnostics'/'retention'/name)
    for name in readout.AUDIT_HELPERS:
        shutil.copyfile(Path(__file__).parent.parent/'paired'/name,output/'diagnostics'/'paired'/name)
    (output/'conditional-jobgraph.json').write_bytes(encode(value))
    (output/'runtime-storage-plan.json').write_bytes(encode(estimates(value)))
    unresolved=dict(original_selection=None, frozen_cuda_initialization=None, actual_trainer_initial_exports=None,
        collection_guard_implementation=None, collection_guard_cuda_results=None, exact_execution_commands=None,
        evaluation_environment=None, opponent_entries=None, evaluator_runtime_results=None, readout_request=None,
        source_publication=None, refreshed_seed_audit=None, storage_admission=None, budget_quote_lease=None,
        note='Null bindings deliberately block execution. Do not modify templates, substitute CPU hashes, reuse original-seed collection evidence, invent successful runtime receipts or infer scientific pass from exit code0.')
    (output/'remaining-gates.json').write_bytes(encode(unresolved))
    hashes={str(p.relative_to(output)):rc.digest(p) for p in sorted(output.rglob('*')) if p.is_file()}
    (output/'package-manifest.json').write_bytes(encode(dict(schema=1,approved=False,files=hashes,
        initializer_sha256=rc.digest(Path(init.__file__)), evaluator_sha256=rc.digest(Path(evaluator.__file__)),
        readout_sha256=rc.digest(Path(readout.__file__)), trainer_commit=rc.SOURCE_COMMIT)))
    return hashes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prerequisites',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();stage(args.prerequisites,args.output)

if __name__=='__main__':main()
