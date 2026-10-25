"""Evaluate one frozen final endpoint with a common original-search controller."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import subprocess
import sys


UPG2_TRAINER = "2e67c2b9b7652ccbe75dbed249708356f38c3608"
AIM_DECODINGS = ("absolute", "relative")  # mirrors arena/aim_decoding.py MODES
# Names of training arguments that --declared-difference may name. The menu is the set of
# trainer flags added behind defaults on top of the frozen trainer; anything else is rejected.
DECLARABLE_ARGUMENTS = frozenset({'aim_decoding', 'mirror_augment', 'timeout_bootstrap', 'discount', 'root_prior_temperature', 'action_temperature'})
# Flag defaults for manifests written before a flag existed; --root-prior-temperature/--action-temperature
# (research/iteration36-explore) default to 1.0, which the trainer test pins as byte-identical to the frozen trainer.
ARGUMENT_DEFAULTS = {'aim_decoding': 'absolute', 'mirror_augment': False, 'root_prior_temperature': 1.0, 'action_temperature': 1.0}


def normalized_arguments(arguments):
    return {**ARGUMENT_DEFAULTS, **arguments}


OPERATIONAL_ARGUMENTS = frozenset({'site','output','resume_state','games','max_seconds'})


def check_declared(declared):
    declared=tuple(declared or ())
    if len(set(declared))!=len(declared):
        raise ValueError('Duplicate declared difference')
    unknown=set(declared)-DECLARABLE_ARGUMENTS
    if unknown:
        raise ValueError('Undeclarable difference: '+', '.join(sorted(unknown)))
    return declared


def compare_arguments(arguments, old_arguments, operational, declared):
    """Reject any training-argument difference outside operational and declared keys."""
    arguments=normalized_arguments(arguments); old_arguments=normalized_arguments(old_arguments)
    skip=set(operational)|set(declared)
    if (not arguments or not old_arguments or
            {k:v for k,v in arguments.items() if k not in skip}!=
            {k:v for k,v in old_arguments.items() if k not in skip}):
        raise ValueError('Training arguments differ beyond declared differences')


def bind_source(manifest, source, declared=()):
    """The --source checkout must be the clean commit whose files trained the candidate."""
    files=manifest['source_provenance']['files']
    source=Path(source).resolve()
    for name, expected in files.items():
        path=(source/name).resolve()
        if not path.is_relative_to(source) or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('Frozen training source changed: '+name)
    receipt=source/'.source-provenance.json'
    if receipt.exists():
        actual=json.loads(receipt.read_text())
        if actual.get('dirty') is not False or actual.get('files')!=files:
            raise ValueError('Frozen source receipt differs')
        commit=actual.get('commit')
    else:
        commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
        if declared and subprocess.check_output(['git','-C',str(source),'status','--porcelain','--untracked-files=no'],text=True).strip():
            raise ValueError('Frozen source checkout is dirty')
    if commit!=manifest['source_commit']:
        raise ValueError('Frozen source commit differs')
    return commit


def validate_source_contract(manifest, original, declared, contract):
    """An admitted contract pins both complete source manifests and exact experimental values.

    This permits a reviewed trainer revision, never an arbitrary source change or cross-source
    resume. The coordinator must freeze the contract before training; it is recorded in results.
    """
    if (not isinstance(contract, dict) or set(contract) != {'schema','mode','candidate','paired','arguments'} or
            type(contract.get('schema')) is not int or contract['schema'] != 1 or contract.get('mode') != 'restart'):
        raise ValueError('Declared variants require a schema 1 restart source contract')
    if not isinstance(contract.get('arguments'), dict) or set(contract['arguments']) != set(declared):
        raise ValueError('Source contract declared arguments differ')
    for key, values in contract['arguments'].items():
        if not isinstance(values, dict) or set(values) != {'candidate','paired'}:
            raise ValueError('Invalid source contract argument values: '+key)
    for side, record in (('candidate', manifest), ('paired', original)):
        expected = contract.get(side, {})
        if expected != {'commit': record['source_commit'], 'files': record['source_provenance']['files']}:
            raise ValueError('Source contract differs: ' + side)
        if (not isinstance(expected['commit'], str) or not re.fullmatch(r'[0-9a-f]{40}', expected['commit']) or
                not isinstance(expected['files'], dict) or not expected['files']):
            raise ValueError('Invalid source contract source identity')
        for name, digest in expected['files'].items():
            if (not isinstance(name, str) or not name or Path(name).is_absolute() or '..' in Path(name).parts or
                    str(Path(name)) != name or not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest)):
                raise ValueError('Invalid source contract file hash')
        args = normalized_arguments(record.get('arguments', {}))
        if args.get('resume_state') or args.get('games') != 2048 or type(args.get('games')) is not int:
            raise ValueError('Source contract requires fresh 2048-game restart')
        if type(args.get('updates_per_game')) is not int or args['updates_per_game'] != 1:
            raise ValueError('Source contract requires one update per game')
        if args.get('seed') != 52731415 or args.get('batch') != 256 or args.get('loss_contract') != 'baseline':
            raise ValueError('Source contract baseline objective differs')
        for key in declared:
            value=contract['arguments'][key][side]
            if type(value) is not type(args.get(key)) or value != args.get(key):
                raise ValueError('Source contract argument value differs: ' + key)
        if args.get('aim_decoding', 'absolute') not in AIM_DECODINGS:
            raise ValueError('Invalid aim decoding')
        if type(args.get('mirror_augment', False)) is not bool:
            raise ValueError('Invalid mirror augmentation flag')
    # These two flags define separate experiments; their composition is not admitted here.
    args = manifest['arguments']
    if args.get('aim_decoding') == 'relative' and args.get('mirror_augment'):
        raise ValueError('Combined relative aim and mirror augmentation is not validated')


def validate_declared(manifest, paired, source, declared, contract):
    """Fresh-endpoint binding used only when differences are declared: the candidate must share the
    paired run's config/environment, be trained by the clean --source checkout, and differ from the
    paired arguments only in the declared keys."""
    declared=check_declared(declared)
    if source is None:
        raise ValueError('Declared differences require the training source')
    original=json.loads((paired/'manifest.json').read_text())
    for field in ('config', 'environment', 'torch', 'python'):
        if not manifest.get(field) or manifest[field]!=original.get(field):
            raise ValueError('Paired run differs: '+field)
    for record in (manifest, original):
        provenance=record.get('source_provenance', {})
        if (not record.get('source_commit') or provenance.get('dirty') is not False or
                provenance.get('commit')!=record['source_commit'] or not provenance.get('files')):
            raise ValueError('Missing or dirty source provenance')
    validate_source_contract(manifest, original, declared, contract)
    compare_arguments(manifest.get('arguments', {}), original.get('arguments', {}), OPERATIONAL_ARGUMENTS, declared)
    bind_source(manifest, source, declared)
    return original


def declared_record(manifest, original, declared):
    arguments=normalized_arguments(manifest.get('arguments', {})); old=normalized_arguments(original.get('arguments', {}))
    return dict(declared_differences=list(declared),
                declared_values={k:dict(candidate=arguments.get(k), paired=old.get(k)) for k in declared},
                candidate_source_commit=manifest['source_commit'], paired_source_commit=original['source_commit'])


def validate_continuation(manifest, paired, source, upg2=False):
    """Bind the new endpoint to its frozen training source and original run."""
    if source is None:
        raise ValueError('Continuation requires frozen training source')
    original=json.loads((paired/'manifest.json').read_text())
    for field in ('config', 'environment', 'torch', 'python', 'source_commit'):
        if not manifest.get(field) or manifest[field]!=original.get(field):
            raise ValueError('Continuation differs: '+field)
    for record in (manifest, original):
        provenance=record.get('source_provenance', {})
        if (provenance.get('dirty') is not False or
                provenance.get('commit')!=record['source_commit'] or not provenance.get('files')):
            raise ValueError('Missing or dirty source provenance')
    files=manifest['source_provenance']['files']
    if files!=original['source_provenance']['files']:
        raise ValueError('Continuation source files differ')
    operational={'site','output','resume_state','games','max_seconds'}
    if upg2:
        operational.add('updates_per_game')
    arguments=manifest.get('arguments', {})
    old_arguments=original.get('arguments', {})
    if (not arguments or not old_arguments or
            {k:v for k,v in arguments.items() if k not in operational}!=
            {k:v for k,v in old_arguments.items() if k not in operational} or
            arguments.get('games')!=(2048 if upg2 else 4096) or
            type(arguments.get('updates_per_game')) is not int or
            arguments.get('updates_per_game')!=(2 if upg2 else 1) or
            (bool(arguments.get('resume_state')) if upg2 else not arguments.get('resume_state')) or
            old_arguments.get('games')!=2048):
        raise ValueError('Continuation training arguments differ')
    if upg2 and (manifest['source_commit'] != UPG2_TRAINER or
            old_arguments.get('updates_per_game') != 1 or type(old_arguments.get('updates_per_game')) is not int or
            old_arguments.get('resume_state') or arguments.get('loss_contract') != 'baseline' or
            arguments.get('batch') != 256 or arguments.get('seed') != 52731415):
        raise ValueError('UPG2 fresh baseline training contract differs')
    source=Path(source).resolve()
    for name, expected in files.items():
        path=(source/name).resolve()
        if not path.is_relative_to(source) or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('Frozen training source changed: '+name)
    receipt=source/'.source-provenance.json'
    if receipt.exists():
        actual=json.loads(receipt.read_text())
        if actual.get('dirty') is not False or actual.get('files')!=files:
            raise ValueError('Frozen source receipt differs')
        commit=actual.get('commit')
    else:
        commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if commit!=manifest['source_commit']:
        raise ValueError('Frozen source commit differs')


def catalog(training,paired,opponent,label,expected_endpoint_games=2048,source=None,expected_endpoint_updates=None,declared=(),source_contract=None):
    report=json.loads((training/'report.json').read_text())
    if type(expected_endpoint_games) is not int or expected_endpoint_games not in (2048,4096):
        raise ValueError('Supported endpoints are 2048 and 4096')
    if expected_endpoint_updates is None:
        expected_endpoint_updates = expected_endpoint_games
    if (type(expected_endpoint_updates) is not int or
            (expected_endpoint_games, expected_endpoint_updates) not in ((2048,2048),(4096,4096),(2048,4096))):
        raise ValueError('Unsupported frozen game/update pair')
    upg2 = (expected_endpoint_games, expected_endpoint_updates) == (2048,4096)
    if (report['requested_run_completed'] is not True or
            type(report['games']) is not int or report['games']!=expected_endpoint_games or
            type(report['optimizer_updates']) is not int or report['optimizer_updates']!=expected_endpoint_updates):
        raise ValueError(f'Frozen{expected_endpoint_games} endpoint incomplete')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if sha(training/'initial.npz')!=sha(paired/'initial.npz'):
        raise ValueError('Paired initialization differs')
    manifest=json.loads((training/'manifest.json').read_text())
    declared=check_declared(declared);original=None
    if declared and (expected_endpoint_games != 2048 or upg2):
        raise ValueError('Declared variants require fresh 2048-game restart')
    if source_contract is not None and not declared:
        raise ValueError('Source contract requires declared differences')
    if expected_endpoint_games==4096 or upg2:
        validate_continuation(manifest,paired,source,upg2)
    elif declared:
        original=validate_declared(manifest,paired,source,declared,source_contract)
    if upg2 and (type(report.get('training_examples')) is not int or
            report['training_examples'] != 4096*256):
        raise ValueError('UPG2 training example count differs')
    weights=training/'weights.npz'
    if sha(weights)!=report['checkpoint_sha256']:raise ValueError('Final checkpoint differs from report')
    if sha(Path(opponent['checkpoint']))!=opponent['sha256']:raise ValueError('Frozen opponent changed')
    entry=dict(id=label,family='independent_muzero',checkpoint=str(weights),sha256=sha(weights),
               config=manifest['config'],root_candidates=8,trained=True)
    # Bind the candidate's own action decoding from its training manifest. Absolute runs produce the
    # historical catalog byte-for-byte; a relative run carries aim_decoding so tournament/run.py decodes
    # only this model's actions egocentrically while the opponent keeps its own (absent = absolute).
    decoding=manifest.get('arguments',{}).get('aim_decoding','absolute')
    if decoding not in AIM_DECODINGS:raise ValueError('Unknown aim decoding in manifest')
    if decoding!='absolute':entry['aim_decoding']=decoding
    if opponent.get('aim_decoding','absolute') not in AIM_DECODINGS:raise ValueError('Unknown opponent aim decoding')
    result={'models':[entry,opponent]}
    # Only a declared comparison adds provenance keys; undeclared catalogs stay byte-identical.
    if declared:
        result.update(declared_record(manifest,original,declared))
        result["source_contract"] = source_contract
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--training',type=Path,required=True)
    p.add_argument('--paired-training',type=Path,required=True);p.add_argument('--opponent',type=Path,required=True)
    p.add_argument('--label',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--site',type=Path,required=True)
    p.add_argument('--expected-endpoint-games',type=int,choices=(2048,4096),default=2048)
    p.add_argument('--expected-endpoint-updates',type=int)
    p.add_argument('--declared-difference',action='append',default=[],metavar='KEY',
                   help='training argument allowed to differ from the paired run (repeatable; '+', '.join(sorted(DECLARABLE_ARGUMENTS))+'). '
                        'Also binds the candidate to the clean --source checkout commit instead of the paired source_commit.')
    p.add_argument('--source-contract',type=Path,help='Frozen exact source/value contract for declared restart variants')
    p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    frozen=catalog(a.training,a.paired_training,json.loads(a.opponent.read_text()),a.label,a.expected_endpoint_games,a.source,a.expected_endpoint_updates,a.declared_difference,
                   json.loads(a.source_contract.read_text()) if a.source_contract else None)
    path=a.output/'catalog.json';path.write_text(json.dumps(frozen,indent=2)+'\n')
    subprocess.run([sys.executable,str(a.source/'tournament/run.py'),'--catalog',str(path),
                    '--site',str(a.site),'--output',str(a.output/'results'),'--device','cuda',
                    '--games','64','--parallel','64','--workers','4','--limit','256','--seed',str(a.seed)],
                   timeout=150,check=True)


if __name__=='__main__':main()
