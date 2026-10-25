#!/usr/bin/env python3
"""Paired development readout for N arms against one Baseline2048 reference (session-29/30 three-arm plan:
D99_2048, CB2048, REP2048).

Generalizes iteration28-next-plan/readout_2arm.py. Every bout is audited by the UNCHANGED public strict audit_bout
(64 games, contiguous seeds, alternating seats, raw records == summary). Each arm is compared with the SAME Baseline2048
reference bouts using analyze()'s paired adjacent-block bootstrap (32 two-game blocks, 10000 draws, linear percentile
quantiles). Each arm gets a fresh random.Random(bootstrap_seed) consuming E1024 then F1024, so an arm's numbers do not
depend on which other arms are present, and a single-arm run reproduces readout_2arm.py exactly.

Per arm the script reports: counts, paired win-fraction difference with 95% interval, cutoff counts, the frozen development
gate (strictly more wins than Baseline2048 vs BOTH E1024 and F1024 AND <= 12/64 cutoffs on each; OwnInitial descriptive),
and a manifest-based recipe-parity check: every training argument in <training>/manifest.json must equal the Baseline2048
manifest except the arm's ONE declared difference (--declared ARM=KEY: discount for D99, timeout_bootstrap for CB, seed for
REP). `output` (a path) is ignored; `max_seconds` (wall-clock bound) is informational provided report.json shows the
requested run completed with 2048 games and games*updates_per_game optimizer updates. Any other differing argument key,
or a differing config/source_commit/source file hash/parameter count, FAILS parity. The declared key must actually differ.
Evaluation-time inference config (protocol models[0].config) may differ from the baseline only in the declared key.

--replication ARM marks a same-recipe/new-seed arm: its declared key must be `seed`, its inference config must be identical
to the baseline, and the output states explicitly that its paired difference estimates seed-to-seed variability of the
baseline recipe itself (not an intervention effect).

Usage:
  python3 readout_3arm.py --baseline Baseline2048=ROOT --baseline-manifest BASE/manifest.json \
      --arm D99_2048=ROOT_B --declared D99_2048=discount \
      --arm CB2048=ROOT_A --declared CB2048=timeout_bootstrap \
      --arm REP2048=ROOT_C --declared REP2048=seed --replication REP2048 \
      --e 95010000 --f 95020000 --own 95030000 --bootstrap-seed 95050000 --json OUT.json --md OUT.md
ROOT holds `<ARM>-E1024/results`, `<ARM>-F1024/results`, `<ARM>-OwnInitial/results` (supervisor layout, e.g.
/workspace/runs/loop-c or its recovered artifact copy). The arm's training directory defaults to ROOT/<ARM>
(manifest.json + report.json); override with --training ARM=DIR.
Exit codes: 0 all arms audited (gate verdicts inside the JSON); 1 audited but at least one parity check failed;
2 any missing/incomplete/inconsistent input (nothing is dropped silently).
Standard library only; CPU only; no GPU, no network, no pods.
"""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

LIB = Path(__file__).resolve().parent
sys.path.insert(0, str(LIB))
from readout import audit_bout, quantile  # noqa: E402  (unchanged public strict audit)

OPP_SHA = {'E1024': '18c18151b820636ef3bf67ea52dbecd3cb47582d9ea6bf63ede36d6c2adb1532',
           'F1024': '4de08ce08adef8344d6cf8707e22a74896c355008ffff18d3b10ce21ec1dcdb7'}
SAMPLING = 'independent PCG64 per game and seat; no root noise'
DRAWS = 10000
CUTOFF_LIMIT = 12
GATE_RULE = ('Candidate strictly more wins than Baseline2048 against BOTH E1024 and F1024 AND candidate cutoffs <= 12/64 '
             'on each; OwnInitial descriptive only, no outcome reclassification.')
PATH_KEYS = ('output',)                 # never part of the recipe
INFORMATIONAL_KEYS = ('max_seconds',)   # wall-clock bound; listed, not a failure, when the run completed
STRICT_MANIFEST_FIELDS = ('source_commit', 'parameters')   # must be identical
RUNTIME_FIELDS = ('torch', 'python', 'environment', 'gpu')  # listed only
# declared training-argument key -> keys it is allowed to change in manifest.config and in the evaluation inference config
CONFIG_KEYS_FOR = {'discount': {'discount'}}


class ReadoutError(Exception):
    pass


def fail(msg):
    raise ReadoutError(msg)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path, what):
    path = Path(path)
    if not path.is_file():
        fail(f'{what}: missing {path}')
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        fail(f'{what}: invalid JSON in {path}: {exc}')


def parse_binding(text, what):
    if '=' not in text:
        fail(f'{what} must be NAME=VALUE, got {text!r}')
    name, value = text.split('=', 1)
    if not name or not all(c.isalnum() or c in '._-' for c in name) or name == 'baseline':
        fail(f'invalid arm name {name!r}')
    return name, value


def load_bout(root, arm, opponent):
    d = Path(root) / f'{arm}-{opponent}' / 'results'
    if not d.is_dir():
        fail(f'{arm} vs {opponent}: missing bout directory {d}')
    protocol = load_json(d / 'protocol.json', f'{arm} vs {opponent}')
    try:
        bout = audit_bout(d, protocol)
    except (ValueError, KeyError, TypeError, FileNotFoundError, json.JSONDecodeError) as exc:
        fail(f'{arm} vs {opponent}: strict audit failed in {d}: {exc}')
    bout['directory'] = str(d)
    return bout


def check_bout_protocols(name, bouts, baseline, seeds, allowed_config_keys):
    """Bind seeds/opponents/sampling; require identical evaluator/environment/runtime; inference config may differ only
    in allowed_config_keys (reported)."""
    notes, config_diffs = [], {}
    for o in seeds:
        pr = bouts[o]['protocol']
        if pr['seed'] != seeds[o] or pr['models'][1]['id'] != o:
            fail(f'{name} vs {o}: wrong seed/opponent (seed {pr["seed"]}, opponent {pr["models"][1]["id"]})')
        if o in OPP_SHA and (pr['models'][1]['sha256'] != OPP_SHA[o] or pr['models'][1]['config']['discount'] != 0.99):
            fail(f'{name} vs {o}: wrong frozen opponent weights/discount')
        if pr['sampling'] != SAMPLING or pr['games'] != 64 or pr['limit'] != 256:
            fail(f'{name} vs {o}: wrong sampling/games/limit')
        if baseline[o]['protocol']['models'][1] != pr['models'][1]:
            fail(f'{name} vs {o}: opponent entry differs from the baseline bout')
        mb, mc = baseline[o]['protocol']['models'][0], pr['models'][0]
        diff = sorted(k for k in set(mb['config']) | set(mc['config']) if mb['config'].get(k) != mc['config'].get(k))
        for k in ('family', 'root_candidates', 'trained'):
            if mb.get(k) != mc.get(k):
                fail(f'{name} vs {o}: inference model field {k} differs from baseline ({mb.get(k)!r} vs {mc.get(k)!r})')
        if diff:
            config_diffs[o] = {k: dict(baseline=mb['config'].get(k), arm=mc['config'].get(k)) for k in diff}
            extra = sorted(set(diff) - allowed_config_keys)
            if extra:
                fail(f'{name} vs {o}: inference config differs from baseline in {diff}; only {sorted(allowed_config_keys)} allowed for this arm')
            notes.append(f'{o}: {name} inference config differs from baseline in {diff} (consistent with the declared difference; flagged)')
        sig = lambda p: {k: v for k, v in p.items() if k not in ('models', 'seed', 'catalog_sha256')}  # noqa: E731
        if sig(baseline[o]['protocol']) != sig(pr):
            fail(f'{name} vs {o}: evaluator/environment/runtime signature differs from the baseline bout')
    models = [bouts[o]['protocol']['models'][0] for o in seeds]
    if any(m != models[0] for m in models[1:]):
        fail(f'{name}: arm model entry changed across its own bouts (different checkpoint or config)')
    return notes, config_diffs


def paired(bouts, baseline, bootstrap_seed):
    rng = random.Random(bootstrap_seed)
    per = {}
    for o in ('E1024', 'F1024'):
        diffs = [c - b for b, c in zip(baseline[o]['scores'], bouts[o]['scores'])]
        blocks = [(diffs[i] + diffs[i + 1]) / 2 for i in range(0, 64, 2)]
        samples = [sum(rng.choice(blocks) for _ in range(32)) / 32 for _ in range(DRAWS)]
        per[o] = dict(baseline=baseline[o]['counts'], arm=bouts[o]['counts'],
                      baseline_win_fraction=baseline[o]['win_fraction'], arm_win_fraction=bouts[o]['win_fraction'],
                      wins_minus_baseline=bouts[o]['counts']['wins'] - baseline[o]['counts']['wins'],
                      paired_win_fraction_difference=sum(diffs) / 64,
                      percentile95=[quantile(samples, .025), quantile(samples, .975)], block_differences=blocks,
                      input_sha256=dict(baseline=baseline[o]['input_sha256'], arm=bouts[o]['input_sha256']),
                      directories=dict(baseline=baseline[o]['directory'], arm=bouts[o]['directory']))
    return per


def gate(per):
    checks = {o: dict(strict_win_improvement=per[o]['arm']['wins'] > per[o]['baseline']['wins'],
                      candidate_cutoffs_le12=per[o]['arm']['cutoffs'] <= CUTOFF_LIMIT) for o in ('E1024', 'F1024')}
    reasons = []
    for o, c in checks.items():
        if not c['strict_win_improvement']:
            reasons.append(f'{o}: wins {per[o]["arm"]["wins"]} not > baseline {per[o]["baseline"]["wins"]}')
        if not c['candidate_cutoffs_le12']:
            reasons.append(f'{o}: cutoffs {per[o]["arm"]["cutoffs"]} > {CUTOFF_LIMIT}')
    return dict(passed=all(all(v.values()) for v in checks.values()), checks=checks, failure_reasons=reasons, rule=GATE_RULE)


def manifest_parity(name, training_dir, base, base_path, declared, role):
    """All training arguments identical to the Baseline2048 manifest except `declared`; list and fail on anything else."""
    training_dir = Path(training_dir)
    cand = load_json(training_dir / 'manifest.json', f'{name} training manifest')
    report = load_json(training_dir / 'report.json', f'{name} training report')
    ca, ba = cand.get('arguments'), base.get('arguments')
    if not isinstance(ca, dict) or not isinstance(ba, dict):
        fail(f'{name}: manifest.arguments missing in candidate or baseline manifest')
    if declared not in ba:
        fail(f'{name}: declared difference {declared!r} is not a Baseline2048 training argument')
    keys = sorted(set(ca) | set(ba))
    differing = {k: dict(baseline=ba.get(k, '<absent>'), arm=ca.get(k, '<absent>')) for k in keys if ca.get(k, '<absent>') != ba.get(k, '<absent>')}
    declared_diff = {k: v for k, v in differing.items() if k == declared}
    path_diff = {k: v for k, v in differing.items() if k in PATH_KEYS}
    informational = {k: v for k, v in differing.items() if k in INFORMATIONAL_KEYS}
    undeclared = {k: v for k, v in differing.items() if k not in (declared,) + PATH_KEYS + INFORMATIONAL_KEYS}
    allowed_cfg = CONFIG_KEYS_FOR.get(declared, set())
    cc, bc = cand.get('config') or {}, base.get('config') or {}
    config_diff = {k: dict(baseline=bc.get(k), arm=cc.get(k)) for k in sorted(set(cc) | set(bc)) if cc.get(k) != bc.get(k)}
    config_undeclared = sorted(set(config_diff) - allowed_cfg)
    other = {}
    for key in STRICT_MANIFEST_FIELDS:
        if cand.get(key) != base.get(key):
            other[key] = dict(baseline=base.get(key), arm=cand.get(key))
    cf, bf = (cand.get('source_provenance') or {}).get('files', {}), (base.get('source_provenance') or {}).get('files', {})
    source_file_diffs = sorted(k for k in set(cf) | set(bf) if cf.get(k) != bf.get(k))
    if source_file_diffs:
        other['source_provenance.files'] = source_file_diffs
    runtime = {k: dict(baseline=base.get(k), arm=cand.get(k)) for k in RUNTIME_FIELDS if cand.get(k) != base.get(k)}
    expected_updates = (ca.get('games') or 0) * (ca.get('updates_per_game') or 0)
    completed = (report.get('requested_run_completed') is True and report.get('games') == ba.get('games')
                 and report.get('optimizer_updates') == expected_updates)
    reasons = []
    if not declared_diff:
        reasons.append(f'declared difference {declared!r} does NOT differ from the baseline ({ba.get(declared)!r}); the arm is not the declared variant')
    if undeclared:
        reasons.append(f'undeclared training-argument differences: {sorted(undeclared)}')
    if config_undeclared:
        reasons.append(f'manifest.config differs beyond the declared key: {config_undeclared}')
    if other:
        reasons.append(f'strict manifest fields differ: {sorted(other)}')
    if not completed:
        reasons.append(f'report.json does not show a completed run with {ba.get("games")} games and {expected_updates} optimizer updates')
    if informational and not completed:
        reasons.append(f'informational keys differ on an incomplete run: {sorted(informational)}')
    if role == 'replication' and declared != 'seed':
        reasons.append('a replication arm must declare `seed` as its only difference')
    statement = (f'{name} is a replication of the Baseline2048 recipe with a new training seed ({ba.get("seed")} -> {ca.get("seed")}). '
                 'Its paired difference versus Baseline2048 estimates the seed-to-seed variability of the baseline recipe itself '
                 '(training-seed noise), NOT an intervention effect. Candidate effects of similar magnitude are within seed noise.'
                 if role == 'replication' else
                 f'{name} differs from the Baseline2048 recipe only in `{declared}` ({ba.get(declared)!r} -> {ca.get(declared)!r}); '
                 'its paired difference versus Baseline2048 estimates the effect of that single change on one training seed.')
    return dict(passed=not reasons, reasons=reasons, declared_difference=declared, role=role,
                baseline_manifest=str(base_path), baseline_manifest_sha256=digest(base_path),
                arm_manifest=str(training_dir / 'manifest.json'), arm_manifest_sha256=digest(training_dir / 'manifest.json'),
                arm_report_sha256=digest(training_dir / 'report.json'),
                differing_argument_keys=sorted(differing), declared=declared_diff, path_keys=path_diff,
                informational=informational, undeclared=undeclared, config_differences=config_diff,
                config_undeclared=config_undeclared, strict_field_differences=other, runtime_differences=runtime,
                run_completed=completed, expected_optimizer_updates=expected_updates,
                arm_report={k: report.get(k) for k in ('games', 'optimizer_updates', 'training_examples', 'elapsed_seconds', 'checkpoint_sha256', 'requested_run_completed')},
                statement=statement)


def seed_noise_comparison(arms):
    reps = {n: a for n, a in arms.items() if a['role'] == 'replication'}
    cands = {n: a for n, a in arms.items() if a['role'] == 'candidate'}
    rows = {n: {o: dict(diff=a['per_opponent'][o]['paired_win_fraction_difference'], ci=a['per_opponent'][o]['percentile95'],
                        wins_minus_baseline=a['per_opponent'][o]['wins_minus_baseline'], cutoffs=a['per_opponent'][o]['arm']['cutoffs'])
                for o in ('E1024', 'F1024')} for n, a in arms.items()}
    flags = {}
    for rn, r in reps.items():
        for cn in cands:
            f = {}
            for o in ('E1024', 'F1024'):
                rd, cd = rows[rn][o]['diff'], rows[cn][o]['diff']
                lo, hi = r['per_opponent'][o]['percentile95']
                f[o] = dict(candidate_diff=cd, replication_diff=rd, replication_interval=[lo, hi],
                            candidate_inside_replication_interval=lo <= cd <= hi,
                            candidate_abs_exceeds_replication_abs=abs(cd) > abs(rd),
                            wins_gap=dict(candidate=rows[cn][o]['wins_minus_baseline'], replication=rows[rn][o]['wins_minus_baseline']))
            f['within_seed_noise'] = all(f[o]['candidate_inside_replication_interval'] or not f[o]['candidate_abs_exceeds_replication_abs'] for o in ('E1024', 'F1024'))
            flags[f'{cn}_vs_{rn}'] = f
    return dict(rows=rows, flags=flags, has_replication=bool(reps),
                statement=('Compare each candidate\'s paired_win_fraction_difference (and wins_minus_baseline) per opponent with the replication '
                           'arm\'s paired_win_fraction_difference and its 95% interval. If the candidate\'s difference lies inside the replication '
                           'interval, or |candidate| <= |replication|, the candidate effect is within one-seed training noise on this evidence. '
                           'One replication seed is a single draw of that noise, not a variance estimate; see interpretation.md.'))


def markdown(out):
    L = [f'# Multi-arm paired readout ({out["baseline_arm"]} reference)', '',
         f'Seeds: E1024 {out["seeds"]["E1024"]}, F1024 {out["seeds"]["F1024"]}, OwnInitial {out["seeds"]["OwnInitial"]}; bootstrap seed {out["bootstrap_seed"]}, {DRAWS} draws per opponent per arm.', '',
         '| arm | role | declared | opp | wins | losses | draws | cutoffs | base wins | base cutoffs | paired diff | 95% interval | gate | parity |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for name, arm in out['arms'].items():
        mp = arm['manifest_parity']
        par = 'OK' if mp['passed'] else 'MISMATCH'
        for o in ('E1024', 'F1024'):
            p = arm['per_opponent'][o]; c, b = p['arm'], p['baseline']; g = arm['gate']['checks'][o]
            gs = ('wins>' if g['strict_win_improvement'] else 'wins<=') + ' ' + ('cut<=12' if g['candidate_cutoffs_le12'] else 'cut>12')
            L.append(f'| {name} | {arm["role"]} | {mp["declared_difference"]} | {o} | {c["wins"]} | {c["losses"]} | {c["draws"]} | {c["cutoffs"]} | {b["wins"]} | {b["cutoffs"]} | {p["paired_win_fraction_difference"]:+.6f} | [{p["percentile95"][0]:+.6f}, {p["percentile95"][1]:+.6f}] | {gs} | {par} |')
        oi = arm['own_initial']
        L.append(f'| {name} | {arm["role"]} | {mp["declared_difference"]} | OwnInitial | {oi["arm"]["wins"]} | {oi["arm"]["losses"]} | {oi["arm"]["draws"]} | {oi["arm"]["cutoffs"]} | {oi["baseline"]["wins"]} | {oi["baseline"]["cutoffs"]} | descriptive | - | - | {par} |')
    L += ['', '## Verdicts', '']
    for name, arm in out['arms'].items():
        mp = arm['manifest_parity']
        line = f'- **{name}** ({arm["role"]}, declared `{mp["declared_difference"]}`): development gate {"PASSED" if arm["gate"]["passed"] else "FAILED"}'
        if arm['gate']['failure_reasons']:
            line += ' (' + '; '.join(arm['gate']['failure_reasons']) + ')'
        line += f'; recipe parity {"OK" if mp["passed"] else "MISMATCH"} (differing argument keys: {mp["differing_argument_keys"] or "none"})'
        if mp['reasons']:
            line += '; ' + '; '.join(mp['reasons'])
        if arm['config_difference_notes']:
            line += '; ' + '; '.join(arm['config_difference_notes'])
        L.append(line)
        L.append(f'  - {mp["statement"]}')
    sn = out['seed_noise_comparison']
    if sn['has_replication']:
        L += ['', '## Seed-noise comparison', '']
        for key, f in sn['flags'].items():
            parts = [f'{o}: cand {f[o]["candidate_diff"]:+.6f} vs rep {f[o]["replication_diff"]:+.6f} {[round(x, 6) for x in f[o]["replication_interval"]]} inside={f[o]["candidate_inside_replication_interval"]}' for o in ('E1024', 'F1024')]
            L.append(f'- {key}: ' + '; '.join(parts) + f' -> within seed noise: {f["within_seed_noise"]}')
        L.append(f'- {sn["statement"]}')
    L += ['', f'Scope: {out["scope"]}', '']
    return '\n'.join(L)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--baseline', required=True, help='NAME=ROOT of the Baseline2048 reference bouts')
    p.add_argument('--baseline-manifest', type=Path, required=True, help='Baseline2048 training manifest.json')
    p.add_argument('--arm', action='append', default=[], help='NAME=ROOT of an arm (repeatable)')
    p.add_argument('--declared', action='append', default=[], help='NAME=KEY the one training argument the arm changes (required per arm)')
    p.add_argument('--replication', action='append', default=[], help='arm NAME that is a same-recipe/new-seed replication')
    p.add_argument('--training', action='append', default=[], help='NAME=DIR training directory override (default ROOT/NAME)')
    p.add_argument('--e', type=int, required=True); p.add_argument('--f', type=int, required=True); p.add_argument('--own', type=int, required=True)
    p.add_argument('--bootstrap-seed', type=int, required=True)
    p.add_argument('--json', type=Path, required=True); p.add_argument('--md', type=Path)
    a = p.parse_args(argv)
    try:
        if not a.arm:
            fail('at least one --arm is required')
        if abs(a.e - a.f) < 64 or abs(a.e - a.own) < 64 or abs(a.f - a.own) < 64:
            fail('bout seed ranges overlap')
        seeds = {'E1024': a.e, 'F1024': a.f, 'OwnInitial': a.own}
        bname, broot = parse_binding(a.baseline, '--baseline')
        arms = [parse_binding(x, '--arm') for x in a.arm]
        names = [n for n, _ in arms]
        if len(set(names)) != len(names) or bname in names:
            fail('duplicate arm names')
        declared = dict(parse_binding(x, '--declared') for x in a.declared)
        training = dict(parse_binding(x, '--training') for x in a.training)
        for n in names:
            if n not in declared:
                fail(f'--declared {n}=KEY is required (one training argument the arm changes)')
        for r in a.replication:
            if r not in names:
                fail(f'--replication {r} is not an --arm')
        base_manifest = load_json(a.baseline_manifest, 'baseline manifest')
        baseline = {o: load_bout(broot, bname, o) for o in seeds}
        for o in seeds:
            pr = baseline[o]['protocol']
            if pr['seed'] != seeds[o] or pr['models'][1]['id'] != o or pr['sampling'] != SAMPLING:
                fail(f'{bname} vs {o}: baseline bout seed/opponent/sampling mismatch')
            if o in OPP_SHA and pr['models'][1]['sha256'] != OPP_SHA[o]:
                fail(f'{bname} vs {o}: wrong frozen opponent weights')
        bconf = base_manifest.get('config') or {}
        for o in seeds:
            if baseline[o]['protocol']['models'][0]['config'].get('discount') != bconf.get('discount'):
                fail(f'{bname} vs {o}: baseline bout inference discount differs from the baseline manifest config')
        out_arms = {}
        for name, root in arms:
            root = Path(root)
            role = 'replication' if name in a.replication else 'candidate'
            bouts = {o: load_bout(root, name, o) for o in seeds}
            allowed_cfg = set() if role == 'replication' else CONFIG_KEYS_FOR.get(declared[name], set())
            notes, cdiffs = check_bout_protocols(name, bouts, baseline, seeds, allowed_cfg)
            per = paired(bouts, baseline, a.bootstrap_seed)
            mp = manifest_parity(name, training.get(name, root / name), base_manifest, a.baseline_manifest, declared[name], role)
            out_arms[name] = dict(role=role, root=str(root), declared_difference=declared[name], per_opponent=per, gate=gate(per),
                                  own_initial=dict(baseline=baseline['OwnInitial']['counts'], arm=bouts['OwnInitial']['counts'],
                                                   input_sha256=dict(baseline=baseline['OwnInitial']['input_sha256'], arm=bouts['OwnInitial']['input_sha256'])),
                                  config_difference_notes=notes, inference_config_differences=cdiffs, bouts_verified=3,
                                  manifest_parity=mp)
        out = dict(schema=2,
                   variant='readout_3arm: audit_bout unchanged; per-arm paired block bootstrap with a fresh Random(bootstrap_seed) per arm (E1024 then F1024); '
                           'inference config may differ from the baseline only in the declared key; manifest recipe parity required for every arm',
                   checked_at=time.time(), baseline_arm=bname, baseline_root=str(broot), baseline_manifest=str(a.baseline_manifest),
                   baseline_manifest_sha256=digest(a.baseline_manifest), seeds=seeds, bootstrap_seed=a.bootstrap_seed, draws=DRAWS,
                   uncertainty=dict(method='Paired adjacent two-game balanced-seat block bootstrap; 32 blocks resampled with replacement per opponent; percentile interval, linear quantiles',
                                    rng='Python random.Random(bootstrap_seed) per arm; E1024 then F1024',
                                    limitations='Conditional on these trained weights and fixed spawn geometry; one training seed per arm; cutoffs and draws score zero.'),
                   gate_rule=GATE_RULE, cutoff_limit=CUTOFF_LIMIT, arms=out_arms,
                   summary={n: dict(role=e['role'], declared=e['declared_difference'], gate_passed=e['gate']['passed'], recipe_parity=e['manifest_parity']['passed'],
                                    E1024=dict(diff=e['per_opponent']['E1024']['paired_win_fraction_difference'], ci=e['per_opponent']['E1024']['percentile95'], cutoffs=e['per_opponent']['E1024']['arm']['cutoffs']),
                                    F1024=dict(diff=e['per_opponent']['F1024']['paired_win_fraction_difference'], ci=e['per_opponent']['F1024']['percentile95'], cutoffs=e['per_opponent']['F1024']['arm']['cutoffs']))
                            for n, e in out_arms.items()},
                   seed_noise_comparison=seed_noise_comparison(out_arms),
                   scope='Development readout only: one training seed per arm, frozen opponents, descriptive OwnInitial; not promotion, ladder ranking, or a variance estimate.')
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(out, indent=2) + '\n')
        if a.md:
            a.md.write_text(markdown(out))
        print(json.dumps(out['summary']))
        return 1 if any(not e['manifest_parity']['passed'] for e in out_arms.values()) else 0
    except ReadoutError as exc:
        print(f'READOUT FAILED: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
