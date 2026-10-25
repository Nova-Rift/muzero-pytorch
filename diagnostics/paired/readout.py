"""Strict frozen-protocol paired readout; standard library CPU only."""
import argparse
import hashlib
import json
from pathlib import Path
import random

BOOTSTRAP_SEED = 92360000
DRAWS = 10000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_bout(directory, expected):
    directory = Path(directory)
    protocol = load(directory / 'protocol.json')
    require(protocol == expected, 'protocol differs from frozen expected protocol')
    for key in ('source_sha256', 'controller_sha256', 'catalog_sha256'):
        require(isinstance(protocol.get(key), str) and len(protocol[key]) == 64, 'missing source/catalog binding')
    require(bool(protocol.get('environment')), 'missing environment binding')
    require(protocol['games'] == 64 and protocol['limit'] == 256, 'wrong game count/limit')
    require(len(protocol['models']) == 2, 'requires exactly two models')
    a, b = [m['id'] for m in protocol['models']]
    files = sorted(directory.glob('*__*.json'))
    require(len(files) == 1 and files[0].name == f'{a}__{b}-0.json', 'missing/extra raw bout files')
    records = load(files[0])['records']
    require(len(records) == 64, 'partial or excess records')
    require(all(type(r['seed']) is int for r in records), 'invalid seed type')
    records = sorted(records, key=lambda r: r['seed'])
    counts = dict(wins=0, losses=0, draws=0, cutoffs=0)
    scores = []
    for i, r in enumerate(records):
        require(r['seed'] == protocol['seed'] + i, 'duplicate/missing/unexpected seed')
        require(type(r['seat_a']) is int and r['seat_a'] == i % 2, 'wrong seat contract')
        require(type(r['finished']) is bool and type(r['winner']) is int and r['winner'] in (-1, 0, 1), 'invalid outcome')
        require(type(r['steps']) is int and 0 <= r['steps'] <= 256, 'invalid steps')
        if not r['finished']:
            require(r['winner'] == -1 and r['steps'] == 256, 'invalid cutoff')
            category = 'cutoffs'
        elif r['winner'] == -1:
            category = 'draws'
        else:
            category = 'wins' if r['winner'] == r['seat_a'] else 'losses'
        counts[category] += 1
        scores.append(int(category == 'wins'))
    summaries = load(directory / 'summary.json')
    require(len(summaries) == 1, 'extra/missing summary bouts')
    summary = summaries[0]
    for key, value in dict(a=a, b=b, games=64, wins_a=counts['wins'], wins_b=counts['losses'], draws=counts['draws'], cutoffs=counts['cutoffs']).items():
        require(summary.get(key) == value, 'summary mismatch: ' + key)
    return {'counts': counts, 'win_fraction': counts['wins']/64, 'seats': [32, 32],
            'scores': scores, 'protocol': protocol,
            'input_sha256': {p.name: digest(p) for p in [directory/'protocol.json', directory/'summary.json', files[0]]}}


def quantile(values, q):
    values = sorted(values)
    position = (len(values)-1)*q
    low = int(position)
    high = min(low+1, len(values)-1)
    return values[low] + (values[high]-values[low])*(position-low)


def analyze(manifest):
    """Manifest requires four bout entries with arm, opponent, directory, protocol."""
    candidate_arm = manifest.get('candidate_arm', 'reference-q')
    require(isinstance(candidate_arm, str) and candidate_arm != 'baseline' and 1 <= len(candidate_arm) <= 64 and all(c.isalnum() or c in '._-' for c in candidate_arm), 'invalid candidate arm')
    seeds = manifest.get('development_seeds', {'E1024':92320000, 'F1024':92330000})
    require(isinstance(seeds, dict) and set(seeds) == {'E1024','F1024'} and all(type(v) is int and v >= 0 for v in seeds.values()), 'invalid development seed binding')
    require(abs(seeds['E1024']-seeds['F1024']) >= 64, 'overlapping development seed ranges')
    bootstrap_seed = manifest.get('bootstrap_seed', BOOTSTRAP_SEED)
    require(type(bootstrap_seed) is int and bootstrap_seed >= 0, 'invalid bootstrap seed')
    bouts = manifest['bouts']
    require(len(bouts) == 4, 'requires four bouts')
    indexed = {}
    for entry in bouts:
        key = (entry['arm'], entry['opponent'])
        require(key not in indexed, 'duplicate bout')
        indexed[key] = audit_bout(entry['directory'], entry['protocol'])
    require(set(indexed) == {(a, o) for a in ('baseline', candidate_arm) for o in ('E1024', 'F1024')}, 'wrong bout identities')
    common = None
    arm_models = {}
    rng = random.Random(bootstrap_seed)
    results = {}
    for opponent, seed in [(o,seeds[o]) for o in ('E1024','F1024')]:
        baseline, candidate = [indexed[(a, opponent)] for a in ('baseline', candidate_arm)]
        for arm, bout in [('baseline', baseline), (candidate_arm, candidate)]:
            p = bout['protocol']
            require(p['seed'] == seed and p['models'][1]['id'] == opponent, 'wrong development seeds/opponent')
            require(p['models'][1]['sha256'] == {'E1024':'18c18151b820636ef3bf67ea52dbecd3cb47582d9ea6bf63ede36d6c2adb1532', 'F1024':'4de08ce08adef8344d6cf8707e22a74896c355008ffff18d3b10ce21ec1dcdb7'}[opponent], 'wrong opponent weights')
            model = p['models'][0]
            require(model['config']['discount'] == 1.0 and model['config']['simulations'] == 32 and model['root_candidates'] == 8, 'wrong arm config')
            require(p['models'][1]['config']['discount'] == 0.99, 'wrong opponent discount')
            require(p['sampling'] == 'independent PCG64 per game and seat; no root noise', 'wrong sampling')
            if arm in arm_models:
                require(arm_models[arm] == model, 'arm model changed across opponents')
            arm_models[arm] = model
            signature = {k:v for k,v in p.items() if k not in ('models', 'seed', 'catalog_sha256')}
            if common is not None:
                require(common == signature, 'incompatible evaluator/environment/runtime/config')
            common = signature
        require(baseline['protocol']['models'][1] == candidate['protocol']['models'][1], 'opponent mismatch')
        ma, mb = [b['protocol']['models'][0] for b in (baseline, candidate)]
        require({k:v for k,v in ma.items() if k not in ('id','sha256')} == {k:v for k,v in mb.items() if k not in ('id','sha256')}, 'arm inference config mismatch')
        differences = [c-b for b,c in zip(baseline['scores'], candidate['scores'])]
        blocks = [(differences[i]+differences[i+1])/2 for i in range(0,64,2)]
        samples = [sum(rng.choice(blocks) for _ in range(32))/32 for _ in range(DRAWS)]
        results[opponent] = {'baseline': {k:v for k,v in baseline.items() if k not in ('scores','protocol')},
                             candidate_arm: {k:v for k,v in candidate.items() if k not in ('scores','protocol')},
                             'paired_win_fraction_difference': sum(differences)/64,
                             'percentile95': [quantile(samples, .025), quantile(samples, .975)], 'block_differences': blocks}
    return {'schema':1, 'scope':'Development evaluation of one paired training seed; not promotion, ranking, or independent training replication.',
            'uncertainty': {'method':'Paired adjacent two-game balanced-seat block bootstrap; 32 blocks resampled with replacement per opponent; percentile interval, linear quantiles', 'seed':bootstrap_seed, 'draws':DRAWS, 'opponent_order':['E1024','F1024'], 'rng':'Python random.Random; statistical resampling only', 'limitations':'Conditional on these trained weights and fixed spawn geometry. Does not estimate training-seed uncertainty. Cutoffs and draws score zero.'},
            'per_opponent':results,
            'secondary_equal_opponent_difference':sum(r['paired_win_fraction_difference'] for r in results.values())/2}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = analyze(load(args.manifest))
    result['manifest_sha256'] = digest(args.manifest)
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
