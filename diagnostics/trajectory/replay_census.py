"""Bounded, read-only CPU census of self-play replay; no training or checkpoint copy."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def distribution(x):
    x = x.double()
    finite = torch.isfinite(x)
    y = x[finite]
    values, counts = torch.unique(y, return_counts=True)
    return dict(count=x.numel(), nonfinite=int((~finite).sum()),
                minimum=float(y.min()) if y.numel() else None,
                maximum=float(y.max()) if y.numel() else None,
                mean=float(y.mean()) if y.numel() else None,
                histogram={str(float(v)): int(c) for v, c in zip(values, counts)} if len(values) <= 32 else None)


def census_block(block, chunk_rows=1024, lanes_per_game=11):
    if chunk_rows < 1 or lanes_per_game < 1:
        raise ValueError('chunk_rows and lanes_per_game must be positive')
    valid, absorbing = block['valid'], block['absorbing']
    t, lanes = block['actions'].shape
    if lanes % lanes_per_game or valid.shape != (t+1, lanes) or absorbing.shape != valid.shape:
        raise ValueError('incompatible replay shape/layout')
    active = valid.any(0)
    ids = valid.nonzero()
    policies = block['policies']
    support_hist = {}
    entropies, maxima, masses = [], [], []
    negative = nonfinite = 0
    for start in range(0, len(ids), chunk_rows):
        ii = ids[start:start+chunk_rows]
        p = policies[ii[:, 0], ii[:, 1]].double()
        negative += int((p < 0).sum())
        nonfinite += int((~torch.isfinite(p)).sum())
        support, counts = torch.unique((p > 0).sum(1), return_counts=True)
        for v, n in zip(support, counts):
            support_hist[str(int(v))] = support_hist.get(str(int(v)), 0) + int(n)
        # Raw target entropy; normalization is separately measured, never repaired.
        entropies.append(-(torch.where(p > 0, p * p.clamp_min(1e-300).log(), 0)).sum(1))
        maxima.append(p.max(1).values)
        masses.append(p.sum(1))
    entropy = torch.cat(entropies) if entropies else torch.empty(0)
    maxmass = torch.cat(maxima) if maxima else torch.empty(0)
    mass = torch.cat(masses) if masses else torch.empty(0)
    reachable_absorbing = absorbing & active[None]
    r = block['rewards']
    targets = block['targets']
    terminal_entry = valid[:-1] & absorbing[1:]
    games_seen = int(active.reshape(-1, lanes_per_game).any(1).sum())
    cutoffs = int((active & ~absorbing[-1]).reshape(-1, lanes_per_game).any(1).sum())
    return dict(allocated_game_slots=lanes//lanes_per_game, games_with_valid_decisions=games_seen,
        player_trajectories=int(active.sum()), active_player_transitions=int(valid[:-1].sum()),
        final_row_valid=int(valid[-1].sum()), cutoff_games_inferred=cutoffs,
        terminal_games_inferred=games_seen-cutoffs,
        policy=dict(rows=len(ids), support_histogram=support_hist, entropy_nats=distribution(entropy),
                    maximum_mass=distribution(maxmass), row_sum=distribution(mass),
                    row_sum_max_abs_error=float((mass-1).abs().max()) if mass.numel() else None,
                    unnormalized_rows_at_1e_minus_6=int(((mass-1).abs()>1e-6).sum()),
                    negative_elements=negative, nonfinite_elements=nonfinite),
        valid_value_targets=distribution(targets[valid]),
        active_absorbing_value_targets=distribution(targets[reachable_absorbing]),
        valid_rewards=distribution(r[valid[:-1]]),
        terminal_entry_rewards=distribution(r[terminal_entry]),
        consistency=dict(indices_equal_valid_nonzero=torch.equal(block['indices'], valid[:-1].nonzero()),
            valid_and_absorbing=int((valid & absorbing).sum()),
            absorbing_reversals=int((absorbing[:-1] & ~absorbing[1:]).sum()),
            valid_restarts=int((~valid[:-1] & valid[1:]).sum()),
            lost_valid_without_absorbing_except_final=int((valid[:-2]&~valid[1:-1]&~absorbing[1:-1]).sum()),
            invalid_nonzero_rewards=int(((r != 0)&~valid[:-1]).sum()),
            absorbing_nonzero_rewards=int(((r != 0)&absorbing[:-1]).sum()),
            active_absorbing_nonzero_targets=int(((targets != 0)&reachable_absorbing).sum()),
            reward_before_terminal_entry=int(((r != 0)&~terminal_entry).sum()),
            raw_absorbing_rows=int(absorbing.sum()), active_absorbing_rows=int(reachable_absorbing.sum())))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state', type=Path, required=True)
    p.add_argument('--sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--chunk-rows', type=int, default=1024)
    p.add_argument('--lanes-per-game', type=int, default=11)
    a = p.parse_args()
    h = hashlib.sha256()
    with a.state.open('rb') as f:
        for data in iter(lambda:f.read(8*1024*1024), b''):
            h.update(data)
    if h.hexdigest() != a.sha256:
        raise ValueError('state hash mismatch')
    torch.set_num_threads(1)
    state = torch.load(str(a.state), weights_only=True, map_location='cpu', mmap=True)
    results = [census_block(b, a.chunk_rows, a.lanes_per_game) for b in state['replay']]
    # Historical trainer appends in collection order to a bounded deque. Ranges
    # are inferred only from totals and actual retained game slots, not timestamps.
    end = int(state['totals']['games'])
    for b in reversed(results):
        b['inferred_training_game_ordinal_range_1_based'] = [end-b['allocated_game_slots']+1, end]
        end -= b['allocated_game_slots']
    result = dict(schema=1, state_sha256=h.hexdigest(), state_bytes=a.state.stat().st_size,
                  loader_torch_version=str(torch.__version__), original_torch_version=state['torch_version'],
                  totals=state['totals'], blocks_oldest_first=results,
                  scope='Census of retained replay, not all training or sampled loss exposure. Recency assumes historical append-only collection order and specified lane layout. Absorbing statistics exclude never-active padded lanes unless explicitly raw. No causal or strength inference.')
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(blocks=len(results), retained_player_transitions=sum(b['active_player_transitions'] for b in results))))


if __name__ == '__main__':
    main()
