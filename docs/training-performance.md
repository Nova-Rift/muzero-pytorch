# MuZero training performance journal

## Target and counting contract

Clark requested 10,000 epochs in at most 600 seconds. Pending any clarification,
one epoch means one complete arena rollout, matching earlier usage. Report natural
terminal games and time-limit cutoffs separately: a cutoff is not a completed win.
Always report joint steps, active-agent transitions, optimizer updates, examples
trained, batch size, search budget, parameters, precision, and wall time. Never
rename updates into games or reduce game duration to claim the target.

Latest game: canonical `clarkwhitehead` arena (`sandboxCombat.ts` plus `lana/arena.ts`).
Reward: winner 1, all others 0. Physics, observations, agent capabilities and cutoff
must be pinned in the run manifest. No new survival-environment substitution.

## Measurement protocol

- Separate cold startup/compilation, collection, search, replay preparation, updates,
  checkpoint writes, and total end-to-end elapsed time.
- Record GPU model/VRAM, driver/CUDA/PyTorch, CPU model/core count, GPU hourly price,
  model parameter count, exact code SHA and command.
- Warm up explicitly; include cold start in total and report steady-state separately.
- Benchmark on development seeds; freeze model selection before fresh held-out tests.
- A faster job is not evidence of a better policy. Benchmark equivalence first.

## Optimization ledger

| ID | Change | Status | Behavior / tradeoff | Evidence |
|---|---|---|---|---|
| P001 | Batch independent games and independent agents in neural calls | Planned | More throughput; larger policy lag if updates happen between game batches | Pending |
| P002 | Tensor-backed batched MCTS, no per-agent Python tree objects | Planned | Preserve search budget and single-player reward backup | Pending |
| P003 | Parallel canonical JS environment workers | Planned | Same rules; avoid Python physics reimplementation | Pending |
| P004 | Binary observations instead of JSON float arrays | Planned | Same float32 observations; measure transport parity | Pending |
| P005 | Large replay batches and vectorized unroll targets | Planned | Report optimizer steps and examples, not just games | Pending |
| P006 | GPU mixed precision / compile | Not enabled | Numerical and compilation cost; needs parity checks first | Pending |

No measured speedup or 10-minute success is claimed yet. Previous MuZero pilot used
third-party network code and is historical context only: 1,000 games in 1,682.29 s
on RTX4090, about 0.594 games/s, with 8 searches/depth3 and one update/game. That
cannot be directly compared to this independently written implementation without
matching architecture, data, search and update budgets. Target requires 16.667
games/s, roughly 28x that old end-to-end rate.

## Append-only benchmark records

New entries must include unsuccessful attempts and exact output paths/commits.
No paid training has been started for this new implementation at journal creation.

### Initial CPU validation (2026-10-20)

Host Intel i7-1165G7 (4 cores / 8 threads), PyTorch 2.2.0+cu121, Node22.22.2.
Environment-only random-action benchmark: 64 rollouts, 61 terminal/3 cutoffs,
7,223 joint steps /29,222 active transitions in 0.3947s (162.14 rollouts/s).
Command: `python arena/benchmark_environment.py --site SITE --games 64 --workers 4`.
This is only a simulator/transport ceiling, not MuZero training throughput.

P002 scalar-vs-batched search parity passes at 20 simulations, including static
search loops needed for CUDA graphs. P003/P004 exact float32 observation, alive,
reward and termination parity passes against existing JSON bridge over40steps.
A reduced 4-game/16-step CPU integration smoke run performed4optimizer updates;
all4cut off. It cannot be used as evidence for the10,000-game speed target.

Planned GPU run: RTX5090 or comparable, at most $5 incremental GPU cost and45minutes
allocation. Preserve256decision cutoff,153actions,11seats (legacy seat10 asymmetry),
64-observation history,8MCTS simulations,5-step training unroll,one Adam update/game,
batch32. New implementation has no depth3 cap; this differs from the old pilot.
Record TF32 and CUDA graph choices. No policy quality claim from throughput.
