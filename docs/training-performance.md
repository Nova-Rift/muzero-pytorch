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
