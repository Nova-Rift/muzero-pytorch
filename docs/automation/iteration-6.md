# Iteration 6 — cross-region duration comparison L

Training and all 576 development games completed. No checkpoint passed the registered development gate; no confirmation, replication, promotion, architecture change or homepage replacement. Pod deletion and empty authenticated account inventory verified.

## Question, provenance and authority

Execute the previously unexecuted K question: does extending one fixed trajectory from 1,024 through 2,048 to 4,096 games improve strength against initialization and frozen E/F? Iteration4 remains a failed allocation, with its report and $3 reservation preserved. Iteration5 was a review-only wait, not failed training. Owner explicitly authorized reviewed continuation and broader region search. All current guidance was acknowledged, including the later $10 aggregate hourly cap for future allocations; this run stayed on its original one-H100/$3.99 plan.

Source e8458129dcbcd3df289941cafd77d94493c3ae6d, isolated branch arena-loop-001-L. [Frozen protocol](iteration-6-protocol.json) pins model/trainer/evaluator/environment/opponents. Original model.py and all scientific settings match K. Only orchestration changed: fresh ephemeral transfer, allocation-relative deadline, and honest confirmation deferral if time is insufficient. This is a from-scratch restart because iteration4 had no trained state; not an exact resume or warm start. Full optimizer/replay/model/NumPy/Torch CPU-CUDA RNG snapshots saved every512 games; recovered locally at milestones and completion. GPU exact-resume equivalence remains untested.

Seed51000000; 4,096 games/updates; hidden448/context64, batch256, replay3blocks, unroll5, learning rate3e-4, Adam weight decay1e-4, discount1, episode returns, 128parallel/fourworkers, 32searches/eightrootcandidates; unchanged timeout bootstrap. Two game players, canonical five0.05s frames per decision, win1/everythingelse0, no scripted training or demonstrations. TF32/device history/CUDA graphs enabled for training.

Each checkpoint plays64 development games against each opponent, balanced32 per seat, seed bases82000000/82010000/82020000, independent per-player RNG, identical sampled noise-free32-search/eight-candidate evaluation and256-action cutoff. These seeds vary action sampling, not fixed spawn geometry. Selection maximizes minimum opponent win fraction, then total wins, fewer timeouts and earlier checkpoint. Every opponent requires >60% wins and <=20% timeouts to advance. Fresh confirmation83000000/83010000/83020000 remains unused; independent training-seed replication is still required before promotion.

## Results

| Challenger | Opponent | Wins | Losses | Timeouts | Draws |
|---|---|---:|---:|---:|---:|
| L1024 | initial | 33 | 17 | 14 | 0 |
| L1024 | E1024 | 16 | 23 | 25 | 0 |
| L1024 | F1024 | 28 | 22 | 14 | 0 |
| L2048 | initial | 26 | 18 | 20 | 0 |
| L2048 | E1024 | 21 | 28 | 15 | 0 |
| L2048 | F1024 | 25 | 24 | 15 | 0 |
| L4096 | initial | 20 | 31 | 13 | 0 |
| L4096 | E1024 | 26 | 24 | 14 | 0 |
| L4096 | F1024 | 34 | 20 | 10 | 0 |

Selection chose L2048, SHA256 ff09496a95a2603b54cc81b94d9de8ef1e00987593881154f1c00584bc89eb39. This is byte-identical to preserved J2048, which supports the repeated trajectory checkpoint, not equality of every unsaved trajectory. The current evaluator uses independent player RNG streams; do not pool the earlier J evaluation outcomes with these results. All three checkpoints failed against initialization, so no confirmation ran. L4096 improved the sampled E/F wins relative to L2048 but regressed against initialization; longer training did not establish broad strength improvement. One trajectory, development selection, small matched matchup samples and fixed geometry limit inference.

[Audited summary and full matchup records](iteration-6-evidence/summary.json). Every game recounted independently, seat balance and opponent-specific discounts checked, selected hash verified, recovered NPZ arrays finite. Training completed4096updates,3538natural endings,558timeouts,552079jointsteps,1104158active transitions and1048576sampled examples in803.242seconds. This does not establish learning from throughput or loss.

## Verification and live metrics

Trainer suite:30passed,2CUDA-only skipped. CPU four-game full versus interrupted/full-state resume reproduced weights with maximum difference0; repeated evaluator outcomes and completed counters matched. Source/environment/opponent hashes matched frozen K. Four launch/recovery gate tests passed. An initial test invocation used the lab cwd and failed imports; corrected repository-cwd run passed, and both logs were preserved. No model settings were changed to resolve that command error.

All32 real aggregate-loss samples were published via the owner-provided training_metrics contract and retained during evaluation. Loss is last-update batch mean, not a block mean or separated policy/value/reward loss. Samples use actual games, updates and elapsed_seconds; no individual wallclock timestamps fabricated. Ordinary bounded scalar publisher and occasional lightweight recovery kept chat/progress working during ~1GB full-state transfers. Training was never stopped to answer sidebar questions. Shared control.py and concurrent Site edits were not modified.

## Hardware, timing, cost and cleanup

Cache-region shortage was not general GPU scarcity. Authenticated search checked52datacenters; H100 SXM80GB offers existed in multiple regions at$3.99/hour. Chose US-GA-2 with same-workload H100 measurements; cheaper A100/4090/L40S alternatives lacked matched workload throughput/VRAM evidence. No claim of universal optimal hardware. Uploaded75,079,364compressed bytes including local Node22.22.2 runtime and pinned source/weights. Verified355files; initial setup149.742seconds. Container disk30GB only; existing cache volume remained unattached and unchanged. No new persistent storage/endpoints/packages.

Actual H10080GB HBM3,81559MiB, driver595.91.07, PyTorch2.8.0+cu128/CUDA12.8. Remote child deadline2200seconds from allocation, local2500seconds, shared45minute guard. Training803.242seconds; nine evaluator processes393.588seconds. Allocation request through confirmed deletion1531.303seconds (25.52minutes), estimated GPU$1.697194. Published container disk rate$0.10/GB/month implies about$0.00177 for30GB over this duration (720-hour month convention); actual invoice/network charges remain unreconciled. Existing separately authorized cache storage is separate. The $3 reservation remains consumed; daily remaining allowance$9. Costs recorded once in the preserved ledger.

Initial artifact poll preceded output directory creation and returned rsync23; later and final recovery passed. Full final state is1,015,145,623bytes. All required artifacts are local; authenticated deletion and zero account pods verified. Private run receipts, launch logs, offers, preflight and cost records are retained outside Git. [Storage price source](https://www.runpod.io/pricing).

## Next evidence-based action

Postrun read-only diagnostic of first-two-game traces found26zero-reward, active-at-final-step, nonabsorbing timeout streams with mean initial target0.453613 (range0.127413–0.792172). [Diagnostic](iteration-6-evidence/timeout-diagnostic.json). Current code explicitly bootstraps truncations; these are continuing-task targets despite finite256-step evaluation. This is a hypothesis about objective alignment, not a proven arithmetic bug or causal explanation.

Next choose an isolated finite-horizon zero-timeout-bootstrap test against the preserved continuing-target baseline, with unchanged reward/physics, preregistration, meaningful local tests, matched evaluation and fresh confirmation seeds. Treat it as a scientific target change; restart or explicitly documented warm start, never exact continuation. Model edits are optional in general; this specific intervention is motivated by repeated duration failure and the trace evidence. No extra allocation occurs in this iteration. No arena roster/model promotion or current-ten leaderboard result is implied.
