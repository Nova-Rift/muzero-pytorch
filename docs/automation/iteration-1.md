# Arena loop 001 — iteration 1, candidate H

Completed training and development; **development gate failed**. No confirmation, replication, promotion, or homepage replacement. GPU deleted and authenticated inventory verified absent.

Protocol: [registered JSON](iteration-1-protocol.json). Machine-readable evidence and all 576 balanced-seat game records: [summary](iteration-1-evidence/summary.json).

## Review and bounded change

Stopped G: 2,944 completed games/updates, 2,587 natural endings and 357 timeouts; all 48 recovered NPZ hashes revalidated. G had no development evaluation; its exact 2,048-game F prefix is prior evidence, not improved play. Earlier reward probes showed sparse positive events and terminal-example memorization without transfer. This motivated broader temporal replay coverage, not automatically more epochs or terminal oversampling.

H retains 12 collection blocks instead of three. The old default remains three; only an explicit `--replay-blocks 12` enables H. No existing model.py, architecture, canonical physics, reward, or observation/action space was changed. Two agents use five 0.05-second frames per action; winner reward 1, every other outcome 0. No scripted policy or demonstrations. Training starts from scratch with no resume claim.

Training source commit: `fc8f93c17a6eed92af6b88ebf8a8b57012c99644` on `arena-loop-001-H`. Training diff: one bounded CLI argument and replacement of `deque(maxlen=3)` with `deque(maxlen=a.replay_blocks)`. Separate runner registers deadlines, development selection and conditional confirmation. Source hashes are in the protocol and evidence summary. Original checkout remained unchanged.

## Protocol and checks

Training seed 51000000; 2,048 games and updates; 128 parallel games/four workers; batch 256; hidden 448/context 64; 32 searches/eight root candidates; episode returns; unroll five; TF32 and CUDA graphs. Other optimizer/search settings unchanged from F/G. Independent development seed bases 52000000/52010000/52020000 identify initialization/E/F. Each matchup has 64 games, 32 per seat, 256-decision limit and no root noise. Three fixed checkpoints: 512/1024/2048.

Select highest minimum opponent win fraction, then total wins, then fewer timeouts, then earliest checkpoint. Advance only with >60% wins and <=20% timeouts against every opponent. Confirmation seed bases 53000000/53010000/53020000 (200 games per opponent) remain UNUSED. Confirmation requires each 95% Wilson lower win fraction >0.5 and <=20% timeouts, followed by independent-seed replication before promotion. Replication ranges are registered but unused.

Before allocation: 30 unittest cases, 25 passed/five environment or CUDA skips; paired CPU retention checks and diff/source-hash checks passed. The initial four-game smoke assertion incorrectly expected immediate divergence on eviction; samples can coincide, and it failed. Preserved extended eight-game check remained tensor-identical through game four and diverged thereafter (maximum difference 0.0020118244 at game eight). This verifies retention behavior, not quality. A relative-path preparation failure was corrected before compute. Two additional canonical cadence/agent-count tests passed during collection. No new CUDA parity test was run; the unchanged graph/search implementations retain earlier parity evidence.

## Commands

From the isolated candidate source with the verified cached site and weights:

```sh
python scripts/run_iteration_h.py --site /workspace/experiment/site --output /workspace/arena-loop-001-H --weights /workspace/experiment/weights
```

The runner invokes:

```sh
python arena/train_selfplay.py --site /workspace/experiment/site --output /workspace/arena-loop-001-H --games 2048 --parallel 128 --workers 4 --hidden 448 --context 64 --simulations 32 --batch 256 --unroll 5 --return-mode episode --root-candidates 8 --cadence homepage --agents 2 --policy-floor 0 --noise-fraction .25 --limit 256 --updates-per-game 1 --seed 51000000 --max-seconds 900 --replay-blocks 12 --device cuda --device-history --cuda-graph --tf32
python scripts/evaluate_selfplay.py --site /workspace/experiment/site --run /workspace/arena-loop-001-H --challenger checkpoint-512.npz --reference initial.npz --games 64 --seed 52000000 --output /workspace/arena-loop-001-H/development-512-initial.json --device cuda
```

The evaluation command repeats for the registered three checkpoints and three opponents, using their distinct seed bases and output names. The remote runner installs a 2,200-second process-group deadline before training; training has a 900-second bound. One allocation only, registered before launch under the active local guard, <=45 minutes/$3 and <=$4/hour. Checkpoints/metrics recovered approximately every 40 seconds; pod deleted immediately after final recovery.

## Results

Training: 2,048 games/updates; 1,829 natural endings and 219 timeouts; 265,840 joint steps, 531,680 active-agent transitions, 524,288 sampled training examples; 383.06 seconds. These collection counts are not evidence of policy strength.

All evaluation results are DEVELOPMENT, with timeouts counted as nonwins. Zero simultaneous draws.

| Checkpoint | Opponent | Wins | Losses | Timeouts | Win fraction |
|---|---|---:|---:|---:|---:|
| H512 | initial | 27 | 29 | 8 | 42.19% |
| H512 | E1024 | 6 | 26 | 32 | 9.38% |
| H512 | F1024 | 30 | 23 | 11 | 46.88% |
| H1024 | initial | 34 | 21 | 9 | 53.12% |
| H1024 | E1024 | 9 | 15 | 40 | 14.06% |
| H1024 | F1024 | 29 | 26 | 9 | 45.31% |
| H2048 | initial | 28 | 26 | 10 | 43.75% |
| H2048 | E1024 | 13 | 32 | 19 | 20.31% |
| H2048 | F1024 | 28 | 24 | 12 | 43.75% |

The registered maximin rule selects **H2048**, SHA256 `68b7e7bff8caec085dc49965c9b0bd736116ef33c87724487c599fea80730cf5`. Its weakest matchup is E: 13/64 wins (20.31%) and 19/64 timeouts (29.69%). It fails both gates there and the win gate against every opponent. No confirmation games ran. Selecting a checkpoint for reporting does not make it promotion-eligible.

All 36 recovered NPZ files are finite. Each of the nine result files was independently recounted from its 64 records; counts, seats and candidate hashes agree. Full weights, traces, execution logs, actual receipt and deletion evidence remain private. Only scoped code and evidence are pushed.

## Runtime, costs and limits

H100 80 GB HBM3 (81,559 MiB reported), PyTorch 2.8.0+cu128/CUDA 12.8, driver 595.91.07, Node 22.22.2. Verified cached payload: 301 files; about 2.43 MB sent for a 176.37 MB payload. No packages, storage or endpoints created. Cached physics matched the current canonical website files.

Nine evaluation processes totaled 382.73 seconds. Allocation request through deletion response: 850.90 seconds (14.18 minutes) at $3.99/hour, **$0.943080 estimated GPU cost**. Existing private ledger reconciled once; cumulative recorded compute estimate $11.978695/$50. The $3 pilot reservation remains consumed. Invoice not reconciled; separately authorized cache storage unchanged.

One training seed, fixed arena geometry, reused development seeds across checkpoints, and no matched-seed three-block control limit inference. Development RNG seeds vary action sampling, not map layouts. The data do not establish that retention caused weakness, nor prove superiority, general intelligence, continual learning or a confirmation result. High E timeouts remain unresolved.

## Proposed next iteration

A matched-seed three-block replay control can distinguish retention effects from initialization: change H retention 12 to 3, keep training seed/config/budget fixed, and compare frozen H on fresh development seeds. No longer run or confirmation tuning.

This invocation stops after iteration 1. No additional allocation was launched.
