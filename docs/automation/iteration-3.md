# Arena loop 001 — iteration 3, candidate J

**Training and development completed; development gate failed.** No confirmation, replication, promotion or homepage replacement. GPU deletion verified. This is the third and final pilot slot.

Registered protocol: [iteration-3-protocol.json](iteration-3-protocol.json).

## Question and intervention

H and I failed their development gates. I's saved training traces contained 27 winning streams with median winning step 102 and mean initial discounted target 0.388 despite realized reward 1. This motivated one bounded test: discount .99 to 1.0 in value targets and search. It did not establish a target arithmetic bug or prove why prior policies failed.

J starts from scratch with I's training seed 51000000, 2,048 games and optimizer updates, three replay blocks, batch 256, hidden 448/context 64, 32 searches/eight root candidates, episode returns and unchanged optimizer settings. Two agents use the canonical five 0.05-second frames per action. Winner reward remains 1; losses, draws and timeouts earn zero. No scripted policy, demonstrations, architecture changes or model.py edits. Existing timeout bootstrap semantics remain unchanged: zero realized timeout reward does not mean the training bootstrap is forced to zero.

Source commit `75c7b9b00b2de2525dc242bc4a021ce3372547a3` on isolated branch `arena-loop-001-J`. The trainer adds an explicit discount option, default .99. The evaluator adds per-player discount overrides, because applying J's configuration to both players would alter frozen opponents. E/F/I retain .99; J and its matching initial policy use 1.0. Frozen I's initial opponent uses .99. Thus E/F are shared behavioral anchors; the initial-policy controls share tensors but differ in their registered search discount. All other evaluator behavior remains unchanged. Original worktrees and website are untouched.

## Preregistration and checks

Checkpoints 512/1024/2048 each play 64 development games against initialization, E1024 and F1024, 32 games per seat, seed bases 72000000/72010000/72020000. Select highest minimum opponent win fraction, then total wins, fewer timeouts, earliest checkpoint. Advance only with >60% wins and <=20% timeouts against every opponent. Freeze hash before conditional confirmation: 200 games/opponent at 73000000/73010000/73020000, requiring each 95% Wilson lower win fraction >.5 and <=20% timeouts. Independent seed replication is additionally required before promotion; none is automatically authorized beyond this final pilot slot.

Frozen I1024 plays the same development seed bases and three opponents; selected J plays I at 72030000. These 256 control games are excluded from selection. Training never consumes evaluation games. Confirmation cannot trigger reselection.

Before allocation: 32 unit tests, 30 passed and two CUDA-only skips. Tests cover undiscounted winner targets and scalar/batched/frozen search parity. Two-game CPU training/evaluation checks verify CLI, checkpoint and mixed-discount evaluation integration; these are not quality evidence. Source, physics and anchor hashes passed. No package installation. One local status/compile command initially used the wrong directory and was corrected before allocation.

## Commands and controls

```sh
ARENA_SITE=VERIFIED_CACHED_SITE python3 -m unittest discover -s tests
python scripts/run_iteration_j.py --site /workspace/experiment/site --output /workspace/arena-loop-001-J --weights /workspace/experiment/weights
```

The committed runner contains exact training/evaluation commands. Training matches I except output and explicit `--discount 1.0`. The remote runner enforces a 2,200-second child-process deadline, with training bounded at 900 seconds. Each training/evaluation process waits for local owner-control review. The local monitor reads guidance and PAUSED/STOPPED on progress updates, recovers artifacts repeatedly and deletes the registered pod in its finalizer. One allocation, <=45 minutes/$3 at <=$4/hour; no additional storage/endpoints.

Acknowledged owner guidance entry 1792563591.7862878: conversational control, guidance at safe boundaries, stop/pause precedence and winner-only rewards. No additional steering was present.

## Results

All 832 evaluation games are DEVELOPMENT: 576 selection games and 256 frozen-I control games. Each row has 64 games balanced 32 per seat; all timeouts and draws count as nonwins. [All records and audited summary](iteration-3-evidence/summary.json).

| Challenger | Opponent | Wins | Losses | Timeouts | Draws |
|---|---|---:|---:|---:|---:|
| J512 | initial | 29 | 30 | 5 | 0 |
| J512 | E1024 | 11 | 26 | 27 | 0 |
| J512 | F1024 | 26 | 30 | 8 | 0 |
| J1024 | initial | 29 | 20 | 15 | 0 |
| J1024 | E1024 | 11 | 24 | 29 | 0 |
| J1024 | F1024 | 23 | 29 | 12 | 0 |
| J2048 | initial | 33 | 17 | 14 | 0 |
| J2048 | E1024 | 20 | 33 | 11 | 0 |
| J2048 | F1024 | 26 | 22 | 16 | 0 |
| I1024 | initial | 24 | 23 | 17 | 0 |
| I1024 | E1024 | 17 | 31 | 16 | 0 |
| I1024 | F1024 | 27 | 29 | 8 | 0 |
| Selected J2048 | I1024 | 31 | 22 | 11 | 0 |

The registered selection chooses **J2048**, SHA256 `ff09496a95a2603b54cc81b94d9de8ef1e00987593881154f1c00584bc89eb39`. It wins 33/64 (51.56%) against initialization, 20/64 (31.25%) against E and 26/64 (40.63%) against F. It fails the win gate against all three, and the timeout gate against initialization (21.88%) and F (25%). Confirmation seed ranges remain unused. Frozen selection is for reporting, not promotion eligibility.

Against fixed E/F anchors, selected J differs from frozen I by +3/-1 wins on these samples. Direct J versus I is 31 wins, 22 losses and 11 timeouts: 48.44% wins counting timeouts as nonwins. These exploratory samples do not establish a reliable advantage. Selection across three J checkpoints and one previously selected I checkpoint also limits causal comparisons. Initial opponents share identical tensors, but differ in search discount as described above.

Training completed 2,048 games/updates, 1,782 natural endings, 266 timeouts, 275,584 joint steps, 551,168 active transitions and 524,288 sampled examples in 411.32 seconds. Initial tensors and their file hash match I exactly. All 24 winning streams in the first-two-game trace samples have pre-win targets exactly 1, confirming the intervention. This is implementation evidence, not policy-quality evidence.

All 37 recovered NPZ files are finite. Independent audits recounted every evaluation record, checked balanced seats, candidate hashes and each player's actual search discount. Frozen selection hash matches. No confirmation files exist.

## Runtime, cost and recovery

H100 80 GB HBM3 (81,559 MiB reported), driver 595.91.07, PyTorch 2.8.0+cu128/CUDA 12.8 and Node 22.22.2. Verified 330 cached payload files, about 42.01 MB total with 12.62 MB transferred. No package, storage or endpoint creation. One initial artifact poll preceded creation of the training output and returned rsync code 23; later recovery and final audits passed. No job retry or second allocation occurred.

Thirteen evaluation processes totaled 559.01 seconds. Allocation request through confirmed deletion took 1202.89 seconds (20.05 minutes) at $3.99/hour: **$1.333199 estimated GPU cost**. Existing private ledger reconciled once; cumulative recorded compute $14.628546/$50. The $3 reservation remains consumed. Actual invoice is unreconciled; existing cache storage remains separate. Authenticated inventory verified this pod absent immediately after recovery.

## Limitations and next hypothesis

One training seed, fixed geometry and development selection provide no confirmation, replication, general-intelligence, continual-learning or superhuman claim. Seeds vary action sampling, not layouts. Discount changes both training targets and search behavior; this experiment does not separate those effects. Timeout bootstrapping retains the historical continuing-task treatment despite the evaluation's finite cutoff.

A post-run read-only diagnostic on training traces found 16 zero-reward surviving streams at the 256-action cutoff, with mean initial target 0.46898 from the final-observation bootstrap. At discount 1 this persists across the whole sampled stream. Only two games per collection were traced; this does not prove that bootstrapping caused the failures. It motivates checking whether the training target should reflect the registered finite evaluation horizon.

Proposed next test, **not executed or authorized by this exhausted pilot**: Test zero bootstrap at the existing 256-action game cutoff versus J continuation bootstrap, keeping discount 1, rewards, physics, seed and budget fixed. Sampled zero-reward timeout streams retained mean target 0.469. This is an untested finite-horizon target hypothesis; new authorization required beyond the exhausted pilot.

Preserve J and all failed earlier results. No additional training, allocation or promotion follows this report. All three pilot slots are consumed; completion means the bounded experiments finished, not that learning succeeded.
