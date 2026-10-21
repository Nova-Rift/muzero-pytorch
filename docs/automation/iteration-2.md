# Arena loop 001 — iteration 2, candidate I

Completed training and all development/control evaluations; **development gate failed**. No confirmation, replication, promotion, or homepage change. Pod deleted and verified absent from authenticated inventory. Registered protocol:
[iteration-2-protocol.json](iteration-2-protocol.json).

## Review and intervention

Iteration 1 trained H for 2,048 games, retaining 12 replay collection blocks.
Selected H2048 scored 28/64 wins against initialization, 13/64 against E1024,
and 28/64 against F1024. Its E matchup had 19 timeouts. These development
results failed the registered gate, and no confirmation ran. Stopped G remains
incomplete at 2,944 games; neither its loss nor additional work proves improvement.

I tests the proposed matched-seed control: retain three blocks instead of 12,
keeping H's training seed 51000000, 2,048 games/updates, batch 256, hidden 448,
context 64, 32 searches/eight root candidates, episode returns, and all other
training settings. No existing model.py, physics, reward, architecture, or
trainer code changes. Two agents, five 0.05-second frames per decision, winner
reward 1 and all other outcomes 0; from-scratch self-play only.

Separate runner and protocol on branch arena-loop-001-I, source 8adea21.
The runner includes a remote 2,200-second subprocess deadline and requires
local owner-control review before each training/evaluation process. The private
manager was adapted to iteration 2, fresh paths and receipt, and the existing
cache; historical pod IDs/reservations were not reused.

## Registration and checks

Three development checkpoints (512/1024/2048), 64 balanced-seat games per
opponent, initial weights and frozen E1024/F1024. Fresh development seed bases
62000000/62010000/62020000. Selection maximizes minimum opponent win fraction,
then total wins, fewer timeouts, earliest checkpoint. Gate: >60% wins and <=20%
timeouts against EACH opponent. Freeze selection before conditional confirmation
at 63000000/63010000/63020000, 200 games each; require 95% Wilson lower bound >.5
and <=20% timeouts against each, then independent training-seed replication.

Frozen H2048 is evaluated on those same fresh development samples, and selected
I plays H2048 on seed 62030000 (64 balanced games). These diagnostic comparisons
are excluded from selection. They are development evidence, not confirmation.
Training never consumes evaluation trajectories.

Before allocation: 30 unittest cases ran, 28 passed and two CUDA-only tests
skipped. Canonical cadence/agent tests ran with the verified cached site. Source,
physics and frozen-opponent hashes all matched. Original model, trainer,
environment and evaluator bytes match H. A formatting-only whitespace check
was corrected before staging. No packages installed.

Early matched-prefix audit: initialization and checkpoint384 hashes identical;
first three blocks' losses, work counts, action histograms and saved trace arrays
match exactly, maximum checkpoint tensor difference zero. Trace coverage is
first two games of each block, not every trajectory. Replay eviction differs
after this prefix; this audit establishes matched initial conditions, not quality.

## Commands

```sh
ARENA_SITE=VERIFIED_CACHED_SITE python3 -m unittest discover -s tests -v
python scripts/run_iteration_i.py --site /workspace/experiment/site --output /workspace/arena-loop-001-I --weights /workspace/experiment/weights
```

The runner records exact training/evaluation subprocess arguments. Its training
command matches H except output path and `--replay-blocks 3`; evaluation uses
`scripts/evaluate_selfplay.py` with the registered counts, seeds and checkpoints.

## Owner guidance acknowledged

Guidance entry timestamp 1792563591.7862878: conversational owner control,
read guidance at stage boundaries, PAUSED/STOPPED override continuation,
winning reward 1 and other outcomes 0. Checked before stages and progress updates.

## Limits

One matched training seed and fixed arena layout; sampling seeds do not vary
maps. Development is used for selection. No general intelligence, continual
learning, superhuman result, or automatic homepage replacement is implied.

## Results

All 832 evaluation games are DEVELOPMENT: 576 selection games plus 256
registered frozen-H controls. All records, hashes and audit summaries are in
[the evidence directory](iteration-2-evidence/summary.json). Timeouts and draws
count as nonwins. Each row contains 64 games, balanced 32 per seat.

| Challenger | Opponent | Wins | Losses | Timeouts | Draws |
|---|---|---:|---:|---:|---:|
| I512 | initial | 25 | 28 | 11 | 0 |
| I512 | E1024 | 11 | 39 | 14 | 0 |
| I512 | F1024 | 32 | 25 | 7 | 0 |
| I1024 | initial | 20 | 30 | 14 | 0 |
| I1024 | E1024 | 17 | 33 | 14 | 0 |
| I1024 | F1024 | 31 | 24 | 9 | 0 |
| I2048 | initial | 29 | 26 | 9 | 0 |
| I2048 | E1024 | 15 | 32 | 17 | 0 |
| I2048 | F1024 | 36 | 18 | 10 | 0 |
| H2048 | initial | 34 | 16 | 13 | 1 |
| H2048 | E1024 | 16 | 32 | 16 | 0 |
| H2048 | F1024 | 31 | 19 | 14 | 0 |
| Selected I1024 | H2048 | 29 | 30 | 5 | 0 |

Selection chooses **I1024**, SHA256
`da52c6658398b468c02a82fa804610de7b46249b5fc62bad92383f5044202ad2`.
Its weakest matchup is E1024: 17/64 wins (26.56%) and 14/64 timeouts (21.88%).
It fails both gates there and win gates against every opponent. Confirmation
seeds remain unused. Frozen selection is retained, not revised after controls.

At equal 2,048-game checkpoints, I versus H wins were 29 versus 34 against
initialization, 15 versus 16 against E, and 36 versus 31 against F. Selected
I1024 versus H2048 was 29 wins/30 losses/five timeouts. This does not establish
an overall benefit from shorter retention, nor support longer retention as the
main cause of E weakness. One matched training seed, small development samples,
and checkpoint selection limit causal/general claims. The equal-budget and
selected-checkpoint comparisons are distinct.

Training completed 2,048 games/updates: 1,798 natural endings, 250 timeouts,
268,245 joint steps, 536,490 active transitions and 524,288 sampled examples.
Runtime 369.98 seconds. All 37 recovered NPZ files are finite. All 13 evaluation
files were recounted independently, including seat balance and checkpoint hashes.
Initial weights and prefix384 match H exactly; checkpoint512 diverges as expected
(maximum parameter difference 0.0468325). Physics/reward/model.py stayed unchanged.

## Runtime and cost

H100 80 GB HBM3, 81,559 MiB reported, driver 595.91.07, PyTorch 2.8.0+cu128,
CUDA 12.8, Node 22.22.2. Verified 319 cached payload files; approximately
12.01 MB transferred for 188.35 MB payload. No packages, storage or endpoints
created. The first artifact poll occurred before training created its output
directory and returned rsync code 23; later recovery and final audits passed.
No experiment was retried or second allocation made.

Thirteen evaluation processes totaled 558.12 seconds. Allocation request through deletion response: 1187.96 seconds (19.80 minutes) at $3.99/hour, **$1.316652 estimated GPU cost**. Reconciled once into the existing private ledger; cumulative recorded estimate $13.295347/$50. Reservation stays consumed. Invoice not reconciled; existing cache storage is separate.

## Proposed next hypothesis — not executed here

A descriptive audit of I's saved training traces found 27 winning sample streams,
median winning step 102, mean initial discounted target 0.388 despite realized
winner reward 1. The current .99 discount explicitly downweights delayed wins;
this is not a target arithmetic bug or proof of the failure's cause. These traces
cover only two sampled games per collection, not the complete training data.

Test discount .99 versus 1.0 as the next single material change, exposing an
explicit trainer configuration option while preserving every existing model.py.
Keep three-block replay, training seed/budget and winner-only rewards fixed;
first test return and search consistency at discount 1. Evaluate on fresh
registered development samples with the failed I control, not confirmation data.
Hypothesis: undiscounted value/search targets better align with maximizing win
probability rather than favoring earlier wins. Risk: greater bootstrap/model-error
propagation; success is uncertain and still requires all original gates.

This invocation executes only iteration 2. No next allocation was launched.
