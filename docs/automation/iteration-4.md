# Iteration 4 — fixed-trajectory training counts; allocation failed

**Failed before training.** Runpod returned HTTP 500 to the single pod creation request. No creation receipt was returned. Subsequent authenticated inventory queries found zero pods, including zero matching this iteration. No training, development, confirmation, replication or promotion occurred. The pilot is paused; there was no retry. The $3 reservation remains consumed under the new $15/day policy.

## Scientific question and frozen plan

[Preregistered protocol](iteration-4-protocol.json), source commit `8d4e558` on isolated branch `arena-loop-001-K`. The worker reviewing evidence and editing code selected this entire plan; scripts only implement it.

Historical H/I/J all failed development gates. J2048 won 33/64 against its initialization, 20/64 against E1024, and 26/64 against F1024. Stopped G reached 2,944 games without development evaluation and used a different discount. Neither lower fitting loss nor more training establishes improved control. The latest owner constraint supersedes J's proposed timeout-bootstrap intervention: this iteration varies training count only.

One from-scratch trajectory, seed 51000000, was registered for 4,096 games and 4,096 updates. Compare its 1,024/2,048/4,096 checkpoints with identical hidden448/context64, batch256, parallel128/four workers, discount1, learning rate3e-4/Adam weight decay1e-4, replay three collection blocks, unroll5, 32 searches/eight root candidates, episode returns, one update per game and unchanged timeout bootstrapping. TF32, device history and CUDA graphs remain enabled for training. Two agents, canonical five 0.05-second physics frames per decision, winner1 and every other outcome0. No scripted training, demonstration, reward change or existing model.py edit.

Each count was to play 64 development games against each of matching initialization, E1024 and F1024: 576 games total, 32 per seat per matchup. Seed bases82000000/82010000/82020000 repeat across count checkpoints. This brackets the prior 2,048-game budget and one doubling at bounded cost; initialization measures learning from scratch, while E/F measure opponent specificity. E/F remain frozen anchors at their original discount.99, not matched-duration candidates. Search is sampled with no root noise, fixed32 searches/eight candidates and256-decision cutoff. Separate reproducible per-player generators are used identically for every count. Historical evaluator outcomes are not pooled with this new RNG protocol.

Selection: highest minimum opponent win fraction, then aggregate wins, fewer timeouts, earliest count. All outcomes count in the denominator. Every opponent must exceed60% wins and have <=20% timeouts before freezing selection/hash and running200 fresh confirmation games per opponent at83000000/83010000/83020000. Confirmation requires each95% Wilson lower win fraction>.5 and <=20% timeouts. Independent training-seed replication remains necessary for promotion and is not part of this single trajectory. Conditional confirmation must fit the remaining deadline; no extension/reallocation. All seed ranges remain unused for scientific evaluation. No cross-family ranking was run, so all arena incumbents remain unchanged.

## Code and verification

Added full model/optimizer/replay/initial-weights/NumPy and Torch CPU/CUDA RNG snapshots at512-game boundaries and completion, reusing the separately prepared trainer. Restore checks settings, source, environment and PyTorch version. This run would start from scratch. CPU interruption/resume reproduced final weights exactly in a four-game tiny engineering test; GPU resume equivalence is unverified and no GPU restart was attempted.

Evaluator changes add independent player RNG streams and atomic live completed-game counters. Natural wins/losses/draws count when observed; unfinished games become timeouts only at the cutoff. Two repeated CPU evaluations reproduced every outcome record, and final live counters balanced. These tiny checks are engineering tests, not learning evidence. All count checkpoints would receive the same modified evaluator, constituting a new matched evaluation baseline.

Thirty of32 unit tests passed; two CUDA-only checks skipped. Existing cadence/agent tests used the verified cached canonical site. Source/opponent/physics hashes and unchanged model.py verified; pretraining diff saved. No packages installed. Exact snapshot and evaluator check results: [local evidence](iteration-4-local-check.json).

Commands prepared/tested:

```sh
ARENA_SITE=VERIFIED_CACHED_SITE python3 -m unittest discover -s tests -v
python scripts/run_iteration_k.py --site /workspace/experiment/site --output /workspace/arena-loop-001-K --weights /workspace/experiment/weights
```

The committed runner contains complete training and evaluation commands and implements the frozen protocol. Its remote child-process deadline is2,300seconds; training bound1,200seconds. Local monitoring checks guidance/stop markers at stage boundaries and progress updates, recovers artifacts, publishes actual completed counts, rejects a training projection exceeding the bound and cleans up in a finalizer. It was prepared but never installed on a pod.

## Preflight and failure

Authenticated current offers at the existing cache datacenter showed H100 SXM80GB at$3.99/hour, low stock. A100/4090 were unavailable there; H200 SXM exceeded the hourly limit. H100 was chosen because prior same-setting J measurements exist:2,048 games in411.32seconds,832 evaluation games in559.01seconds. This is not a universal price/performance claim. New evaluator and state-save overhead make extrapolation uncertain.

Conservative estimate: setup150s, training900s, development400s, conditional confirmation300s, recovery180s,20% contingency386s, cleanup120s =2,436seconds/40.6minutes and$2.6999 GPU cost. Existing storage separate; disk/network invoice charges unknown. Full private preflight includes source/environment hashes, offers, settings, artifact paths, checkpoint validation and abort conditions. Daily allowance before reservation was$15.

The active guard was checked and shared reservation for iteration4 obtained before the one POST request. Request time2026-10-22T02:42:28 UTC (October21 locally). Provider HTTP500 returned no pod ID/receipt, so registration could not occur. No retry, substitute GPU or counter reset occurred. The shell then invoked monitoring despite the failed creation; it correctly could not find a receipt and stopped, but its receipt-dependent cleanup verification also errored. Those secondary failures are preserved and are not training errors. A separate authenticated reconciliation queried account inventory twice and verified zero pods at02:42:51 UTC. Shared cleanup was also called. A future launcher must gate monitoring on successful creation/registration and reconcile receipt-less creation errors directly.

Observed allocated runtime: none; no GPU hardware was assigned. Estimated attributable GPU cost is$0 based on empty inventory and absent receipt, **not a reconciled invoice**. The conservative$3 reservation remains consumed. Existing private cost ledger records this failure once without increasing cumulative estimated spend. There are no private weights or scientific result logs from this attempt. Local engineering-test artifacts remain private.

## Owner controls, result and next hypothesis

Acknowledged guidance1792563591.7862878 (conversational controls and stop precedence) and1792636607.7061172 (fresh daily budget, MuZero/two agents, fixed trajectory, preflight and recovery). Guidance checked before work/stages; no owner stop marker existed at launch. Monitoring persisted PAUSED after the failure, and it remains in place. No unrelated repositories, account settings, website defaults or arena roster changed.

Scientific result: **unknown**, zero registered training/evaluation games completed. This is an infrastructure failure, not a failed learning hypothesis and not a completed research iteration. Preserve the frozen count-only question for owner-reviewed continuation within a newly bounded attempt. Fix the launch handoff before any future allocation; no automatic retry is authorized. Do not change the scientific settings on the basis of this failure.
