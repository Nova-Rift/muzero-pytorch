# Self-play G — longer data/optimization budget, registered before training

Status: started October 21, 2026 UTC after full tournament recovery, audit and
GPU deletion. The registered settings and gates below are unchanged.

Question: can additional from-scratch self-play and optimizer work improve the
unchanged independent MuZero, without a new loss, architecture, or reward?
F's larger batches alone did not pass development. The terminal fit probes show
small-sample learnability but poor transfer/calibration. Those probes do not
justify deploying biased terminal oversampling or establish an architectural bug.
This trial tests training duration, not those replay modifications.

## Fixed comparison

Copy F's exact settings: hidden448, context64, support10, discount.99,
32 simulations,8 sampled root candidates, explorationfraction.25, TF32 ON,
parallel128,4 workers,batch256,unroll5,episode returns,learningrate3e-4,
Adam weightdecay1e-4,gradientclip100,1 update/game,replay3collectionblocks,
CUDA graphs/device history,2 agents,256decision limit,homepage5x.05s cadence,
seed40000000. Reward only1 for the winner/0 otherwise. No scripted policy,
human demonstrations, tactical reward, or architecture change.

Only intended learning-budget change:2048→8192games and optimizerupdates.
Record actual transitions and examples; this is4x requested games/updates,
not equal data or compute. Reinitialize from the same seed, do not resume a
weights-only checkpoint without optimizer/replay/RNG state. Initial checkpoint
and first2048game trace/metrics/checkpoint comparisons against F are required;
report any numerical divergence rather than calling unmatched prefixes identical.
Preserve F and E results, including failures. model.py remains unedited.

## Selection and confirmation

Evaluate2048(control),4096,8192 on the existing32game development seed41000000,
using the established evaluator and matching saved untrained initialization,
16games in each seat. Freeze selection by highest wins, then fewer cutoffs,
then earliest checkpoint. Development is reused for selection and not held-out
proof. Only if >60%wins and <=20%cutoffs: freeze weights/hash and evaluate200
new matches at reserved final seed42000000,balanced seats,without reselection.
Require95%Wilson lower win-fraction bound>.5 and <=20%cutoffs for confirmation.

A passing first run then triggers a separately recorded from-scratch replication
with seed40010000,development41010000 and final42010000, same budgets/settings
and selection/gates. These ranges do not overlap G's8192training game seeds.
No thirdagent or learning-success claim until both independent training seeds
pass confirmation. No tuning on either final set. A failed run remains evidence;
no extra checkpoints selected after looking at final scores.

## Bound and evidence

One H100<=4.00/hour allocation,45minutes/$3.00 maximum; training wall bound2100s,
then only completed listed checkpoints evaluated. Incomplete8192work is reported
incomplete, not equivalent to8192. Recover all metrics, checkpoints, traces,
source receipt, selection and evaluation records, then delete GPU pod.
Current recorded compute~$6.43339 plus active tournament<=3.00,originalcap$50.
Persistent10GBstorage$0.70/month separately authorized. Replication is conditional
and requires a fresh bounded allocation record, not idle background compute.

Use new verified source receipt from tools/runpod/cache.py; do not identify cached
code from stale .git metadata. Runtime/package versions and exact physics hashes
must match or differences must be disclosed. No new packages are required.

## Pre-training robustness amendment

Added BEFORE G starts, based on completed exploratory matchup evidence:
Lana100 beats its initialization123–5 (0timeouts); that initialization beats
legacyMuZero initialization77–4 (47timeouts); legacyMuZero initialization beats
Lana10060–2 (66timeouts). Thus matched-initialization success alone can hide
opponent-specific behavior. This does not tune any policy or use reserved seeds.

Keep the checkpoint-selection rule unchanged. Before thirdagent promotion,
require each independently selected G policy also to pass two frozen-own-policy
anchors: E1024 SHA18c18151b820636ef3bf67ea52dbecd3cb47582d9ea6bf63ede36d6c2adb1532
and F1024 SHA4de08ce08adef8344d6cf8707e22a74896c355008ffff18d3b10ce21ec1dcdb7.
Both anchors are existing independently implemented, from-scratch self-play
policies, not scripted opponents or human demonstrations. Evaluation only;
G training remains unchanged latest-policy self-play.

200balanced-seat games per anchor, no root noise, same32search/8candidate
inference. First trainingseed uses new seed bases43000000 (E) and43010000 (F);
replication uses43020000/43030000. Freeze selectedhashes before opening any of
these tests. Require each win-fraction95%Wilson lowerbound>.5 and <=20%timeouts,
as well as the previously registered matching-initialization confirmations.
Report all tests, including failures; no reselection/tuning on these seeds.
If allocationtime is insufficient, recover/delete and register a separate bounded
confirmation allocation within remaining$50; never silently extend a live guard.
This strengthens the success criterion before the experiment; it is not a claim
that either earlier candidate already passes or that passing proves superhuman
play/generalization beyond this fixed-layout arena.
