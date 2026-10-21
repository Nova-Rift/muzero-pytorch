# Self-play G — longer data/optimization budget, registered before training

Status: prepared, NOT started. Finish the active frozen all-model tournament first.

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
