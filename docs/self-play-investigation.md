# Self-play learning investigation — October 20, 2026

Authorization: Clark requests persistent diagnosis/improvement, a log of all
attempts, no human demonstrations or scripted opponents. Subsequent steering:
start with TWO agents; add one only after success and notify at each addition.
No tactical rewards, supplied aim policy, or fixed opponent curriculum. Frozen
previous self-play checkpoints are allowed as evaluation opponents. Preserve
original 11-agent artifacts and model.py; implement extensions in new files.
Existing total compute cap $50 includes previous approximately $1.2835 GPU costs;
account-wide spending is not inferred. First phase CPU diagnostics, then bounded
GPU allocations with shutdown guard and recovered evidence.

## Protocol 001 — diagnosis and two-agent pilot (registered before new runs)

1. Inspect historical metrics and original frozen checkpoint without training.
   Measure action entropy, search visits, action distribution and predictions.
2. Add separate two-agent adapter using canonical game physics, slots0/1 equally
   capable, other slots inactive including former human slot10. No reward changes.
   Preserve 64-dimensional observations/153 actions and old adapter behavior.
3. Both opponents are the same newly initialized learner during self-play;
   random initialization and sampling only, no external behavior data.
4. Save periodic checkpoints and evaluate side-swapped matches against frozen
   earlier checkpoints with independently seeded action sampling. Development
   seeds start41000000, training40000000; untouched final seeds42000000. Do not
   select checkpoints using final scores. Randomize/rotate seats for comparisons.
5. Success gate before adding agent3: selected learner wins >60% of at least200
   final matched games versus initial frozen learner, lower95% Wilson bound >50%,
   <=20% cutoffs; replicate direction on a second independently trained seed.
   Win rate counts cutoffs/draws as nonwins. Also compare intermediate checkpoints
   to detect exploitation/regression. This is a progress gate, NOT superhuman.
6. Bound each attempted change, record source/config/work/cost and negative
   results. Escalate fidelity/debugging before simply increasing throughput.

## Entry 001 — historical audit, no optimizer steps

Original 11-agent run first256games:242natural ends. Through2048games:1951natural
ends. At batch ending2560games:0/256natural ends; every later batch also0.
Final2191/10000natural ends. Loss fell from roughly4.8 to0.139 at collapse, then
near0.02. Thus an initial inability to finish games is NOT the whole explanation;
training destroyed previously decisive behavior. Hypotheses: collapsed policy
and underexploration, erroneous value targets/model, search bias. Not yet proven.

Next: frozen-checkpoint/action audit and isolated two-agent environment tests.

## Entry 002 — frozen initial-state diagnosis

Read-only script scripts/diagnose_policy.py, full numbers in
diagnosis-initial-state.json. Original initialization seed30000000, same real
initial observations. Initial mean policy entropy5.0193nats (maximum log153=5.0304),
largest probability~0.009; search explores4–5actions with8simulations. Final model:
entropy0.0001322, action136 probability~0.999992 in ALL11positions; noisy training
search gives100% visits to136 at every root. Action136 = move up, shoot315degrees.
This confirms behavioral/policy collapse in inspected states and failure of
existing root noise to escape it. It does NOT identify the original cause.
All results fromCPU, no learning updates or paid compute.
Two-agent adapter leaves model.py and canonical game physics/reward unchanged;
slots0/1 active, including no privileged human slot10. Old11-agent default retained.

## Entry 003 — registered two-agent ablation

Pilot A: original algorithm,2agents,seed40000000,2048games,128parallel,
hidden448/context64/8searches/batch32/unroll5/256decision cutoff/1update pergame.
Pilot B: identical seed/work with ONLY policy-target smoothing0.05:
target = .95 * self-play search visits + .05 /153. This contains no tactics or
demonstrations; it tests whether preventing complete loss of action support
avoids collapse. It is an algorithm deviation, not evidence of original MuZero
reproduction. Neither model.py nor game rules/rewards change.
Both save every128-game checkpoint including initialization; development compare
initial vs each selected intermediate using independent action RNG/seat swaps.
Pilot hypothesis: floor preserves exploration; it might NOT improve win rate.
Further trials require a new log entry, not silent hyperparameter tuning.
CPU smoke2games/8steps/hidden32 only validates the new pipeline, not policy quality.
