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

## Recording standard (Clark's follow-up)

Each trial must distinguish learning-quality, implementation correctness, sample
efficiency, GPU throughput, environment transport, memory, and evaluation limits.
Retain failed approaches. Record the exact symptom, hypothesis, changed variable,
source revision/config/seed, metric and units, resource cost, causal strength,
and reusable lesson. Hypotheses do not become causes solely because a later run
improves. No claim of a leading-lab affiliation or superhuman performance.
Pilot001 provisioned one H10080GB at reported$3.99/hour with a$4/45-minute
maximum allocation guard (whichever is sooner); preserve actual receipts privately.

## Entry 004 — confirmed integration timing discrepancy

The original bridge do/while(decisionClock>0) takes6frames because subtracting
.05 five times from.25 leaves a positive floating remainder. Headless homepage
explicitly takes5frames and resets its decision clock. One unobstructed right
move measured65.9999996px in bridge versus expected55px in homepage.
Older transport test compared against another bridge with the SAME cadence,
so it missed this discrepancy. Lesson: cross-check the actual consumer loop,
not only two transport implementations sharing an assumption.
New explicit cadence='homepage' takes exactly5frames; legacy remains reproducible.
Regression asserts55px; model.py/environment physics and rewards unchanged.
PilotsA/B underway use LEGACY cadence and are valid only as an algorithm pair;
do not call them compatible homepage checkpoints. Subsequent pilots use homepage
cadence and document this separately. No causal claim that timing caused collapse.

## Entry 005 — A development evaluation and next registered pair

A2048:1929/2048natural training ends, but frozen evaluation vsinitial32games,
seed41000000,seat-swapped:0wins/0losses/32cutoffs. No pass, no agent addition.
Training search uses root noise; evaluation does not. Need distinguish search
degeneracy from learned competence. Add diagnostic policy-only/noisy-search modes;
these are clearly separate evaluators, not substitute scores for the original gate.

Next C/D matched pair: homepage cadence,2048games,seed40000000,32simulations,
otherwise same as A. Cstandard search; Dfirst8root visits reserved for8distinct
actions sampled WITHOUT replacement from mixed network/noise prior; remaining24
standard tree expansions. No policy-floor. Dtests root coverage, not tactics.
This is an independently written exploratory variant, NOT full Gumbel MuZero.
Motivation: Danihelka etal2022 documents low-simulation root-coverage failures:
https://openreview.net/pdf?id=bERaNdoegnO . Their guarantees require correctly
evaluated action values and do not prove our learned model is accurate.
Tests verify default parity and forced candidate visitation despite peaked prior.
