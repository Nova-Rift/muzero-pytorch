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

## Entry 006 — behavior/evaluation interaction, not yet a causal conclusion

Frozen2048 checkpoints vs owninitial,32developmentgames,swappedseats:
Astandard: policy-only0wins17losses15cutoffs; noisy-searchsame totals;
noise-free search0/0/32.
Bsmoothing: policy-only30wins2losses0cutoffs; noisy-search29/3/0;
noise-free search0/0/32.
These are preliminary ONE-seed development results on legacy cadence, not final.
Changing evaluation mode also changes the opponent, so this alone DOES NOT prove
search harms the trained agent. Register fixed-opponent cross-mode test:
Btrainedsearch vsinitialpolicy and Btrainedpolicy vsinitialsearch. Evaluate only,
no optimizer updates. This isolates one side's behavior change. Add evaluator and
checkpoint hashes to future reports. Preserve earlier reports without retroactive
claims that they contained those fields. Both tests remain development data.

## Entry 007 — fixed-opponent cross-mode evidence

Btrainedsearch vsinitialpolicy:31wins1loss0cutoffs.
Btrainedpolicy vsinitialsearch:25wins0loss7cutoffs.
SameBtrainedsearch vsinitialsearch fromEntry006:0wins0loss32cutoffs.
Thus against a fixed initial SEARCH opponent, changing only the trained decision
rule to sample its policy improved this32-game development outcome. Search is
not universally harmful: against initialPOLICY, trainedsearch won31/32.
This is opponent-dependent failure with one seed, NOT proof that removing all
planning is the correct solution. No final seeds used; two-agent gate not passed.

A separate CPU32-search initial-state audit at seed40000000 confirmed1root action
visited for each active agent, with root values1.1704/1.1555 despite maximum true
discounted game return1. This is initial model optimism, not necessarily a trained
value error. Low-compute search can spend all expansions along one path; increasing
8→32 did not by itself guarantee comparing root alternatives in these states.

## Evaluation scope clarification

Canonical newWorld seeds initialize legacy brain RNG, but our external controller
replaces that brain. Spawn positions themselves are fixed. Evaluation seeds here
therefore vary learned-policy sampling, NOT initial arena layouts. Seat-swapping
controls the two selected starting positions only. A passed gate would establish
progress on this fixed-layout two-agent task, not generalization to new maps,
human-level performance or larger populations. Do not label these new worlds.

Ctraining late batches finish all128games in7decisions. This can reflect a
degenerate shared-policy interaction; do not infer competence from this speedup.
Held-out frozen-opponent evaluation is still required. Throughput comparisons must
report actual transitions and episode length, not justgames/sec.

## Entry 008 — implementation checks and evidence recovery

A/B initial tensors compared equal exactly after downloading both full runs.
All18NPZ files per arm (initial,16periodic,final) plus reports/logs recovered
privately in .local/runpod/muzero-selfplay-001/artifacts/. Public JSON evidence
copied under docs/selfplay-evidence/. No raw weights committed.
H100 CUDA-graph/eager parity passed for forced-root search, including after weight
updates (small regression), and full448hidden/153action/32search/256active-root
test: maximum policy difference0.0. This rules out that suspected fast-path
mismatch on tested inputs, not every possible GPU issue. Test stdout retained
privately full-parity.log. Droot entropy~2.08nats early vsC0 confirms intervention
changes coverage; quality still awaits frozen-opponent evaluation.

## Entry 009 — instrumentation and prepared return-target probe

Ccompleted2048games:1869natural179cutoffs,273.68s.
Dcompleted2048games:1764natural284cutoffs,404.80s. Both32search,homepage cadence.
Dmaintained root diversity but this is NOT yet strength; frozen evaluations running.
Future trainer additionally records153action counts,hits,movement pixels,
search-value MSE against realized discounted returns in completed episodes.
First2games per collection batch save private replay traces(observations,actions,
policies,rewards,targets,masks); this is a sample, NOT all raw trajectories.
Instrumentation-only CPU smoke2games passed;17tests pass,2skipped locally.

Prepared OPTIONAL episode-return target variant (not yet a GPU trial):
propagate actual self-play terminal outcomes through completed episodes, retaining
final-observation bootstrap at true truncations. No reward shaping. Original
10-step targets remain default. This is a credit-assignment hypothesis, not an
identified arithmetic bug. MuZero used final outcomes for board games and10step
returns forAtari: https://arxiv.org/html/1911.08265v2 (AppendixG).
Do not claim this variant improved anything before a registered comparison.

## Entry 010 — evaluation launch overhead

C512development32-game evaluation still running after75s;32-search eager Python
tree traversal launches many smallGPU operations, whereas training captured the
same tensor graph. Interrupted that evaluation before any report was written;
no score inferred. Registered exact-output graph optimization, not a new policy.
FrozenSearch has no RNG inside capture; root noise/candidate sampling stays outside.
CPU parity tested; require CUDA parity before rerunning the SAME pending development
evaluations. Future reports include helper hash and full-process elapsed time in
addition to rollout time. Preserve interruption in private next.log. No final seeds.

Entry010 correction: the first targeted stop found C512had already finished.
Its REAL report is retained:0wins0loss32cutoffs,124.2634s rollout.
The subsequent C1024process was stopped instead (no completed report).
The completed C512evaluation will be rerun with captured search under a separate
filename to measure exact end-to-end score/trajectory parity and speed; never
overwrite the original. This correction supersedes the premature interruption
description above. Preserved next.log is the authority on actual completion.

## Entry 011 — evaluation timing result

C512same32developmentmatches: eager rollout124.263s; captured rollout39.997s,
full captured process43.939s. Same reported0/0/32 andmean256steps; this does NOT
prove full trajectory identity because old run did not record action hashes.
Separate CPU/GPU tensor parity tests pass. Roughly3.1x rollout speedup on this
case, not a universal GPU speedup. Recovered C/D checkpoints locally; initialization
tensors identical across C/D.

Additional performance limitation identified in code: synchronous collection
keeps calculating actions for completed/dead rows until the whole batch ends.
Future logs add padded_search_roots versus active_transitions, so utilization is
measurable before an async-reset optimization. No async-reset change implemented.

Test-count clarification: earlier phrases such as "17tests pass,2skipped" referred
to tests RUN, including skips. Exact latest local result:19run,16passed,3skipped
(CUDA-only2 and opt-in legacy bridge1). Separate H100 frozen-search suite:
2run,2passed. Full-shape graph/eager check separately passed. No skipped test is
claimed as executed. This clarification governs earlier shorthand.

## Entry 012 — checkpoint audit and next registered probe

Read-only checkpoint-series audit saved diagnosis-two-agent-A/B/C/D.json.
At2048,A/Cinitial-state policies essentiallyonehot; Bmaxprob~.95; Dmaxprob~.03,
entropy~4.79. Thus coverage intervention prevented the inspected collapse.
Cvalue estimates differ strongly byseat(~0vs.966) while policy chooses86inboth;
a correct-looking seat value is not evidence of a useful policy improvement.
Ddevelopment512:12wins13loss7cutoffs;1024:12/14/6. No gate passed.

Conditional next E: ifD2048development also fails the registered gate-level
threshold (>60%wins,<=20%cutoffs), run2048games with identical Dsettings/seed and
ONLY return_mode='episode' instead of'nstep'. Source now additionally records
behavior metrics/traces, which do not change RNG or optimization targets.
This tests propagation of actual terminal win information, retaining discount.99
and truncation bootstrap; no human data or tactical reward. Evaluate512/1024/2048
on the same development suite; final seeds remain unused until selection frozen.

## Entry 013 — D outcome and E alignment check

D2048development16wins12loss4cutoffs: insufficient for gate; conditional Etriggered.
Eepisode-return run started fromscratch with registeredDmatched settings.
First128games match D's initial collection totals102natural26cutoffs and identical
search entropy1.030126093; target loss differs as expected. This is a useful
control check, not proof of every trajectory matching.
Etrace0 first2games replayed exactly for256steps against canonical homepage-cadence
physics; all observation/reward arrays matched, valid policy distributions sum1.
Trace hash and command stored trace-audit-E-0.json. No final seeds or agent3.
Latest full local suite with ARENA_SITE:19run,17passed,2CUDA-only skipped.
