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

## Selection rule before E evaluations

For E's prelisted512/1024/2048checkpoints, select highest development win count
(out of32), then lowest cutoff count, then earliestcheckpoint. Freeze selected
checksum/config before using any42000000final seeds. If no candidate exceeds
60%development wins with<=20%cutoffs, do not run a confirmatory claim or add agent3.
A second independent training seed is required even if one final evaluation passes.
D2048seat breakdown:seat0 7wins6loss3cutoffs;seat1 9wins6loss1cutoff, so its50%overall
score is not simply always winning from one seat. Bpolicy-v-initialsearch:
seat0 13wins0loss3cutoffs;seat1 12wins0loss4cutoffs (different evaluation mode).

## Entry 014 — E results and requested experimental homepage release

E completed 2,048 games / 2,048 updates in 415.00 seconds on H100 80GB HBM3:
1,894 natural endings, 154 cutoffs, 242,401 joint steps / 484,802 active transitions.
Development checkpoint512: 16 wins / 11 losses / 5 cutoffs; 1024: 18/12/2;
2048: 9/12/11. All32 games use seed41000000, swapped fixed-layout seats,
noise-free 32-simulation search with eight prior-sampled distinct root candidates.
Registered selection chooses1024 (hash in selfplay-E/selection.json). Later training
worsened this development score; no monotonic learning claim. E1024 scored higher
than D2048's16wins, but this small development comparison does not prove superiority.
No candidate passed >60%wins, so no final suite, confirmation claim, or agent3.

Clark requested the best model on the homepage. Export E1024 as explicitly
experimental, default two agents, preserve older selectable models. This is an
interactive research preview, not promotion based on a passed success gate.
Browser deployment uses sampled actions and prior-weighted sampling without
replacement for candidates; per-agent PRNG differs from PyTorch. Fixed-candidate
Python/WASM search parity passes. No model.py edits, human demonstrations, scripted
opponents, tactical reward shaping, or paid website inference.

All A–E checkpoints recovered to private local storage before pod deletion.
Pod sg87iusfa5it3r deleted; subsequent authenticated inventory shows zero pods.
Allocation2,503.73s at$3.99/hour estimates$2.77497 compute, plus earlier$1.28350
=~$4.05846 total recorded compute (not reconciled account-wide billing).
This bounded GPU pass is stopped. Preserve development/final split and original
negative results. Further research needs a bounded hypothesis and independent-seed
replication; adding agents remains gated.

## Entry 015 — read-only reward and replay audit; bounded next probe

Previous goal turn made progress: experimental E1024 preview published as v49,
with fixed-candidate Python/WASM parity; all GPU resources deleted.
Read-only CPU audit of all saved first-two-game Etraces (7,130 active transitions,
29 immediate wins) shows E1024 predicted reward on wins mean.00649 vs target1;
E2048 mean.00838. These are conditional sampled outcomes, not proof each expected
reward should equal1 in a stochastic multi-agent process. Zero predictions exceed.5.
Aggregate reward MSE~.004 hides the rare positives. E2048 root-value MSE.08498
beats pooled constant.09869 despite weaker development play, so root-value MSE
alone is an inadequate checkpoint selector. Data are training-derived, mixed policy
versions, and only two of each128games; no held-out performance claim.

A five-step replay segment contains a win for145/7130 valid starts. Under pooled
uniform sampling, probability a32sample batch contains none is51.82%. Actual
trainer samples one of its lastthree blocks first; this pooled estimate is NOT an
exact reconstruction of every minibatch. No causal explanation established yet.
Evidence: diagnosis-reward-E.json, diagnosis-replay-exposure-E.json;
reproduce reward audit with scripts/audit_reward_signal.py RUN NEW_OUTPUT.

Register bounded CPU reward-only fit probe before results: freeze E1024
representation/dynamics/policy/value and fit copies of its reward head only.
First12saved trace files are fitting data, last4 diagnostic data (already observed,
not a pristine final holdout). Two arms: uniform replay versus halfuniform/
halfpositive replay, using exact inverse-probability weights1/(N*p) so expected
objective/reward remains identical. Three matched RNG seeds123/456/789,
512Adam steps perarm, batch32, lr3e-4, weight_decay1e-4, same original supportCE.
Report pre/post rewardMSE, separate positive/zero predictions and fitting/probe
counts. No tactical shaping, opponent, environment or model.py change; no policy
learning or new final seeds. This isolates whether rare reward supervision is
learnable with frozen features and lower sampling variance, not whether it improves
control. A later full self-play experiment requires separate registered criteria.

## Entry 016 — frozen-feature reward replay probe outcome

All six registered CPU runs completed. Uniform arms saw65/65/77 positive samples;
importance-corrected arms8322/8318/8382. Neither learned strong positive reward
prediction: diagnostic seven winning transitions remain mean<.001 in everyarm.
Fitting22win predictions also remain<.006. Diagnostic aggregateMSE~.004045 is
mostly zero rewards. This falsifies the simple claim that exposing the fixed
reward head to more positive samples with unchanged expected objective is enough
in this512step setup. It does NOT prove representation irrecoverable, rewards
unlearnable, or reward error caused poorer policy. Frozen features, tiny positive
count, chronological/mixed-policy split and already-seen diagnostics limit inference.
No original weights changed; copied rewardheads discarded. No paidcompute or
newgames. Next bounded diagnostic should examine feature separability/optimization
before paying for a full prioritized-replay training run. Rawstates and latent
features can be compared on identical stored transition labels with a separately
registered fitting protocol; do not reinterpret diagnostic results as finalpolicy
validation. Evidence reward-replay-probe-E.json and scripts/probe_reward_replay.py.

## Entry 017 — registered frozen-feature linear probe

Previous turn completed requested homepage speed controls (Sitesv50); no new
research outcome in that site-only work. Resume next diagnostic now.
Before results: compare raw current observation+onehot action (217dimensions),
last4observations+onehot action (409), and frozen E1024 raw recurrent feature(448).
Use the same first12trace fitting / last4trace diagnostic split as Entry016.
Fit class-balanced ridge least squares with fitting-only standardization, unpenalized
intercept, fixed regularization1 (sum weighted loss, weights sumN). Report all arms,
AUROC with ties, positive/negative score means and class-balanced MSE, on bothsets.
This is a linear separability diagnostic; scores are not calibrated rewards and
never enter search/training. Only22positive fitting/7diagnostic examples, chronological
mixed-policy traces, and unequal feature widths prohibit broad generalization or
architecture claims. Current and last4 observations do not span all64historyinputs.
No new game, reward alteration, optimizer update to agent, paidcompute or finalseed.

## Entry 018 — separability evidence and bounded larger-batch next trial

All three registered linear probes completed. Diagnostic AUROC current.8077,
last4.8459, recurrent.8042; fitting.9927/.9972/.9877. Balanced diagnosticMSE
.2516/.2298/.2122 respectively. These are uncalibrated class-balanced linear
scores, not the agent's reward predictions. Learned features retain some signal;
this contradicts a blanket claim they contain no information about immediatewins.
Tiny7positive diagnosticset, different dimensions, alreadyobserved traces and
chronological policyshift prevent robust architecture comparisons. No replacing
model.py or switching toTransformer is justified by this probe alone.
Evidence feature-separability-E.json and scripts/probe_feature_separability.py.

Next registered trial F: same E from-scratch settings, ONLY batch32→256.
Retain2048games/2048optimizersteps, lr3e-4, discount.99, episode targets,
rootcoverage8/search32, hidden448, selfplayseed40000000, homepagecadence and2agents.
This uses8x more replayexamples, not matched compute/data; report that explicitly.
Hypothesis: larger batches reduce sparse reward/gradient noise while updating
features jointly; frozen-head corrected-replay failure means success is uncertain.
No new shaping, demonstrations, opponent policy, or architecture. Evaluate listed
512/1024/2048 checkpoints on existing32game developmentseed41000000, rankwins then
fewestcutoffs thenearliest. Same >60%wins/<=20%cutoffs trigger before fresh final
suite plus independent training-seed confirmation. No finalseeds or thirdagent
unless that protocolpasses. Preserve E and websitev50 regardless of Fresult.
Authorize within existing usergoal/budget one <=20minute H100 allocation at
<=4.00/hour (ceiling$1.34), trainingmax900s thenevaluate/recover/delete; automated
allocationguard. Prior recordedestimatedcompute$4.05846, totalcap$50. Allocation
has NOT started in this entry. If capacity/rate unavailable, do not silently
expand limits. First recover checkpoints and logs, then delete; accountwidebilling
is not inferred from recorded runtimecosts.

## Entry 019 — F running, matched-start check

F manifest comparison against E: only outputdirectory and batch32→256 differ;
model config and all environment module hashes identical. First128game collection
before any optimizer update exactly matches E's naturalends/cutoffs/jointsteps/
active transitions, action histogram, and search entropy. This supports matched
initial conditions for the intended one-variable training comparison. Reported
firsttwo updateblocks take2.97/3.19seconds versus~21.6seconds search percollection;
GPU spotcheck95%utilization/1285MiB. Do not confuse GPUutilization with useful
active-root utilization; paddedfinishedgames remain a known waste. Full run and
policy evaluations still pending at this entry. Pod and guarded collector handles
are in private handoff; no success claim or extra agent.

## Entry 020 — F complete, negative larger-batch result

F completed2,048games/updates,524,288replayexamples,276,321jointsteps/552,642active
transitions in371.96s;1,769naturalends/279cutoffs. Development512:15wins/15loss/2cutoffs;
1024:16/13/3;2048:16/9/6 plus1draw. Registered tiebreak selects1024 (fewercutoffs),
SHA4de08ce08adef8344d6cf8707e22a74896c355008ffff18d3b10ce21ec1dcdb7. No candidate
passes >60%wins. No confirmation/finalseeds, replication, thirdagent or sitereplacement.
Larger replaybatch alone did not improve this tested development comparison;
this is one matched trainingseed and small32game scores, not proof it neverhelps.
E selected18wins remains the current homepageexperimental model.

Offline Fcheckpoint audit uses EXACTLY E's7,130saved transitions/29wins (tracehashes
match); report diagnosis-reward-F-on-E.json. F1024 win-reward mean.01076 versus
E1024.00649, F2048.00243 versus E2048.00838: stillmostly nearzero. Aggregate reward
MSE~.004 masks rare positives. Ebehavior returntargets are offpolicy for F, so root
valueMSE here is a descriptive fit diagnostic, not an unbiased Fvaluequalityscore.

All F NPZs recovered and checkedfinite before deletion; archivechecksum inprivate
recovered.json. Podple0re8b6cgeg4 deleted; authenticated inventoryzero pods.
Allocation658.19s at$3.99/h estimated$0.72950; recorded runningtotal~$4.78796/$50cap
(not account-wide reconciled billing). No packages/model.py/physics/reward changes.
No trainingprocess remains from F. Do not equate371.96s versus E415s with a pure
batchspeedup: rollout lengths, host and timing variation also differ. Actualgames
and all workcounts retained. Next research remains open; inspect uncertainty and
planning evidence before another architectural or replay change.

## Entry 021 — fast evaluation and terminal-value diagnosis

GPU batch32→128 benchmark completed768games over3workloads; all384 matched
records (outcome and length) exactly equal. Speedups1.151x/2.861x/3.881x; full
hardware/setup/cost records in docs/tournament-001.md and benchmarkJSON.
Full14entry/91pair/11648game exploratory tournament now running, not complete.
No promotion or training outcome inferred from partial standings.

Read-only Etrace value-equivalence audit:7,130 valid transitions,58terminal,
7,066 surviving-next and6cutoff transitions. E1024 recurrent value after actual
terminal transition averages.294074 versus target0; E2048.246219. E1024 surviving
nextvalue MSE.002189 versus encoded-next-value variance.003608. Thus aggregate
one-step agreement masks terminal failures. Reward at winning transitions also
remains weak per prior audit. This is training-derived data, stochastic opponent
and partial state information; not an independent test or proof of a target bug.
Evidence scripts/audit_value_consistency.py and docs/value-consistency-E.json.

Copied-network learnability probe starts from E1024, same frozen saved data:
first12traces fit (5400transitions/44terminal/22wins), last4 diagnostic
(1730/14/7). Full model gradients, unroll1, batch32, Adam3e-4/wd1e-4, clip10,
seed0; no new games or deployment, original weights/hash unchanged. This is not
an online policy experiment and does not reuse finalworld seeds.
128steps uniform vs50%-terminal sampling: balanced fit winreward.791 and terminal
value.117, versus uniform.00111/.266. But diagnostic other-reward mean.0674 and
rewardMSE.02588 versus uniform.00404: biased sampling raises false predictions.
512step follow-up adds exact importance correction (terminal weight2p, other
weight2(1-p), p=44/5400). Uncorrected balanced fit reward.880/terminalvalue.00687
but diagnostic winreward.00217/terminalvalue.288: strong memorization, no durable
transfer. Corrected diagnostic winreward.01707/terminalvalue.48682; uniform
.00760/.62387. Neither yields a demonstrated useful predictor or policy fix.
Probe times128steps arms4.19/4.08s;512steps22.81/23.42/26.63s, localCPU,$0paid.
Tiny7positive diagnosticset and chronological policyshift limit inference.
Reports terminal-fit-probe-E.json and terminal-fit-probe-E-corrected.json;
scriptprobe_terminal_fit.py now reproduces uniform/balanced/corrected arms with
--steps(default128). Initial128 report predates addition of corrected thirdarm;
firsttwo arm algorithms retained. Do not promote probe weights (not saved).

Interpretation: the architecture can fit a small terminal sample, but this alone
does not solve generalization or calibration. Do not simply oversample endings
and call it a fix. No model.py, reward, planner, environment, or live policy edits.
A source review considered EfficientZero's auxiliary consistency training
(https://arxiv.org/abs/2111.00210), but no such architectural/loss extension is
implemented or justified by these diagnostics alone. Next online trial still
needs a bounded registered design; finish current comparison first.

## Entry 022 — next trial registered, not launched

Prepared selfplay-G-protocol.md: only F requestedgames/updates2048→8192; same
from-scratch seed/config,3 developmentcheckpoints,existing fixed successgate and
reserved untouched finalseeds. Fullmodel tournament finishes first. No replayloss
change inferred from terminalfit probes. Additional data/optimization is an
untested hypothesis, not a claim that scaling fixes the problem. Runtime budget
45min/$3 within remaining cap; conditional independent-seed replication explicit.
Cached training now verifies a source receipt/hash map, avoiding old .git metadata
left in a persistent workspace. Local and stagedreceipt verification plus changed
file rejection test passed; current running tournament source remains untouched.

Clarification of Entry021 terminal-value language: recurrent nextvalues are
predictions made BEFORE observing the transition outcome, not values supplied
a terminal flag and then ignoring it. Nonzero predictions on realized terminal
samples can partly reflect stochastic opponent actions/partialinformation.
The audit is an error-stratification diagnostic, not proof of a simulator,
termination-target, or causal planning bug. Agent execution still ends correctly.

## Entry023 — partial tournament reveals matchup specificity

Three already completed128game comparisons: Lana100 vsLana-init123–5/0timeouts;
Lana-init vslegacyMuZero-init77–4/47timeouts; legacyMuZero-init vsLana10060–2/
66timeouts. These are pair-specific advantages, not an overall tournamentranking
or a formal population nontransitivity proof. They show why one initialization
opponent is insufficient evidence of general competitive strength.
Before G training starts, amended protocol to retain original selection/baseline
confirmation and additionally require two frozen independentMuZero selfplay
anchors E/F on new fixed200game seedsets for each of2trainingseeds. No checkpoint,
policy or training change from these results; new finalseeds remain unused.
No thirdagent promotion until these stronger preregistered checks pass.

## Entry 024 — complete frozen all-model comparison

Completed 91 pairs / 11,648 games, equal seats, fixed two-agent geometry. Audited
all records and frozen payload hashes, preserving the original 50 batches across
a bounded allocation resume. Both comparison GPUs deleted after recovery. Full
data and paper-style plots: [tournament report](tournament-001.md). There were
5,391 timeouts and five simultaneous draws; timeout outcomes remain zero reward.
E1024 vs initialization: 61 wins / 46 losses / 21 timeouts; F1024: 51 / 54 / 23.
E vs F: 53 / 31 / 44. Neither qualifies for the registered success claim. The old
10,000-episode checkpoint won zero games against the collection; reaching a
throughput target did not establish learning. No model or third-agent promotion.

## Entry 025 — registered longer-training G launched

After Entry024, launched the unchanged registered 8,192-game/8,192-update trial
on H100 SXM80GB at $3.99/hour with 45-minute/$3 allocation guard, 2,100-second
training bound and automatic recovery/deletion. Reused the verified prepared
275-file source/weight payload: approximately 174 MB cache, 96 KB transferred.
No new package, model.py edit, reward change, scripted policy or human data.
Cumulative completed compute estimate before G: $10.22923/$50; separately
authorized cache $0.70/month. Final confirmation seeds remain untouched.

Early observation at 256 games: F and G both have 212 natural endings/44 cutoffs
and loss 4.2194437980651855. This is only an early aggregate check. Expanded the
prefix auditor to compare every saved observation/action/target/validity array
through game2,048, alongside checkpoints, counts and action histograms. Saved
traces cover two games per collection block, not every training game. A regression
checks that reordered actions with identical histograms are detected; eight
selection/provenance/tournament/prefix tests passed, plus all16 F self-comparison
trace blocks. G training and development evaluations remain in progress.
