# Arena tournament 001 — protocol and implementation log

Requested: many fast simulations comparing all named models, including untrained.
Scope frozen before results:9homepage choices, latestF1024candidate, matching
untrained legacyMuZero/unofficialDreamer4/independent8search/independent32search.
Lanauntrained alreadyin9. Total14entries/91unorderedpairs. Intermediatecheckpoint
archives are preserved but not distinct released entries. No scriptedopponents,
human demonstrations or training in this tournament. Dreamer4 is unofficial;
legacyMuZero uses the preserved older upstream-derived implementation. Neither is
misrepresented as new independentDreamer3 or the current independentMuZero.

All two-agent games use current canonicalJS homepage physics,5*.05sframes/decision,
256decisioncutoff. Trained and untrained within each family use identicalinference
settings. InitialLana digest matches browsermanifest; oldMuZero/Dreamer4/independent8
initializations reconstructed from their originalseed, labeled accordingly. E/F
use saved initialcheckpoint. Privatecatalog pins14weightfiles/SHA256; publicprotocol
omits paths. Modelsource and physics hashes pinned at invocation. PCG64 separately
seeded pergame/seat, independent of batching. Seeds change sampling, not spawnlayout.
One frozen trainingrun perentry; this evaluates fixed-layout two-agent behavior,
not population generalization, continuallearning or superhumanperformance.

Runner saves atomicperpair batches and refuses a changedprotocol onresume. Reports
wins forbothmodels, actualdraws, timeouts, 95%Wilson descriptive intervals, setup/
rollout/inference/environment seconds andgames/sec. Cutoffs earn0, not .5drawcredit.
Model loading/serialization is not included in batchthroughput; fullallocationcost
will also be reported. No significanceclaim from ranking many pairwisecomparisons.

Local validation before GPU: all14NPZweights strictload; all6pairings among4family
untrainedrepresentatives completed two-stepCPU smoketests. Those artificial2step
cutoffs are correctnesschecks, NEVER benchmarkwins or speedclaims. Three-step
batched-vs-single-stream sampledaction parity passed for each4familyrepresentative.
Existing independentsearchPython/WASM parity preserved; no inferencealgorithmschanged.

Next bounded throughputbenchmark (not started in this entry): H100<=4/hour,
<=20minuteallocation/$1.34, auto-stop. Three pairs representing workloads:
lana-100 vsdreamer4-1000; muzero-100 vsselfplay-E1024;
selfplay32-untrained vsselfplay-E1024. Each128games atparallel32 and128,
seed49020000; samepair-specific benchmarkseeds across widths. Limit256.
First profile; do not extrapolate2step smoke. Preserve allresults, even slowfailures.
Then freeze fulltournamentsize from measuredthroughput/cap before seeing tournament
scores. Candidate128games/pair=11648games (64perseat/pair), benchmarkdoesnot
count towardranking. Tournamentseeds51000000+, separate from MuZerofinal42000000.
No modelselection fordeployment or thirdagentpromotion from this exploratorymatrix.

## GPU setup diagnosis and persistent cache

The first GPU benchmark failed before any complete game. A successor captured
stderr and identified an unnecessary `safetensors` import from the legacy Lana
executable evaluator. The NPZ inference adapter now imports only inference
components and preserves the original controller's action logic. Four decisions
across two fixed CPU streams exactly matched the original controller. No package
installation, architecture edit, weight change or training occurred.

A private 10 GB persistent Runpod volume is now authorized at approximately
$0.70/month. Private setup tools stage before GPU allocation, rsync only changed
files, resume interrupted transfers, and verify SHA256. A measured second sync
verified 555 cached files in 5.614 seconds and resent no checkpoint weights
(~200 KB protocol/bytecode traffic versus ~174 MB payload). Python bytecode is now
excluded. This measures payload reuse, not total pod cold-start or game speed.
Shared storage rejects chown; rsync deliberately does not preserve ownership.
Collectors now retain stderr and partial results before deleting GPU resources.

## Full matrix registration

Before full-matrix scores are seen: 14 frozen entries, all 91 unordered pairs,
128 games per pair (64 in each seat), 11,648 games total. Seed base51000000,
256decision limit, 4 Node workers, parallel128. Benchmark128 has identical
aggregate results for the first two measured pairs and improves speed; final
third workload and per-game identity checks will be retained. These benchmark
outcomes do not select models or tune policies. Every catalog entry stays included.

Next allocation is bounded to45minutes / $3.00 at <=$4/hour, with a40minute
process limit to recover partial results and delete the pod. Resume only the
same frozen protocol if incomplete; never erase slow or losing pairs. Existing
recorded compute spend is approximately$5.795 plus current boundedbenchmark;
this allocation fits the original$50totalcap. Storage$0.70/month is separately
explicitly authorized. Full matrix remains exploratory, not promotion evidence;
reserved MuZero confirmation seeds42000000 are untouched.

## Measured throughput result

H100 SXM80GB (81559MiB), driver595.91.07, 700W power limit; PyTorch2.8.0+cu128,
4 Node workers, identical128games per workload, one measured run per width.
- Lana100/Dreamer4-1000:32parallel70.094s;128parallel60.877s;1.151x.
- LegacyMuZero100/E1024:102.718s→35.900s;2.861x.
- Independentuntrained32/E1024:169.602s→43.700s;3.881x.
ALL384 paired game records (seed/seat/winner/termination/length) exactly match
between widths. Total768 fullbenchmarkgames; smoke games excluded. These are
workload-specific wall times, not a universal GPU speedup. One GPU spot sample
showed82%utilization,1093MiB allocated,137.13W; not a time-averaged measurement.
Lana's smaller gain leaves CPU tree work as a profiling candidate, not a proven
sole bottleneck. Public compact data: tournament-benchmark-001.json.

Benchmark allocation deleted after recovery; estimated$0.63883. Recorded cumulative
compute through setup failures+benchmark~$6.43339/$50 (not accountwide billing).
Initial failed setups cost$0.62640+$0.38020 and remain logged rather than hidden.
Fullmatrix is separately registered above and not yet complete in this entry.

## Inventory and figure preparation

Strict-loaded checkpoint inventory: Lana3,358,684parameters;
legacyMuZero3,440,323;unofficialDreamer4 3,305,300;independentMuZero3,226,691.
Sizes are within~6.6% across families, but historical training budgets, activeagent
counts and planning costs differ. Do not equate similar sizes with a controlled
training comparison. Public safe inventory: tournament-model-inventory.json.
`tournament/report.py` exports pairwise win/time-limit matrices and matched-initial
win fractions with descriptiveWilson intervals (PNG/SVG/PDF), plus CSV/JSON.
Partial-matrix rendering was inspected as a generated plot, never a screenshot;
partial reports explicitly suppress aggregate rankings. Final charts stillawait
all91pairs. Plot labels distinguish percentages and fractions.

A single conditional resume is prepared: only after firstallocation recovery+
verified deletion and a timeout, reuse the exact protocol/output/source/weights,
skip completedbatches, <=30min/$2 allocation with25minuteprocess bound. Do not
resume successful91paircompletion or blindly retry another error. This raises the
maximum two-allocation tournamentbudget to$5, within original$50totalcap, without
changing match count/seeds or optimizing policies against observed scores.

## Independent record audit before final reporting

`tournament/audit.py` independently recounts every saved game's outcome, checks
expected unique seed/seat order, batchsize, valid termination/timeout length,
summary totals and timing arithmetic. Optional `--stage` verifies the preserved
payload manifest, protocol code/physics hashes and all checkpoint hashes.
Report generation now requires this record audit; it cannot call a duplicate or
incomplete summary list a complete matrix. Verified against the completed128game
independent benchmark and its260file preservedstage/45protocol hashchecks.
This is artifact integrity, not an independent reproduction of every gameplay
trajectory or proof of learning. Fullmatrix audit follows recovery, not live files.

## Completed results — October 21, 2026 UTC

All 91 pairs / 11,648 games completed. Independent record audit passed every
record, all 261 staged payload files and 57 protocol hash checks. The 50 batches
completed before the allocation timeout were reused byte-for-byte; protocol
unchanged. Both GPU allocations were deleted after artifact recovery.

[Pairwise matrix](tournament-001-results/comparison.svg) ·
[Trained vs matching initialization](tournament-001-results/matched-initializations.svg) ·
[CSV](tournament-001-results/pairwise.csv) ·
[All game records](tournament-001-results/games.jsonl) ·
[Protocol](tournament-001-results/protocol.json) ·
[Audit](tournament-001-results/audit.json). PDF versions are beside the SVGs.

Across all games, 5,391 (46.3%) reached the time limit; only five were simultaneous
draws. This remains a major limitation. Unofficial Dreamer4-100 had the highest
aggregate win fraction against this particular collection (42.2%), but tied its
matching untrained model 55–55 with 18 timeouts. That does not demonstrate a
training benefit. Lana-100 beat its initialization 123–5, yet its aggregate win
fraction was only 22.7% with 62.0% timeouts. Opponent-specific success is not
general competence. Historical models also differ in training budgets and agent
counts, so this is not a controlled architecture ranking.

Independent E1024 scored 61 wins / 46 losses / 21 timeouts against initialization;
F1024 scored 51 / 54 / 23. E beat F 53 / 31 with 44 timeouts, even though F had a
higher collection-wide win fraction. Neither earns promotion under our registered
gates. The old independent 10,000-episode checkpoint won zero of 1,664 games;
its historical throughput target did not produce a useful policy. All negative
checkpoints and evidence remain preserved. No final confirmation seeds were used.

Full tournament allocation estimates: $2.71537 + $1.08047 = **$3.79584**, including
startup/recovery time. Cumulative recorded investigation compute ~$10.22923/$50;
these are elapsed-time estimates, not a reconciled invoice. Persistent cache
$0.70/month is separate. No claim of superhuman or continual-learning performance.

The next experiment remains the already registered G duration comparison. These
exploratory scores do not change its architecture, loss, seed, checkpoint selection
rule, confirmation seeds or promotion gates.
