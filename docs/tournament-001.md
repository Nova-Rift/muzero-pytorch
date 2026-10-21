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
