# MuZero training performance journal

## Completed result

**10,000 capped training episodes and 10,000 optimizer updates completed in
314.907 seconds (5 minutes 15 seconds), including the full Python process.**
The H100 was terminated after local checkpoint verification; the account pod list
then contained zero pods.

- 2,191 episodes ended naturally; 7,809 hit the unchanged time limit.
- 2,259,249 joint decisions and 20,855,529 active-agent transitions were collected.
- 320,000 replay sequences were sampled for optimization (batch 32 × 10,000).
- Training plus checkpoint writing took 311.603 s; full process took 314.907 s.
- Checkpoint SHA-256: `914ab929002ea50707b5becd71b72ef8318b22441a70e9d373eeb35e96bf2811`.
- Trained source: `51f835c`; protocol: `muzero-speed-10000-002`.

The speed target is met **for capped episodes**, not 10,000 natural game endings.
Later batches all reached the cutoff, so this is not evidence of strong play.
The checkpoint has not been deployed to the homepage or evaluated as a stronger
policy against held-out opponents. Finite weights and decreasing loss alone do
not establish useful learning.

[Report](benchmarks/muzero-speed-10000-002/report.json) ·
[Full-process timing](benchmarks/muzero-speed-10000-002/process-timing.json) ·
[Raw metrics](benchmarks/muzero-speed-10000-002/metrics.jsonl) ·
[Manifest](benchmarks/muzero-speed-10000-002/manifest.json) ·
[Checkpoint/work-count audit](benchmarks/muzero-speed-10000-002/audit.json) ·
[Compute estimate](benchmarks/muzero-speed-10000-002/compute-cost.json)

Weights are retained locally at `runs/muzero-speed-10000-002/weights.npz` in the
researcher's checkout; large weights are not committed to this repository.

## Target and counting

**10,000 arena episodes and 10,000 optimizer updates in at most 600 seconds.**
An epoch here means one arena rollout, consistent with the previous arena runs.
Episodes end naturally or at the unchanged 256-decision cutoff. We report both
counts: a cutoff is not a naturally completed game or a win.

The final timing includes the training Python process, initialization, collection,
search, updates, checkpoint writing and worker cleanup. Cloud provisioning and
software installation are reported separately from training. Projections and short
benchmarks do not establish that the full target was met.

The latest game remains the canonical website arena (`sandboxCombat.ts` and its
observation module). Reward is 1 for the winner and 0 otherwise. There are 11 agent
slots, including the existing slot-10 size/fire-rate asymmetry. No physics, action
space, observation information, reward or cutoff was changed for speed.

## Hardware

| Item | Actual configuration |
|---|---|
| GPU | One NVIDIA H100 80GB HBM3, 81,559 MiB reported |
| Power | 700 W configured limit; not measured consumption |
| Driver / CUDA / PyTorch | 595.91.07 / 12.8 / 2.8.0+cu128 |
| Host CPU | Intel Xeon Platinum 8480+ |
| Allocated CPU / RAM | 28 vCPUs / 251 GB; host-wide CPU count is not the allocation |
| Listed GPU price | $3.99/hour; storage and final invoice are separate |
| Model | 3,226,691 parameters, hidden width 448, 64-observation history |
| Training | Batch 32, unroll 5, one Adam update per episode |
| Search | 8 MCTS simulations per decision; no fixed depth-3 cap |
| Parallelism | 256 worlds, 8 canonical JavaScript environment workers |
| Numerics | FP32 tensors with TF32 matrix multiplication enabled |

The new implementation is independently written, not the earlier third-party
MuZero network. Similar parameter counts do not make earlier results directly
comparable. No paper-score reproduction or useful-policy claim is implied.

## Optimization ledger

| ID | Method | Status / evidence | Tradeoff or limitation |
|---|---|---|---|
| P001 | Batch independent worlds and agents in model calls | Implemented; 128- and 256-world measurements below | Policy stays fixed while each world batch is collected |
| P002 | Tensor-backed batched MCTS | Matches the scalar reference in tests, including independent streams and visit counts | Same search budget; not an independent speed ablation |
| P003 | Parallel canonical JS workers | Implemented; binary and original JSON bridge matched through 256 steps | Imports exact game rules; no Python physics fork |
| P004 | Binary float32 transport | Observations, rewards, alive flags and termination matched | Float32 observation contract retained |
| P005 | Larger replay training batches | Not enabled: batch remains 32 | More games per collection batch does not mean larger optimizer batches |
| P006 | CUDA graphs and TF32 | GPU graph parity passes, including changed model weights | TF32 changes arithmetic precision; no separate accuracy ablation |
| P007 | Keep histories on the model device | Padding, previous actions and rollover match replay histories at contexts 1/4/64 | Transfer each current observation once; no history information removed |

Component timings are CPU wall times. GPU history operations finish at a later
synchronization inside search, so those components are not independent CUDA-event
measurements. Whole-process throughput is the deciding metric.

## Measured runs

| Run | Episodes / updates | Natural ends / cutoffs | Loop time | With checkpoint | Loop throughput |
|---|---:|---:|---:|---:|---:|
| Initial H100, CPU histories, 128 parallel | 128 / 128 | 123 / 5 | 13.5229 s | 14.0133 s | 9.47 episodes/s |
| Device histories, 256 parallel | 256 / 256 | 247 / 9 | 11.1342 s | 11.6436 s | 22.99 episodes/s |

Raw records: [initial 128](benchmarks/benchmark-128.json),
[device histories + 256](benchmarks/benchmark-history-256.json).
The combined change improved short-run loop throughput by 2.43x. This is not a
one-factor measurement of history placement alone, and neither run completed 10k.

The first run spent 6.7574 s preparing/transferring histories, 3.0419 s in search,
0.6753 s in environments and 2.2787 s in updates, plus 0.5169 s setup. The second
spent 0.0654 s enqueueing device histories, 2.6621 s in search, 1.0824 s in
environments and 6.3332 s in updates, plus 0.5117 s setup.

A separate laptop environment-only test ran 64 random-action episodes in 0.3947 s:
61 natural ends, 3 cutoffs, 7,223 joint steps and 29,222 active transitions. Host:
Intel i7-1165G7, 4 cores/8 threads, Node 22.22.2. This is a simulator/transport
ceiling, **not neural training throughput**.

```sh
python arena/benchmark_environment.py --site SITE --games 64 --workers 4
```

Historical context only: the earlier third-party-network pilot took 1,682.29 s for
1,000 games on RTX 4090, with 8 searches/depth 3 and one update/game. Architecture,
search and pipeline differ, so no controlled speedup is claimed against it.

## Corrections and unsuccessful attempts

- The RTX 5090 allocation failed. A read-only account check confirmed no pod had
  been created before selecting H100.
- The first CUDA graph capture failed on a CPU scalar used in indexed assignment.
  Creating that assignment tensor on-device fixed it; GPU parity then passed.
- The first H100 allocation was deleted at Clark's explicit pause. Its benchmark
  report was not recovered and is not counted as a result.
- `/usr/bin/time` was absent in the container. That launch failed before training;
  a Python subprocess timer now records full-process wall time and exit code.
- The preliminary 10k run was deliberately interrupted for the terminal-target
  correction below. Its [completed-batch logs](benchmarks/preliminary-interrupted.json)
  are retained. It is not a completed 10k result.

## Training fidelity correction

The preliminary loss masked value/reward targets after true termination. MuZero
expects terminal states to be absorbing during training ([paper, Appendix A](https://arxiv.org/html/1911.08265v2)).
The corrected replay supplies zero future reward/value targets after termination,
while leaving unavailable time-limit continuations masked. Policy targets are
present only at actual decision states. Tests distinguish terminal and truncated
extensions and verify that the model uses the masks.

This is a fidelity correction, not a speed optimization or a proven explanation
for policy collapse. The preliminary run eventually produced all-cutoff batches
despite decreasing loss. That negative evidence is preserved: low loss does not
establish useful play.

[Protocol 001](muzero-speed-10000-001.json) preserves the interrupted design.
[Protocol 002](muzero-speed-10000-002.json) freezes the corrected run with fresh
training seed 30000000 and otherwise unchanged hardware, model and work budgets.
No held-out policy-quality assessment or checkpoint selection is performed here.

## Reproduce and audit

```sh
python arena/train.py --site SITE --output runs/speed-002 \
  --games 10000 --parallel 256 --workers 8 --hidden 448 \
  --cuda-graph --tf32 --device-history --seed 30000000 --max-seconds 600
python scripts/audit_run.py runs/speed-002 --expected-games 10000
```

The time guard is checked between batches and may overshoot. A success claim uses
actual elapsed time, not the configured limit. Save an external process timer as
`process-timing.json` alongside the report for the strict full-process audit.
NPZ files contain inference weights, not optimizer/replay resume state. Checkpoint
hashes, finite tensors, parameter counts and cumulative work counts are audited.
