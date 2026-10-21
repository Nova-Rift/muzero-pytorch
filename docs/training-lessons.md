# What has slowed learning and evaluation so far

This is a teaching summary of measured findings, not a claim that MuZero is now
strong or superhuman. The larger comparison is still running. The full attempt
history, including negative results, is in [the investigation log](self-play-investigation.md).

## Make the experiment correct before making it faster

An early training bridge advanced **six** physics frames per decision; the
homepage advances **five**. Repeatedly subtracting 0.05 from 0.25 left a tiny
positive floating-point remainder, so a `while (clock > 0)` loop ran once more.
One unobstructed move traveled 66 pixels instead of 55. Comparing two bridges
with the same mistake did not catch it; comparing against the real homepage loop
did. New runs explicitly use five frames. Old checkpoints and their negative
results remain preserved, with their training cadence labeled.

**Reusable lesson:** test the actual consumer's timing and observation/action
alignment, not just two implementations that may share an assumption. This was a
real integration defect; it is not proof that timing explains every weak policy.

## Batch independent games, and avoid paying for repeated setup

On an H100 SXM with 80 GB HBM3 (81,559 MiB reported), PyTorch 2.8.0/CUDA 12.8,
four Node workers, and 128 games per workload:

| Workload | 32 parallel games | 128 parallel games | Speedup |
|---|---:|---:|---:|
| Lana-100 vs. unofficial Dreamer4-1000 | 70.09 s | 60.88 s | 1.15× |
| Legacy MuZero-100 vs. independent MuZero E | 102.72 s | 35.90 s | 2.86× |
| Independent MuZero initialization vs. E | 169.60 s | 43.70 s | 3.88× |

All 384 corresponding game records—including outcomes and lengths—matched
exactly across batch sizes. These are single measurements for these workloads,
not a universal speed guarantee. Larger batches help amortize overhead and use
the GPU better; they do not automatically make the policy smarter. The older
Lana search still does substantial Python tree work, so its smaller gain needs
profiling before recommending a larger GPU.

A private 10 GB network volume now preserves the approximately 174 MB model
payload and runtime. A fresh pod sent about 109 KB of changes instead of all the
weights. A same-pod reuse check took 5.6 seconds. Storage is approximately
$0.70/month, separately authorized; GPU time is separate. Source is prepared
before allocation, transfers resume, and hashes are verified. The source receipt
also prevents old cached Git metadata from being mistaken for the current code.

See [benchmark measurements](tournament-benchmark-001.json) and
[the tournament protocol](tournament-001.md) for hardware, costs, and boundaries.

## A small average loss can hide the events that matter

In 7,130 preserved training-derived transitions, only 29 contained a winning
reward. Predicting nearly zero everywhere can therefore look good under an
aggregate error metric while failing on the very event the planner needs to
recognize. Reward error, terminal value error, and actual game wins must be
reported separately.

At E1024, the imagined next-state value averaged about 0.294 on transitions that
actually terminated, whose realized next-value target is zero. This prediction
was made **before observing the outcome**. Stochastic opponent actions and
partial information can explain some conditional error; it is not evidence that
the simulator keeps acting after death or that the terminal targets are wrong.

A copied-network probe could fit 44 terminal examples when half its training
samples were terminal. After 512 updates, its fitting-set winning reward averaged
0.880, but the seven winning events in later diagnostic traces averaged only
0.002. That is memorization without demonstrated transfer. Correcting for the
altered sampling frequency did not resolve the weakness either. No probe weights
were deployed, and the game's reward was not changed.

**Reusable lesson:** inspect rare-event errors and a separate diagnostic split.
Improved fitting loss, larger batches, or oversampling are hypotheses to test,
not evidence of better control. The next registered trial tests more self-play
experience and updates while keeping the model and loss unchanged.

## One easy opponent is not a sufficient success test

Completed exploratory matchups already show opponent-specific behavior:

| Matchup | Wins for first / second | Timeouts |
|---|---:|---:|
| Lana-100 vs. untrained Lana | 123 / 5 | 0 |
| Untrained Lana vs. untrained legacy MuZero | 77 / 4 | 47 |
| Untrained legacy MuZero vs. Lana-100 | 60 / 2 | 66 |

These results do not establish an overall ranking. A model can exploit one
opponent and struggle against another. The next trial must therefore confirm
against its matching initialization **and** frozen earlier independent self-play
policies, across two training seeds, before adding a third agent.

Checkpoint selection uses development games. Selected hashes are frozen before
new final games, and failures cannot trigger reselection on those final games.
Timeouts earn zero and are shown separately from simultaneous draws. See the
[registered G protocol](selfplay-G-protocol.md) for exact gates and budgets.

## Keep speed, fitting, and competence separate

- **Speed:** completed games, transitions, updates, wall time, hardware, and cost.
- **Fitting:** what the model predicts on familiar and separate diagnostic data.
- **Competence:** frozen-policy game outcomes against meaningful opponents.

None of these alone demonstrates continual learning, broad generalization, or
superhuman play. The current evidence has not met the registered learning-success
gate. Preserving failures and exact artifacts is part of finding a real fix.

## Lana inference batch limit (October 21, 2026 UTC)

A naive increase from 32 to 128 failed with `Invalid latent tensors`: the preserved
legacy model validates a maximum batch of 32. Its neural calculations are batched,
but its state contract blocks larger calls. We did not edit `model.py`. An opt-in
evaluation adapter validates every 32-row slice through the original validator,
checks matching leading dimensions and an explicit larger total bound, and refuses
training mode. The planning helper now accepts a batch size; its default stays 32.

A real Lana-100 CPU check compared 65 streams over four decisions at batch sizes
32 and 128: sampled actions matched exactly and all parameter/buffer bytes stayed
unchanged. Invalid states after row 32, mismatched dimensions, excess batches and
training mode were rejected. Evidence: [CPU check](lana-microbatch-cpu-check.json).
This is not yet a GPU speed measurement or full-game parity test. The running
91-pair tournament retains its frozen original code and batch size.
