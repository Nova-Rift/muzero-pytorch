# MuZero — independent PyTorch implementation

A readable implementation written for Nova-Rift from the MuZero paper. This is
**unofficial, newly implemented, and not a reproduction of published scores**.
The core model and training implementation do not import third-party model code.
Optional historical comparisons load separately preserved baseline evaluators
from a website checkout; those baselines are identified in the tournament protocol.

## Read the model

[`model.py`](model.py) contains the entire neural architecture and search:

1. **Representation h** encodes observations/history into a latent state.
2. **Dynamics g** predicts the next latent state and reward after an action.
3. **Prediction f** predicts policy and value from a latent state.
4. **MCTS** explores learned transitions, backs up rewards and values, and produces
   a visit-count policy target. It never sees the actual simulator.
5. **Unrolled learning** fits reward, n-step value and search-policy targets.

Includes transformed categorical value/reward supports, latent min/max scaling,
PUCT with value normalization, root Dirichlet noise and recurrent gradient scaling.
`nstep_returns` distinguishes terminal states from time-limit bootstrapping.

## Run

Python 3.10+, PyTorch. CUDA is selected automatically when available.

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python train.py --updates 10
```

The example interacts with a tiny two-action cue task. It checks the training path;
it is not an arena benchmark, trained checkpoint, or learning-performance claim.
For custom environments, provide flattened observation histories to `initial`,
use `search` for actions, and pass replay tensors to `loss` (docstrings specify
alignment). Keep each agent's history private and independent.

## Fidelity and scope

Reference: Schrittwieser et al., [MuZero](https://arxiv.org/abs/1911.08265).
This vector MLP version uses the paper's core learned-model/search/training design,
not its image ResNet or distributed Atari training stack. Search is single-player:
rewards belong to the controlled agent; no alternating-player sign flip. Scalar
support size and all dimensions are configurable. Uniform replay examples,
optimizer schedule and default sizes are engineering choices, not paper settings.
Prioritized replay, reanalysis, image encoders and benchmark reproduction are not
implemented. Legal masks apply only at the real root.

## Measured lessons

[What slowed learning and evaluation](docs/training-lessons.md) explains the timing
defect, measured GPU batch speedups, rare-event fitting failures, and why one
untrained opponent is insufficient evidence of strength.

## Speed work

[Training performance journal](docs/training-performance.md) records the 10,000-game/
10-minute target, methods, measurements, hardware and tradeoffs. Unmeasured ideas
are explicitly marked. No game rules or rewards are changed to meet the target.

## Current arena training

`arena/train.py` imports the canonical website TypeScript physics through Node22
rather than reimplementing the game in Python. The Python model/training is ours.
Pass an existing website checkout as `--site`; no account keys are needed:

```sh
python arena/train.py --site /path/to/clarkwhitehead --output runs/arena \
  --games 10000 --parallel 128 --workers 8 --cuda-graph --max-seconds 600
```

`--cuda-graph` requires CUDA; omit it on CPU. Defaults use64observation history,
153actions,8search simulations,5-step unroll,one optimizer update/game, and batch32.
A rollout ends naturally or at the unchanged256decision cutoff; logs separate both.
The600second guard is checked between batches, so the final batch can overshoot;
the report uses measured wall time and never claims success from a projection.
Saved NPZ weights are inference snapshots, not full optimizer/replay resume files.

Terminal replay extensions train zero future reward/value (absorbing states).
Time-limit extensions are masked and retain the actual final-observation bootstrap
for preceding n-step targets. Policy targets only exist at real decision states.
The first speed pilot lacked absorbing extensions; its logs are retained and
clearly marked in the performance journal.

## Recorded speed result

The corrected arena run processed **10,000 capped episodes and 10,000 optimizer
updates in 5m15s on one H100**. Only 2,191 episodes ended naturally; 7,809 timed out.
This meets the throughput target, not a policy-quality benchmark. The
[performance journal](docs/training-performance.md) includes raw evidence, all
optimizations, hardware, cost estimates and unsuccessful attempts. No website
model has been replaced by this checkpoint.

## Self-play learning investigation

[Investigation journal](docs/self-play-investigation.md) records the observed
policy collapse, confirmed integration timing bug, controlled attempts and
evaluation gates. Training uses only self-play from random initialization:
no demonstrations, scripted opponents or tactical reward shaping.
The curriculum starts with two agents; increasing the count requires a recorded
held-out performance gate. A speed result is not a policy-quality result.
`arena/train_selfplay.py` is the isolated investigation trainer; original
`arena/train.py` and `model.py` remain preserved for historical reproduction.
