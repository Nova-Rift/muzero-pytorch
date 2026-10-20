# MuZero — independent PyTorch implementation

A readable implementation written for Nova-Rift from the MuZero paper. This is
**unofficial, newly implemented, and not a reproduction of published scores**.
No third-party model implementation is cloned, imported, or vendored.

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

## Speed work

[Training performance journal](docs/training-performance.md) records the 10,000-game/
10-minute target, methods, measurements, hardware and tradeoffs. Unmeasured ideas
are explicitly marked. No game rules or rewards are changed to meet the target.
