# Source-bound loss gradient audit

Five CPU tests confirm different gradient contracts without changing either trainer.
Local `model.py` SHA256 is
`c546948594aa70456b9361ccd4e5acc82fa2790b3d3c9d069fb769f095888e17`.
Google learner revision is
[`5b09c22d73a9d35eb6c5d2a99b95677a45053466`](https://github.com/google-research/google-research/blob/5b09c22d73a9d35eb6c5d2a99b95677a45053466/muzero/learner.py),
file SHA256 `c5b2fedee2bce4da4ef9bba42f4583099f81b8fccd3cce629a12b716b3ba62fe`.

The diagnostic calls the real local `MuZero.loss` with instrumented inference outputs.
It executes the pinned Google `compute_loss` and `scale_gradient` function AST bodies
unchanged, removing only annotations, with a small Torch implementation of the TF
operations used. Google source is supplied separately and hash checked; it is not
vendored. This checks source logic using PyTorch 2.2.0 CPU float64 autograd, **not
native TensorFlow numerical parity or full network/training equivalence**. No third-party
network runs. Both use identical zero logits, target distributions, masks, unit head
coefficients, zero regularization, and uniform importance weights. Google batch mean
is multiplied by batch size to compare with local batch sum.

With five recurrent steps and all targets valid:

| Term | Local gradient coefficient | Reference coefficient |
| --- | ---: | ---: |
| Root value/policy | 1/6 | 1 |
| Each recurrent value/policy/reward | 1/6 | 1/5 |
| Root reward | absent | explicitly masked out |

Reference root:individual recurrent weighting is 5:1; local is 1:1. Global loss
scaling cannot remove that difference. Sparse masks are tested separately by head:
with one valid recurrent policy, two values, three rewards, reference scales are
1, 1/2, 1/3 respectively; local remains 1/6 for each valid term. A second batch
row has no valid recurrent targets; it gives zero recurrent gradients and finite
root gradients, without a zero-denominator failure.

Independent logit leaves expose every term's gradient. An identity hidden-state
fixture additionally isolates one policy target at each depth. With only that target
valid, root hidden gradient for local versus reference is:

| Target depth | Local | Reference |
| --- | ---: | ---: |
| 0 | 1/12 | 1/2 |
| 1 | 1/24 | 1/2 |
| 2 | 1/48 | 1/4 |
| 3 | 1/96 | 1/8 |
| 4 | 1/192 | 1/16 |
| 5 | 1/384 | 1/32 |

The isolated recurrent ratio of 12 combines loss weight (sixfold with one valid
recurrent target) and first-transition hidden scaling (twofold). Later transitions
halve gradients in both. These ratios are fixture-specific, not measured trained
representation gradient norms. Reference `scale_gradient` preserves forward losses;
local division also scales forward losses, so displayed loss magnitudes are not
interchangeable even after batch reduction is aligned.

Run with a separately obtained pinned learner file:

```sh
REFERENCE_LEARNER=/path/to/learner.py python3 -m unittest discover -s diagnostics/gradients -v
python3 diagnostics/gradients/audit_loss.py --reference-learner /path/to/learner.py --output /private/results.json
```

Without `REFERENCE_LEARNER`, tests explicitly skip rather than silently substitute
an invented reference. SHA mismatch fails closed. No optimizer steps, games,
checkpoint loads, GPU work, installations, model edits or strength claims. The
iteration22 archive reports 32 wins/30 losses/2 cutoffs against its own initial
checkpoint; this test does not explain those outcomes. Next useful work is a
frozen real-minibatch measurement of head and representation gradients, then a
separately registered single-change training comparison if justified. Neither
contract is declared universally correct for the arena.
