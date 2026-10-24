# Unvisited-action Q completion: frozen diagnostic

The current independent search sets unseen raw Q to zero, then min/max-normalizes
all actions. Google Research MuZero instead leaves the final unseen Q score zero:
[core.py at 5b09c22](https://github.com/google-research/google-research/blob/5b09c22d73a9d35eb6c5d2a99b95677a45053466/muzero/core.py#L507).
These differ whenever the minimum is nonzero and bounds differ. Positive minima
penalize unseen actions; negative minima boost them. This is not universally
underexploration, and no causal strength conclusion follows from source comparison.

We tested a single in-memory change: after the existing normalization, set unseen
actions' Q score to zero. Original model and trainer files remain unchanged.
Source-extracted arithmetic fixtures cover positive, negative, mixed, zero and equal
bounds. A deterministic five-root neural fixture verifies scalar versus batched,
dynamic versus static execution, legal masks and independent batch streams.
Both tests passed on CPU.

Frozen development diagnostic: iteration22 fresh640 weights, 32 evenly spaced valid
histories from its trace512 (agents0/1), 32 simulations, eight identical initially
forced root candidates per condition, seed23001, no noise, no optimizer updates.
Checkpoint SHA256: `171e979f9d78aa22d192cb5f4e0970009679cd25b990c728820b0bc2be92a568`.
The probe records exact source/trace/weight hashes, selected roots, candidates and
all policy/value outputs in private evidence.

31/32 visit distributions and 8/32 argmax actions changed. Mean total variation was
0.1201171875. Mean root action support increased from 8.0 to 9.21875. This establishes
decision sensitivity on these histories, not better decisions or stronger play.
Whole-search equivalence to Google is not claimed: other search conventions remain.

Next experiment, if admitted, should change only this Q-completion rule, retain the
existing loss and reward, use fresh paired training seeds and matched frozen
development opponents. A changed-search run is a new experiment, never exact
continuation of the original checkpoint. Confirmation remains separate and unused.
