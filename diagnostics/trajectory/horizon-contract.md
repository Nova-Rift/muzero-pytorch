# Synthetic finite-horizon input counterexample

`horizon_contract.py` imports the actual archived M8 `arena.train_selfplay.history`
function, checks its module path, and requires trainer SHA256
`9d373ae8ef63d131b003a87db992e4429e14fa38eaac5e4483567fe3971ca18a`
(source revision `d10af2d72dbf03dfec113c0b6e7e47481c5cc88d`). No model,
trainer, environment, adapter or checkpoint is changed. No games or fitting run.
The proof concerns this NumPy history function; it does not validate every adapter.

The fixture gives two synthetic agents the same nonconstant trailing 64 observations
and 63 actions at decision times 226 and 246, with different discarded prefixes.
Both times are beyond the initial padding. The actual encoder yields exactly the
same 4,159 float32 inputs, including historical action division by 153. A proposed
extra scalar `(256 - t) / 256` distinguishes these inputs: 0.1171875 versus
0.0390625. This scalar is diagnostic only; it has not been added to a model.

Here `t` counts completed transitions; the next action starts transition `t`.
A hypothetical reward of 1 on the twentieth future transition has return 1 with
30 transitions left, and 0 with 10 left, under discount 1 and zero cutoff bootstrap.
A reward on the final available transition is included. This is an **analytic
counterexample**, not an actual game trace or evidence that the two windows are
reachable in the arena. It shows that the rolling input need not identify the
remaining horizon required by this hypothetical finite-horizon target.

Tests exercise the actual historical function at t=0, 1, 62, 63 and 64, including
zero padding and the first full window; prefix irrelevance and suffix sensitivity;
clock bounds and malformed times; inclusive terminal-return boundaries; and source
hash rejection. Historical-source tests are required, never silently skipped.

Run using an available archived M8 model directory and an ignored private output:

```bash
M8_SOURCE="$ARCHIVED_M8_MODEL" python3 -B -m unittest discover -s diagnostics/trajectory -p test_horizon_contract.py -v
python3 -B diagnostics/trajectory/horizon_contract.py --source "$ARCHIVED_M8_MODEL" --output "$PRIVATE_OUTPUT"
```

Absence of an explicit clock does not imply absence of time inference: observations,
action histories and early padding may correlate with age. No causal weakness or
playing-strength benefit follows from this fixture. A clock-conditioned learned
value could harm anytime arena behavior or mismatch another evaluation cutoff.
Any proposed new contract needs trainer/replay/device-history/planner/browser
adapter validation and an explicit compatible-weight warm-start or restart plan;
changing the input dimension is not exact resume. Frozen held-out comparisons would
still be needed before any advancement claim.
