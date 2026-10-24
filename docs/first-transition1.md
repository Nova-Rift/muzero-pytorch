# First recurrent hidden gradient experiment

Independent hypothesis from pinned gradient audit `988356e1e064d08aee9aa35c3fd2121eeb0f8f16`:
local hidden inputs are scaled by 0.5 before every recurrent transition, whereas
the audited reference leaves the first transition unscaled and halves later ones.
This difference is separate from root/recurrent loss weighting. Reference audit
used source AST bodies with a Torch shim, not native TensorFlow network parity.

`first-transition1-v1` changes only the first recurrent hidden-input backward
factor from 0.5 to 1.0. It preserves the original coefficient 1/(K+1), all head
masks and reward indexing, original search/Q semantics, network architecture,
reward contract and initialization. It does not combine root5 weighting.
`arena/first_transition.py` subclasses MuZero with an exact copy of its loss;
the one conditional scale is the sole loss-body change. Canonical model.py,
train_selfplay.py, train_loss.py and rootweighted.py remain byte-identical to
base `2e67c2b9b7652ccbe75dbed249708356f38c3608`.

Use `arena/train_first_transition_fixed.py` for the runtime gate. It refuses an
explicit --loss-contract override and invokes the separate trainer with
`first-transition1-v1`. The separate trainer also supports baseline/root5 for
explicit controlled checks. Generic resume argument and exact source/PyTorch/
environment guards remain intact; all six cross-contract pairs reject before
model-state loading. No checkpoint contract migration is allowed.

CPU float64 tests verify exact forward loss (dense, sparse and default masks),
unchanged root-only gradients, twice the representation gradient from each isolated
recurrent policy/value/reward target, unchanged downstream head/dynamics gradients,
0.5 attenuation for every later transition, reward mask alignment and zero masks.
These are correctness tests, not training-strength evidence or GPU exact-resume
validation. Tiny full/split resume validation must follow the final source freeze.

A pre-existing baseline may be reused only after the coordinator verifies its exact
original behavior, initialization, configuration, environment and artifact provenance.
This new revision adds files only; benign cross-source comparison must be explicit.
The candidate starts fresh with the same seed/initial tensors; it never resumes a
baseline or root5 checkpoint. Otherwise use a freshly paired baseline with separately
costed storage/recovery. Main owns admission, final protocol, publication and any
promotion. No current experiment outcomes were inspected for this preparation.

Run `python3 -m unittest discover -s tests -p 'test_first_transition.py' -v` and
the existing `test_rootweighted.py` suite. Private preparation logs live in ignored
`.evidence/`; no GPU jobs, optimizer training, publication or model promotion were
performed by this helper.
