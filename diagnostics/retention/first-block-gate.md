# UPG2 strict first-block gate

This external wrapper imports the frozen trainer without editing it. Run it from
an independent diagnostic checkout, supplying the unchanged trainer source
`2e67c2b9b7652ccbe75dbed249708356f38c3608`. It replaces only the Python `collect`
call boundary: the original function executes, and its exact returned arrays and
statistics are inspected before control can reach replay insertion or an update.
Comparison uses no Torch or NumPy RNG operations. Subsequent candidate blocks call
the original collector without checks. Source hashes, arguments, initial tensors,
environment, Python, PyTorch and GPU name bind both passes to the archived baseline.
The separate intended-CUDA exact-resume gate remains required.

The old `trace-0.npz` contains only slots0–21: two complete games, **not all128**.
All128 aggregate non-timing statistics are also archived. The wrapper checks both
against SHA256-pinned archive inputs. It does not call that full trajectory parity.
A new frozen ratio1 reference pass collects all128 games, saves every returned
array including replay indices, then raises a dedicated sentinel before updates.
Trainer `finally` closes its environment pool. This reference performs zero optimizer
updates. The real ratio2 candidate compares all full reference arrays exactly,
in addition to the old slice/aggregates and initial tensors. Failure raises and
terminates training before its first update. Success returns the original objects
and continues the same trainer process; no restart, seed reset or state reload.
Later candidate trajectories intentionally diverge after its extra updates.

Run both with the full frozen production arguments (2048 games, parallel128,
seed52731415, hidden448/context64, simulation32, batch256, unroll5, horizon256,
2agents, baseline loss, episode returns, zero timeout bootstrap, CUDA graphs,
device history and TF32). Only ratio and output differ. The reference receives
`--updates-per-game 1`; the candidate receives `--updates-per-game 2`.

```
python DIAGNOSTICS/first_block_gate.py --source FROZEN_SOURCE --baseline ARCHIVED_BASELINE --mode reference --reference NEW_REFERENCE --output NEW_REFERENCE -- PRODUCTION_ARGS_WITH_RATIO1
python DIAGNOSTICS/first_block_gate.py --source FROZEN_SOURCE --baseline ARCHIVED_BASELINE --mode candidate --reference NEW_REFERENCE --output NEW_CANDIDATE -- PRODUCTION_ARGS_WITH_RATIO2
```

Do not pass `--output` again after `--`. Run the candidate command inside the
existing checkpoint encoder/supervisor, using the same candidate output binding.
The reference is a prerequisite job; candidate depends on successful reference
exit and on the separate CUDA resume gate. Archive the reference receipt, full
block, initial weights and manifest, plus candidate gate receipt. Receipts bind
full-block/initial/manifest hashes; the candidate records the reference receipt hash.
Reference success is diagnostic collection, never a completed training endpoint.
No default or override permits partial trajectory parity.

Historical first128 collection consumed about23.0 seconds plus1.63 seconds setup
on H100, excluding the new full-block compression/check. Budget a separate120second
reference task, with a finite supervisor deadline; this is a provisional allowance,
not measured new throughput. Production comparison may add up to60seconds reserved
inside the candidate's outer task deadline; it does not add optimizer work.
At $3.99/hour the extra180seconds provision is $0.1995. No execution or compute
admission is implied by this document.

A full block at the frozen maximum horizon occupies at most326,354,688 raw array
bytes (including a conservative maximum indices count). Reserve335,544,320 bytes
for its NPZ (compression is not assumed), plus12,582,912 initial-weight bytes and
1MiB metadata:349,175,808 bytes per retained copy. Reserve both recovered and
immutable archive copies locally (698,351,616 bytes), and a reference copy remotely.
Existing candidate/runtime memory must additionally permit decompressing the largest
reference array (about222MiB); allow400MiB CPU RAM overhead. Payload inputs are
12,145,554 bytes (the four pinned archive files). Add all these costs to the exact
job storage plan, outstanding obligations and headroom; never silently count zero.

Tests cover stop-before-update, full unsaved-game/indices differences, initial
mismatch, archive/reference corruption, aggregate mismatch, dtype/keys/nonfinite
failures, source-validation failure, repeat-call behavior and unchanged NumPy RNG.
Mock tests do not establish actual production CUDA equality. Both production gates
must still run and pass on the admitted intended runtime.
