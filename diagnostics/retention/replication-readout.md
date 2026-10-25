# Independent-seed readout

`readout_replication.py` is a CPU evidence reader for the conditional fresh seed
73142017 pair. It does not choose an original arm, train, evaluate, admit a job,
or publish a model. Null initialization contracts remain invalid. Original EGO/MIR
selection and the real CUDA initialization/parity/collection gates remain external
prerequisites, enforced by the coordinator's future prepared job plan.

Run only after recovering complete artifacts:

```
python3 diagnostics/retention/readout_replication.py request.json \
  --request-sha256 PREDECLARED_REQUEST_SHA256 --output new-readout.json
```

The output must not already exist. Malformed/missing evidence raises an error and
writes no readout. A scientifically failed gate is valid audited evidence and is
reported as `gate.passed=false`, not a process failure.

The request has exactly these fields:

- `schema`: integer 1; `phase`: `development` or `confirmation`.
- `candidate`, `paired`: recovered final training directory paths, containing
  `manifest.json`, `report.json`, `initial.npz`, `weights.npz`.
- `contract`, `source_receipt`, `initialization`, `initial`, `template`,
  `evaluation_environment`: each `{path, sha256}` pinned before readout. Template
  retains `initial_sha256:null`; frozen contract differs only in that hash.
- `evaluation_environment` JSON contains exactly `gpu` and `legacy_sources`, frozen
  from the actual prepared site/runtime; GPU must match both manifests and the
  initialization receipt. Legacy source paths/hashes match the tournament protocol.
- `opponents`: exact E1024/F1024/OwnInitial development set, or E1024/F1024
  confirmation set. Each value has `entry` (pinned original opponent JSON) and
  `checkpoint` (pinned recovered weight file), each `{path, sha256}`. Original
  catalog paths remain unchanged; recovered files may live elsewhere.
- `bouts`: exactly one `{side, opponent, directory}` per side (`candidate`,
  `paired`) and opponent. Each directory contains the adapter's `catalog.json`,
  `evaluation-receipt.json`, and `results` directory with exactly protocol,
  summary, and one 64-game raw bout JSON.

Readout validates both endpoint recipes/counters/checkpoints against the immutable
replication contract. It binds catalogs and receipts to source, initializer,
evaluator, contract, runtime and opponent entries. Every protocol must exactly
match the frozen tournament fields. The unchanged raw audit checks contiguous
seeds, alternating 32/32 seats, all outcomes and summary agreement. Frozen helper
SHA256s are verified before use; `readout.py` and `readout_3arm.py` remain unchanged
from revision 7499be4. Only their raw audit and paired bootstrap are reused; their
historical gate/manifest logic is not used for this replication.

Development seeds are 121110000/121210000/121310000, bootstrap 121510000;
confirmation seeds are 123110000/123210000. These match prerequisite seed audit
`333ed93a68ff026c940a5daee6152c9108c8ca721c3530df263d06489d978c39`.
Development requires more candidate wins on **each** opponent, including OwnInitial.
Confirmation requires candidate Wilson95 lower bound > .5 and cutoffs/64 <= .20
on each E/F opponent; matched baseline differences are descriptive. Every
fraction includes all 64 games; draws/cutoffs are not wins. No pooling, arm choice,
ranking or training-seed variance claim is made.

Receipt times establish initialization-before-evaluation ordering only. The
historical trainer has no absolute start/end timestamps in its report, so training
chronology requires separate supervisor evidence. Locally passing synthetic tests
prove validation behavior, not CUDA execution, source/runtime parity or strength.
Before packaging, integrate the verified initializer handoff and retain exact
adapter source hashes; new initializer bytes require newly generated receipts.
