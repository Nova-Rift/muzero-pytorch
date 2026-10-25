# Conditional independent-seed replication

This is preparation, not scientific selection or permission to train. Historical
EGO/MIR restart evaluation remains seed52731415. The separate
`diagnostics/retention/replication_contract.py` validates exactly seed73142017,
trainer14fece1209067004f4c3936a6171bc40da55577d and its complete tracked-file hash map.
It does not add a seed/source allowlist or change any running trainer/evaluator.
Run the validator from its own pinned diagnostics checkout; `--source` is the
separately staged, unchanged trainer14fece checkout.

A replication compares one fresh flag-off Baseline2048 and one fresh EGO or MIR
run, sharing seed73142017 and independently established initial weights. All other
scientific arguments, complete source files, runtime, environment,2048 games,
2048 updates and524288 training examples are fixed. There are two players with
the existing terminal reward1/0 rules, bound by the environment source hashes.
No warm start, seed61837209 reuse, combined intervention or self-pairing is allowed.
Both endpoints are required, including completed counters and initial/final hashes.
The CLI requires the predeclared exact contract and full source-receipt digests.
Templates with missing initialization hashes deliberately fail validation.

Before training, require audited original-seed development and held-out confirmation
passes. The coordinator chooses at most one passing arm without tuning the held-out
set. For replication development, predeclare64 games each against E1024,F1024 and
OwnInitial on starts121110000,121210000,121310000. Both fresh endpoints use the
same sets, balanced32/32 seats and the existing original-search sampling protocol.
Bootstrap seed121510000 is descriptive only. Require strictly more candidate wins
than matched baseline wins against EACH opponent; equal wins fail. Preserve all
cutoffs and use all64 games as the win denominator.

Only after that pass, run independent confirmation against E1024,F1024 on starts
123110000,123210000;64 games each, balanced32/32 seats. Candidate must have the
95% Wilson win-rate lower bound greater than0.5 and no more than20% timeouts against
EACH required opponent. Run matched-baseline confirmation bouts too for transparent
paired reporting. No pooling across opponents, optional stopping, seed replacement
following a loss, or treating timeout as a win. The earlier97010000/97020000
confirmation belongs to the original training seed and is not reused here.

The private seed audit found no collisions in2868 unique JSON documents and8459
hash-verified archived references, including conservative forward seed windows.
No missing/invalid JSON inputs remained; archive receipt path files point to records
already scanned. This is scoped evidence, not proof about unarchived/plain-text use.
The audit rejected earlier proposed evaluation ranges due historical training seeds;
that evidence remains private. Re-audit before admission.

Remaining gates are explicit: coordinator scientific approval; exact initial hash
from a no-update frozen-runtime initialization gate; per-arm CUDA split-resume and
production collection/reward parity; separate replication evaluator/catalog/readout
plumbing (the historical evaluator intentionally rejects this seed); complete source
and diagnostic revision publication; exact prepared-job storage/runtime admission.
This validator is not a GPU launcher, readout implementation or strength test.
