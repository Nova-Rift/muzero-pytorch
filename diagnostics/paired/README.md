# Frozen paired readouts

`readout.py` is the unchanged historical strict 64-game raw-record auditor.
`readout_3arm.py` preserves the historical multi-arm arithmetic and recipe rules;
only its library location is now relative for portability.
`readout_variants.py` admits exactly the EGO2048 and MIR2048 restart experiments
using the evaluator's schema-1 source contract, all source-file hashes, exact
variant values, final checkpoint and initialization hashes, completed endpoints,
recipe parity, preserved catalogs, frozen opponent entries, matched reruns and
unchanged raw-record checks. It never turns source or flag differences into blanket
permissions. Exit 2 means invalid/missing evidence; exit 0 means fully audited,
with the development verdict in `gate.passed` (false is a scientific rejection).

```
python diagnostics/paired/readout_variants.py \
  --arm EGO2048 --root RECOVERED_LOOP_A --training RECOVERED_LOOP_A/EGO2048 \
  --baseline-root RECOVERED_LOOP_A --baseline-training FROZEN_BASELINE2048 \
  --contract FROZEN_CONTRACTS/ego.json --source-receipt FROZEN_SOURCE_RECEIPT \
  --opponents FROZEN_EXPERIMENT_DIRECTORY --output readout-ego.json
python -m unittest discover -s diagnostics/paired -p 'test_*.py' -v
```

The opponent directory contains E1024.json, F1024.json and OwnInitial.json from
admission. MIR uses its corresponding contract, root and training directory.
Inputs must be frozen by admission, not rewritten to match observed results.
Development seeds are 95010000 / 95020000 / 95030000 and bootstrap 95050000.
The gate requires strictly more wins than matched Baseline2048 on BOTH E and F,
with at most 12/64 candidate cutoffs on each. OwnInitial is descriptive.
Paired intervals use the unchanged adjacent balanced-seat block bootstrap:
32 blocks, 10,000 draws, fresh Random(95050000), E then F, linear quantiles.

Neither a development pass nor a synthetic test establishes strength, promotion,
training-seed stability, or a ladder score. Held-out confirmation and independent
training replication need separately admitted protocols and jobs.
