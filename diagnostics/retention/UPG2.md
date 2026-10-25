# Frozen UPG2 diagnostic contracts

These diagnostics extend the baseline 2048-game endpoint to the separately frozen
UPG2 fresh restart: trainer `2e67c2b9b7652ccbe75dbed249708356f38c3608`,
baseline objective, seed 52731415, 2048 games, 4096 optimizer updates, batch 256,
and 1,048,576 sampled training examples. Model/trainer source is unchanged.
Compressed recovery and objective helpers originate in diagnostic revision
`294e71accd5db6f72e2b57e8475aef202187dac8` and are included here so one diagnostic
checkout supplies the complete interface. The evaluator base is
`fbc8c5fb10640afe401b495483ca4368c94b377d`.

Use a diagnostic checkout separate from the immutable training source. Existing
ratio-one defaults remain. New explicit bindings:

* `evaluate_endpoint.py --expected-endpoint-games 2048 --expected-endpoint-updates 4096`
  additionally requires source/config/runtime and common initial-weight agreement
  with the archived ratio-one baseline, clean frozen source hashes, fresh-start
  arguments and the exact reported example count. Its other required source,
  training, paired-training, opponent, output, site and seed arguments are unchanged.
* `verify_compressed_states.py --expected-updates-per-game 2 --expected-loss-contract baseline`
  checks the complete default 512/1024/1536/2048-game chain at updates
  1024/2048/3072/4096, source/objective/arguments, counters, state fields, RNG,
  replay, optimizer, finiteness, initial/final weights and compressed/raw hashes.
  Decode remains sequential with an explicit scratch-space reserve.
* `resume_gate.py --entrypoint train_loss.py --updates-per-game 2 --loss-contract baseline`
  requires `--source`, `--site`, a new `--output`, and `--device cpu|cuda`.
  It validates the clean frozen source receipt and hashes before launch. Full four
  games versus two plus two resumed games must have exact model, optimizer, replay,
  RNG and counter equality after byte-exact compression/restore. All three actual
  saved states and reports must match the requested games, update ratio and batch8
  example count. Successful ratio2 output reports games4, updates8, examples64.

* `first_block_gate.py --mode candidate --allow-arg NAME` (repeatable) excludes
  NAME from the training-argument comparison only, via
  `validate_continuation(..., allowed_arguments=...)`. Accepted names are exactly
  `timeout_bootstrap` and `discount`; others and reference-mode use are rejected.
  Source, config, environment, runtime, provenance, games/ratio/seed/batch/loss
  contract, initial tensors and all array/aggregate parity checks stay strict, and
  the receipt records the allowlist with the tolerated values. Only arguments that
  provably leave collection unchanged before the first update can yield parity:
  `timeout_bootstrap` alters `targets` inside `collect` for cutoff games
  (train_loss.py:108–121) and the archived first block has 17 cutoffs, so a
  `continue` candidate is expected to fail array parity against the zero-bootstrap
  reference; `discount` changes search backups and remains bound by `config`.
  See first-block-gate.md. The endpoint evaluator (`catalog`) passes no allowlist.

The CUDA resume gate uses hidden448/context64/simulations32 and CUDA graphs/TF32,
but only parallel2, batch8 and an eight-step horizon. It is a bounded numerical
resume check, **not** a production first128-game parity check, throughput benchmark
or strength result. Production-runtime parity and paid admission remain separate
gates. CPU tests cannot establish CUDA equality.

Validation: run `PYTHONPATH=diagnostics/retention python3 -m unittest discover
-s diagnostics/retention -p 'test_*.py'`. Unit fixtures exercise correct ratios and
reject altered counts, objective, configuration, source, RNG/state, checkpoint
hashes and incomplete recovery. Real CPU split/resume tests additionally exercise
both ratio-one and ratio-two training with the frozen trainer.
