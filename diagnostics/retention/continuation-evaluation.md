# Evaluating an exact continuation endpoint

`evaluate_endpoint.py` retains the original 2048 endpoint default. Explicit
`--expected-endpoint-games 4096` admits only a completed 4096-game/4096-update
endpoint, with final checkpoint and paired original initialization hashes checked.
The paired training directory must be the original 2048 run. Its manifest binds
config, environment, Python/Torch, clean source commit and every source file hash.
Scientific arguments must match; only site/output/resume-state/games/max-seconds
may differ, matching the frozen trainer's resume allowlist. A resume state must
be present in the continuation manifest. These are evaluation admission checks,
not independent proof of full-state resume; retain the separate resume-state gate.

Stage **only this evaluator script** at a separate diagnostics location. Keep the
training tree at its original commit (for the prepared baseline continuation,
`2e67c2b9b7652ccbe75dbed249708356f38c3608`) and leave its provenance receipt intact.
Do not train from this evaluator revision or overwrite its copy in the frozen
training tree. Invoke the separately staged script with `--source` pointing to
that original tree; it verifies its hashes and launches that tree's unchanged
`tournament/run.py`. Record the evaluator commit/hash separately in the job plan.

All original evaluation settings remain: 64 games, balanced tournament runner,
parallel64, workers4, limit256, CUDA, explicit frozen seed. This parameter changes
endpoint admission only; it does not alter the game, model, trainer, opponents,
selection protocol, or authorize publication. A wrong endpoint remains rejected.
