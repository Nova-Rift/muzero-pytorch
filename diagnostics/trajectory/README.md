# Frozen trajectory diagnostics (iterations 12–15)

This experiment branch preserves bounded recorder and saved-action replay diagnostics, including checks that rejected invalid analysis implementations. It does not change model weights, training settings, rewards or canonical physics.

`trajectory_recorder.py` records a bounded subset of actual Pool observations/actions. `diagnostic_evaluate.py` requires the frozen evaluator source hash and records the first eight seed-ordered trajectories. `diagnostic_evaluate_32.py` is the separately preregistered expanded capture used in iteration14; it records32. Both preserve all evaluator outcomes separately. These are diagnostic wrappers, not automatic ladder publishers.

`prepare_observer.py` reads an exact compatible canonical Site source and creates a new observer module at a caller-selected private path. It adds callbacks after shot emission, collision and boundary exit. It fails if the expected source statements differ. `replay-projectiles.mjs` consumes saved trajectories, reproduces the frozen five-frame cadence, checks every observation/outcome, and reports emissions, hits, exits and unresolved projectiles. The candidate index is its position in the two-entry evaluation catalog (0 or1); seats alternate per saved seed order.

```sh
python3 diagnostics/trajectory/prepare_observer.py SITE_CHECKOUT PRIVATE_OBSERVER.ts
node diagnostics/trajectory/replay-projectiles.mjs ABSOLUTE_SITE_CHECKOUT ABSOLUTE_PRIVATE_OBSERVER.ts 0 PRIVATE_TRACE.jsonl > PRIVATE_RESULT.json
python3 -m unittest discover -s diagnostics/trajectory -p 'test_recorder.py'
```

Use Node22.22.2 with TypeScript stripping support. The observer imports the exact caller-supplied simulator dependencies. Do not check generated observer modules, checkpoints or private traces into this repository. A compatible two-player trace is required; no scripted training or new inference occurs during replay.

## Verified checkpoint and limitations

Three recorder unit checks passed. The portable replay reproduced all48 saved iteration14 traces and5,273,664 observation scalars, maximum difference2.98e-8 (assertion threshold2e-6), with identical outcomes and exact shot-accounting totals. Ninety-six per-seat collision totals matched recovered HP damage. GPU recording parity separately passed two streams through256 decisions under PyTorch2.8.0+cu128 in iterations14 and15. These are engineering checks, not a stronger-model claim or universal parity proof.

Early local replay incorrectly used legacy cadence; observation parity rejected it at the first step. Counting after terminal frames was also rejected by shot accounting. Both were corrected before interpreting results. Those failed development checks remain documented privately; no fabricated intermediate Git commits are claimed.

For the selected40 J/E development traces, shot-weighted bearing error was90.54°/84.39° and near-bearing fraction6.15%/8.42%. This angle uses post-frame target position, not interception or future dodging. It did not distinguish J finished/cutoff behavior. In eight traced cutoffs, actual hits/emissions were J8/773 and E11/783; in32 finished games J60/1551 and E67/1569. These selected, outcome-conditioned, dependent observations do not establish causation or statistical superiority. Shot-weighted and equal-game summaries differ. Unresolved projectiles remain separate.

A frozen matched M1024/M2048-versus-E/F diagnostic is running separately to test whether the descriptive pattern applies to M. Do not transfer J-specific observations to M or describe diagnostic evaluation as training. Source provenance is in `source-provenance.json`; exact experiment records and raw artifacts remain in the private immutable archive.

`value_audit.py` performs frozen forward-only predictions on recovered candidate-first histories. It reuses exact DeviceHistory and excludes censored cutoff outcomes from terminal labels. `--self-test` checks history/action alignment. This exploratory checkpoint comparison is not a training intervention, held-out confirmation, or a causal diagnosis.
