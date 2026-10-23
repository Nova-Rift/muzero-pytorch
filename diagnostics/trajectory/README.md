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


## Next planner diagnostic, registered but not executed

`prior_audit.py` extends the same frozen forward pass with entropy and top-eight prior mass. On32 recovered candidate histories, equal-game mean entropy was4.259 versus4.824nats for M1024/M2048; mean top-eight prior mass0.390 versus0.175. Top-eight mass is an upper bound for the probability mass of any8-action subset, not the actual sampled root mass. Values matched `value_audit.py` within1e-6. These are selected-state diagnostics, not a causal explanation of strength.

`test_root_candidates.py` checked repeated root8 CPU decisions, unique root32 candidates, finite search results, chosen actions within candidates and unchanged weights on two saved initial states. No new game was played. GPU correctness and strength remain pending. `next-planner-protocol.json` registers root8 versus root32 for the same frozen M2048 weights at fixed32simulations, against E/F with separate balanced catalogs. Different root counts consume random streams differently; matched labels do not imply identical RNG draws. This changes inference search configuration, not training duration or architecture. The candidate is not promoted.

Iteration15 four64-game development results (candidate wins/opponent wins/cutoffs; zero draws): M1024/E24/31/9; M2048/E13/37/14; M1024/F26/22/16; M2048/F24/31/9. This does not support unchanged M training extension. A paired-seed bootstrap for M2048-minus-M1024 win fraction was -0.172[-0.328,-0.016] against E and -0.031[-0.203,0.141] against F. Exploratory, one training seed, no confirmation or promotion claim.
