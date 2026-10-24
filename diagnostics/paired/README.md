# Frozen paired development readout

CPU-only, standard-library diagnostics for the predeclared four 64-game bouts:
baseline2048 and reference-Q2048, each against E1024 and F1024. This is one paired
training seed, not independent replication, promotion, or a ranking claim.

Before inspecting outcomes, the coordinator must freeze a private manifest with
`bouts`, a list of four objects containing `arm` (`baseline` or `reference-q`),
`opponent` (`E1024` or `F1024`), `directory`, and `protocol`. `protocol` must contain
the complete expected evaluator protocol JSON, bound to independently verified
final2048 checkpoint hashes, evaluator/controller/environment hashes and the frozen
catalog. Do not generate expected protocols from received result protocols as a
substitute for independent provenance verification. This parser checks supplied
bindings; it cannot prove checkpoint training provenance or that code was executed.
Runtime fields must also match: resolve legitimate runtime differences explicitly,
never silently relax contracts. Paths are private and are not copied into output.

Run `python3 diagnostics/paired/readout.py PRIVATE_MANIFEST --output PRIVATE_RESULT`.
Run tests with `python3 -m unittest discover -s diagnostics/paired -v`.

Exactly one raw `A__B-0.json`, `protocol.json`, and one-bout `summary.json` are
required per directory. Validate all 64 consecutive seeds, alternating seats
(32 each), outcome/step fields and recomputed summary counts. Both arms must match
seed/seat records and evaluator settings; opponent weights/configurations must
match. Historical seed ranges cannot pass the iteration23 four-bout analysis.
Development starts are E92320000 and F92330000; reserved confirmation is excluded.

Predeclared uncertainty: candidate-minus-baseline win indicator per matched game;
cutoffs and draws score zero, never dropped. Form 32 adjacent two-game blocks,
each containing one game in each seat (the two seeds are distinct). Resample 32
blocks with replacement, 10,000 draws per opponent, using Python `random.Random`
seed92360000, processing E then F through the same RNG stream. Report empirical
2.5/97.5 percentiles with linear interpolation and all block differences. This is
statistical resampling, not new game seeds. Intervals are conditional on these two
trained weight sets and fixed spawn geometry; they omit training-seed uncertainty.
Equal-opponent mean difference is secondary, without an aggregate interval.

Private `.evidence` may hold parser checks and fixtures; never publish historical
fixtures as iteration23 results. Synthetic unit fixtures are created in temporary
directories and have no research interpretation. No GPU/network/model changes.

New experiments may supply candidate_arm, development_seeds (E1024/F1024) and bootstrap_seed in the frozen input manifest. Defaults preserve the original Q experiment. Bind this manifest before reading outcomes; the parser checks identity, nonoverlapping64-game seed ranges and exact protocol agreement, not the chronology of that freeze. Candidate labels cannot contain paths. Twelve fixtures pass; original Q result recomputation is exactly unchanged.
