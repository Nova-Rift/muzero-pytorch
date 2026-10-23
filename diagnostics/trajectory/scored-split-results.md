# Frozen split-search development result

Evaluator source: 90bf0252afa334c797f1fb1beeb9f28f712cd3b7, branch experiment/iteration-20-scored-split. Parent adapter: b001b55ceced22aaaac2341a65a9a4ea0602c71e. No model weights were trained or changed.

The frozen comparison used M2048 original 1x32 simulations/8 forced roots versus four independent 8-simulation/2-root searches with averaged policies. Root noise was disabled. CPU, horizon 256, eight games per variant/opponent, two players, canonical observations/actions and winner-only reward were unchanged. E seeds 92021000–92021007; F seeds 92022000–92022007. Each variant used the same seed/seat schedule; seats were balanced across seeds, not mirrored for each seed. Candidate RNG consumption differs.

| Opponent | Variant | Wins | Opponent wins | Draws | Horizon cutoffs |
|---|---|---:|---:|---:|---:|
| E1024 | Original | 1 | 6 | 0 | 1 |
| E1024 | Split | 0 | 6 | 0 | 2 |
| F1024 | Original | 2 | 4 | 0 | 2 |
| F1024 | Split | 6 | 2 | 0 | 0 |

The preregistered rule required split wins at least baseline wins and split cutoffs at most baseline cutoffs against EACH opponent before considering larger fresh confirmation. E fails both conditions. This frozen inference candidate is not advanced. The F result does not override that rule. These are 32 development games from one frozen training checkpoint, not a promotion or general strength claim. They do not establish that a future split-target training revision would help.

Three scoring fixtures passed, covering all four outcomes, incomplete/cutoff handling and malformed records. Two truncated wiring games and four full-horizon gate games preceded this separate development matrix; they are not pooled with its results. Source and checkpoint hashes are bound in private protocols. All four development subprocesses completed successfully. A horizon cutoff is an observed game outcome, not an execution failure.

This is a new diagnostic protocol, not an import into the historical contender ladder. Per-job elapsed times under concurrent CPU load do not establish randomized-order throughput or GPU scaling. No GPU was allocated. Private raw records and exact manifests remain in the research archive; this public summary contains no private checkpoints or logs.

## Observation-only diagnostic replay

A separate wrapper (source3413c13b43f85ff783240a365e3b4b0ed0501c10) recorded all16 E baseline/split development trajectories. Every seed, seat, winner, finished flag and decision count exactly reproduced the original records. Canonical saved-action replay checked1,591,744 observation scalars with maximum error below2.99e-8 and matching outcomes. These repeated games add no independent strength evidence. Three existing recorder forwarding/bounds checks and three scorer fixtures passed.

The two split cutoffs were not failures to dispatch firing. On seed92021003 (challenger seat1), baseline won after48 decisions with3 hits from19 emitted shots; split reached256 decisions with2 hits from98 emissions. On92021004 (seat0), baseline lost after107 decisions with0 hits from40 emissions; split reached256 with1 hit from98. Counterexample92021000 changed from baseline cutoff (2 hits/98 emissions) to split loss (1 hit/36 emissions,97 decisions). Counts are observed collisions/emissions, not independent Bernoulli trials. Unresolved projectiles are retained separately in private records.

This outcome-selected description cannot separate aim, movement, evasion, search depth or changed random draws as causal mechanisms. It does not justify changing physics/reward or immediate split-target training. The frozen rejection remains. Any new training proposal needs a separately tested hypothesis and matched control; a changed-target warm start must reset incompatible state explicitly and must not be called exact continuation.
