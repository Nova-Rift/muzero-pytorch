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
