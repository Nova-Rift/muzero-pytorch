# Archived replay census

`replay_census.py --state STATE --sha256 EXPECTED --output NEW_JSON` verifies the input hash and uses CPU `weights_only=True`, memory-mapped loading. Policy reductions run in chunks (default 1,024 rows); it does not copy or save checkpoints. PyTorch with mmap support is required. Use a private ignored output directory. Run fixture checks with `python3 -m unittest discover -s diagnostics/trajectory -p test_replay_census.py -v`.

The diagnostic uses stored valid masks and indices, not all allocated lanes. A row is one player decision/transition, not one game. The historical arena has 11 allocated lanes per game, only two active players; `--lanes-per-game` must match the producer. Raw absorbing rows include never-active dummy lanes; active absorbing statistics explicitly exclude them. These are unique stored rows, not frequencies of training minibatch exposure (the trainer samples blocks uniformly and positions uniformly within each selected block).

Policy entropy uses natural logarithms and raw target masses, without silently renormalizing malformed data. Normalization, negatives, nonfinites, support and maximum mass are separately reported. Histogram output is bounded to 32 distinct values for scalar distributions. The census checks reward/mask consistency; it does not independently reconstruct the environment or prove targets are semantically correct.

Recency ranges are inferred from saved cumulative game totals and retained block sizes, assuming the historical collector's append-order deque. They are not stored timestamps or proof of arbitrary-producer chronology. An integrity census cannot establish why playing strength changes, justify an architecture change, or reproduce exact GPU resume.
