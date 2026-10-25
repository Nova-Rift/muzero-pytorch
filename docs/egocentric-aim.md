# Egocentric (opponent-relative) aim decoding — `--aim-decoding relative`

Branch `egocentric-aim`, base c069f5ef (trainer `arena/` identical to the frozen 2e67c2b9 recipe).
Implements Proposal 2 of the iteration-29 cutoff analysis: trained candidates fire constantly but their aim
relative to the opponent stays at chance (0.17–0.26 within ±1 of 16 bins); training only sharpens an
absolute "fire west" habit that is correct from one spawn seat only (34 % vs 44 % seat win rate).

## What changes

The model still emits one of 153 indices (`move = a % 9`, `aim = a // 9`, aim 0 = no fire, 1..16 = 22.5° bins).
Only the interpretation of the aim bins at the point where the index becomes an arena command changes:

| mode | fire angle (arena coordinates, +x east, +y south) | bin 1 means |
|---|---|---|
| `absolute` (default, historical) | `(aim − 1) · 22.5°` | due east from every seat |
| `relative` | `bearing(self → nearest live opponent) + (aim − 1) · 22.5°` | straight at the opponent from every seat |

* **Bearing** = `atan2(opponent.y − self.y, opponent.x − self.x)` over fighters with `hp > 0` and `id ≠ self`,
  choosing the smallest Euclidean pixel distance (strict `<`, so ties keep the lowest fighter id). It is computed
  by the bridge from the world state (fighter `x`, `y`, `hp`) at every simulated frame (five 0.05 s frames per
  decision), so the aim tracks the opponent inside one decision while the chosen bin is fixed. With no live
  opponent the bearing is 0 (identical to absolute). `arena/aim_decoding.py` mirrors this from the observation
  vector: channels `4 + 4j .. 7 + 4j` hold `(dx/1000, dy/460, hp/3, alive)` for the ten other fighters in id order,
  so the bearing is `atan2(dy·460, dx·1000)` of the nearest alive entry. The bridge and the mirror agree on all
  16 bins from both seats (`tests/test_aim_decoding.py`).
* The move component, the environment, the observation, rewards, the search, the loss and the replay are unchanged.
* **Wire protocol.** `arena/bridge.mjs` accepts int32 actions `0..305`: `0..152` decode absolutely with the
  pre-change arithmetic (byte-identical commands, verified against the committed pre-change bridge), `153..305` are
  the same indices decoded relatively. Recorded `actions` in traces and replay remain model indices.

## Trainer

`arena/train_loss.py --aim-decoding {absolute,relative}` (default `absolute`). `collect()` adds the wire offset to
the commands sent to the pool; nothing else changes. The flag is part of `manifest.json → arguments`, so a relative
run's manifest differs from Baseline2048's only in `arguments.aim_decoding` (and `output`). Because the decoder does
not touch model construction or the seed, `initial.npz` is identical to an absolute run with the same seed/config
(tested at CPU scale), so the evaluator's paired-initialisation check against Baseline2048 still holds.

## Evaluator

* `diagnostics/retention/evaluate_endpoint.py::catalog` reads `arguments.aim_decoding` from the candidate's
  manifest and writes `aim_decoding: "relative"` into the candidate's catalog entry only (absolute/historical runs
  produce the historical catalog byte-for-byte; opponent entries are never modified).
* `tournament/run.py::play` applies each model's offset to its own actions, so an egocentric model plays E1024 /
  F1024 / OwnInitial with its decoder while the opponents keep the absolute decoding bit-identically.
  `controllers.py` is unchanged (same `controller_sha256`).
* `protocol.json` carries the model entries, so the binding is frozen with the bout. `source_sha256` (run.py)
  changes with this branch: for the strict paired readout both arms (Baseline2048 and the relative arm) must be
  evaluated with the same payload, and the readout's arm-config comparison must treat `aim_decoding` as a declared
  arm difference (readout.py lives outside this repository).

## Tests (CPU)

`ARENA_SITE=<canonical site checkout> python3 -B -m unittest discover -s tests -p 'test_*.py'` and
`cd diagnostics/retention && python3 -B -m unittest discover -s . -p 'test_*.py'`.
