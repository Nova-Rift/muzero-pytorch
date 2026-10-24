# Fixed-root search depth development, October 23, 2026

Frozen M2048 weights (SHA256 9e8684f3f627b950465466a0ce0cae77275e6ced04b0f28fe11d198fef8aa118), eight forced root candidates, 32 versus64 simulations. Two model players; canonical rules/observations, winner1/otherwise0. Each condition has64 games with32 seats each,256-step cutoff, no evaluation root noise. E seed92120000 and F seed92130000, contiguous64 seeds per condition. All256 raw records were checked against summaries and balanced seats. This is an inference setting comparison, not additional training.

| Challenger | Opponent | Wins | Losses | Draws | Cutoffs | Batch seconds |
|---|---|---:|---:|---:|---:|---:|
| M2048-s32 | E1024 | 8 | 33 | 0 | 23 | 43.17 |
| M2048-s32 | F1024 | 28 | 33 | 0 | 3 | 42.92 |
| M2048-s64 | E1024 | 9 | 33 | 0 | 22 | 96.89 |
| M2048-s64 | F1024 | 35 | 21 | 0 | 8 | 100.74 |

The frozen gate requires95% Wilson lower win fraction>0.5 and at most20% cutoffs against EACH opponent, with no fewer wins than32sim baseline. The64sim candidate fails: E lower bound0.0758 and34.375% cutoffs; F lower bound0.4257. No confirmation, promotion or top-ten entry. The modest E change and larger F win count are development observations, not robust superiority. Increasing depth approximately doubled batch time; no GPU overlap was used.

Source: depth probe and diagnostics commit8880e833f9c735ab5fb4acdb42f2b9e3f027eb50. Actual runtime PyTorch2.8.0+cu128 on one H10080GB; synthetic64sim probe maximum policy/value error0 and weights unchanged. Source/config/protocol and every private raw record retained in the research archive. Prior split-search and wider-root failures remain separate evidence. No tuning against unused confirmation seeds.

## Retention checks

The adjacent diagnostic scripts test checkpoint retention/transport separately. Tiny CPU tests validate the actual trainer save closure, immutable milestone behavior and corruption detection. The duplicate-final-save suspicion was rejected: existing66814cb already guards repeated saves. No trainer change. The synthetic recovery fixture is explicitly not real training, GPU resume equality or full-state semantic validation; it was not run at full size. A separately registered fresh640-game real training engineering run completed; results follow.


## Fresh640 engineering training and frozen evaluation

Fresh seed52400000, random initial weights, empty optimizer/replay/RNG state; no warm start or continuation. Published retention trainer66814cbffc94996a51a7509a0755b76a3d28e215, unchanged M8 recipe (32sim/eight roots, hidden448, context64, replay3, batch256, unroll5, episode returns, zero timeout bootstrap, discount1,256 decisions, winner1/otherwise0). Completed640 games and640 updates in139.79 reported trainer seconds; last-update batch-mean loss is not a whole-block mean or strength evidence.

Full states at512 and640 were each approximately1.015GB. Both recovered locally, matched authenticated remote SHA256 manifests, loaded with weights_only=True, and passed source/environment/argument, optimizer/replay/RNG presence, finite-tensor and saved-weight alignment checks.17 files verified. Transfer241.13seconds. This verifies actual checkpoint recovery; GPU interrupted/uninterrupted equality remains untested. The synthetic fixture was not used to establish this gate.

Frozen final checkpoint SHA256171e979f9d78aa22d192cb5f4e0970009679cd25b990c728820b0bc2be92a568. Predetermined640 engineering endpoint, not best-of-many checkpoint selection. Three64-game, balanced-seat comparisons at32sim/eight roots, seeds92220000(E),92230000(F),92240000(own initial):

| Opponent | Wins | Losses | Draws | Cutoffs |
|---|---:|---:|---:|---:|
| E1024 | 30 | 33 | 0 | 1 |
| F1024 | 18 | 14 | 0 | 32 |
| FreshInitial | 32 | 30 | 0 | 2 |

No promotion. E Wilson lower win fraction0.3518; F50% cutoffs; own-initial32/30/2 does not establish robust improvement. This is one short training seed, not a completed2048-game replication or broad MuZero claim. These fresh evaluations use different seeds from the M2048 depth experiment and must not be presented as a matched head-to-head comparison. All raw records, states, source/config and checkpoint save timestamps remain privately archived. The pod was reused across depth evaluation, training/recovery and fresh-checkpoint evaluation, then released after recovery. Further long training requires sufficient storage for immutable checkpoints plus archive copies and a frozen costed plan.
