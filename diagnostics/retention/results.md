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

The adjacent diagnostic scripts test checkpoint retention/transport separately. Tiny CPU tests validate the actual trainer save closure, immutable milestone behavior and corruption detection. The duplicate-final-save suspicion was rejected: existing66814cb already guards repeated saves. No trainer change. The synthetic recovery fixture is explicitly not real training, GPU resume equality or full-state semantic validation; it was not run at full size. A separately registered fresh640-game real training engineering run is in progress to assess two full-state recoveries; no outcome is claimed here.
