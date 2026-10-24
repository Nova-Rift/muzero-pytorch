# Paired Q-completion experiment — October 24, 2026 UTC

Two fresh MuZero arms trained for 2,048 games and 2,048 optimizer updates each from the same initialization seed 52500000. The candidate changes only training-search unvisited-Q normalization: unvisited actions receive zero after normalization. The baseline retains the original search. Both use the same original search during evaluation. This isolates training-search effects; it is not an evaluation of revised inference search.

Source: Nova-Rift/muzero-pytorch commit c1971134f6edf302af1e41f892fca13721452d9b. Original model.py, training loss, game physics and winner-only reward are unchanged. This is a fresh restart, not continuation. Hidden448, context64, simulations32, root candidates8, unroll5, batch256, three replay blocks, discount1, episode returns, zero cutoff bootstrap, 256-decision limit, two players, 128 parallel games, four environment workers, CUDA graphs and TF32. Both actual CUDA split/save/compress/restore resume fixtures passed; all eight full-state milestones (512/1024/1536/2048 per arm) were recovered and integrity-verified.

## Frozen final-endpoint results

Each row is 64 games, balanced 32 per seat. No intermediate endpoint selection. Draws and timeouts score zero. Common original planner, no root noise, matched source/runtime/environment and fixed spawn geometry. Seeds vary policy sampling, not worlds.

| Arm | Opponent | Wins | Losses | Draws | Timeouts |
|---|---|---:|---:|---:|---:|
| Baseline2048 | E1024 | 9 | 26 | 0 | 29 |
| Q2048 | E1024 | 9 | 12 | 0 | 43 |
| Baseline2048 | F1024 | 25 | 28 | 0 | 11 |
| Q2048 | F1024 | 35 | 18 | 1 | 10 |
| Baseline2048 | Shared initial | 17 | 24 | 0 | 23 |
| Q2048 | Shared initial | 39 | 8 | 0 | 17 |

E/F development seeds are 92320000/92330000. Own-initial seed92370000 was fixed before outcome inspection. Reserved confirmation seeds92340000/92350000 remain unused. Initial-control results do not change the development gate.

Candidate-minus-baseline win fraction is 0.000 against E (paired block-bootstrap95% interval [-0.125,0.125]) and +0.15625 against F ([0.000,0.328125]). The bootstrap uses 32 adjacent balanced-seat two-game blocks, 10,000 draws, seed92360000. These intervals are conditional on one paired training seed and fixed geometry; they do not estimate training-seed uncertainty. Equal-opponent secondary difference is +0.078125.

The frozen replication gate required a strict win-fraction improvement separately against both E and F, and <=20% candidate timeouts on each. It failed on E: equal wins and 43/64 timeouts. Conditional independent-seed replication was therefore not admitted. No model was promoted, no top-ten ranking or incumbent changed, and no confirmation games were run. Q's better own-initial and F scores are mixed evidence, not general superiority. Declining Q training loss (final batch mean1.9903 versus baseline5.2336) did not establish success on the common-opponent gate.

## Exact identities and preservation

- Shared initialization SHA256: a15c7cc9535e469f8c3b0ebfcc0aab4c3cb6f9333e9d51afc8480fa1288c6723
- Baseline2048: 0f421217adb43c7e5fb1f908fb671e116bff540f1f1d294733b060704e784efa
- Q2048: 41327b37a5796dfe59d4667bd738baf58f81778d35d7966d79fb112948835f97
- E1024: 18c18151b820636ef3bf67ea52dbecd3cb47582d9ea6bf63ede36d6c2adb1532
- F1024: 4de08ce08adef8344d6cf8707e22a74896c355008ffff18d3b10ce21ec1dcdb7

Readout source b496ca6bf70d63b1fffe34c0cd043c0809050649 verifies exact frozen protocols, 64 unique seeds, seat balance and raw-summary agreement. Recovery verifier478bf5bc67fd528031fcaa31981600d128191d43 validates compressed/raw hashes and optimizer/replay/RNG fields. Entrypoint identity is bound by the external immutable job manifest; model state/config alone cannot distinguish the training-search variant.

Both candidates, every full-state milestone, raw per-game records, source/config manifests, loss samples and rejected followup plan remain in the private immutable archive. No private checkpoints or logs are uploaded with this summary. The same H100 pod ran both arms and all six bouts serially without a cycle teardown; it was released after complete recovery and the failed followup gate. Provider deletion was verified. Next work is a separate real-replay gradient diagnostic of loss weighting, retaining original search; no strength benefit is assumed.
