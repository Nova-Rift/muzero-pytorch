# Bottleneck index

This is a teaching index, not a list of established universal fixes.
Full attempt history: [self-play investigation](self-play-investigation.md).
Hardware/throughput history: [training performance](training-performance.md).

| Symptom | Evidence | Status / next test | Reusable lesson |
|---|---|---|---|
| Repetitive action, all games timeout | Frozen original policy gives action136 about99.9992%; noisy8-search has only1visited action | Policy collapse confirmed at inspected states; original trigger unresolved | Log policy entropy and root action coverage, not just loss |
| Loss approaches zero while behavior worsens | Original loss~4.8→.14 at timeout collapse | Observed; not an optimizer-success metric | A policy can fit its own poor search targets perfectly |
| Training behavior differs from website |66px vs55px per decision; floating remainder adds frame | Confirmed, explicit5-frame option plus regression | Test simulator integration against actual product loop |
| Self-play finishes but tournament stalls | A/B~94%natural training ends; noise-free evaluation alltimeouts | Confirmed for development runs, explanation under test | Match exploration/evaluation protocols and test frozen opponents |
| Probability-floor model looks better only in some evaluations | Bpolicy30/32wins vsA0/32, one seed, legacy cadence | Preliminary, opponent mode confounded; cross-mode tests registered | Hold opponent fixed before assigning causal credit |
| Search cost rises steeply | New32search pilot firstbatch21.6s search vs8search pilots~1.9s | Not a matched speed benchmark; work and capture differ | Measure tree traversal, expansion and transfer separately; more searches can have superlinear implementation cost |
| More episodes mistaken for better learning | Original10kgames/10kupdates metspeedtarget but failedplay | Confirmed reporting distinction | Report interactions, updates, replay sequences, cutoffs and held-out wins separately |

No scripted policies, demonstrations, hand-written aim rules or tactical rewards
are used in this investigation. No superhuman result is established.
