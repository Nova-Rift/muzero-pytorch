# Experimental split-search adapter

`split_search.py` is an isolated inference adapter, not a training integration. It accepts an explicit historical search backend, shares read-only weights through a shallow module view and changes only that view's simulation configuration. Default four independent8-simulation searches, each with2 forced roots, average policy and value outputs; total32 recurrent expansions. Original model.py and historical checkpoint source remain unchanged. No arena adapter, GPU parity or playing-strength claim is established by this module.

Three fixtures verify stream grouping, caller configuration isolation, global RNG restore reproducibility and malformed controls. `check_split_search.py` binds model/trainer/history/search files to saved full-state hashes, loads M2048 frozen weights, and compares batched groups to separate stream calls with explicit noise/roots on two saved observations. Policy/value maximum error0; weights, caller configuration and explicit-control RNG unchanged. Observation hashes are retained privately. No new games or optimizer updates.

## Scientific status and next gate

Iteration18 equal-rollout teacher-distance diagnostics favored this estimator on mean but improved only8/12 initial and6/12 repeated histories. Both batches use the same frozen checkpoint/replay; their reference is a stochastic historical teacher. They are not independent training-seed replications or game-strength evidence. See equal-rollout-results.md and iteration17 target-audit-17.md.

Before GPU admission, integrate a separate evaluator hook and validate canonical observation/action/seat mapping and process reset against the existing checked arena runner. Keep two players and winner1/other0 unchanged. Pin exact model, source, config and opponent hashes. Preserve RNG streams across batching and record search controls. Record block/game/time/player identity plus observation hashes for further replay-based tests; hashes alone do not establish distinct histories.

Proposed bounded development protocol: fixed M2048 weights, compare original1x32/8 and split4x8/2 against the same frozen incumbent and E/F controls with balanced seats and matched fresh seeds. Freeze the complete protocol and selection rule before results; do not tune on confirmation. First establish no action-contract regressions and measure warmed randomized-order throughput/memory at intended batch/device. Equal recurrent expansions are not equal FLOPs or end-to-end runtime. If development is promising, fresh held-out confirmation and independent training-seed replication remain required for promotion; retain incumbent otherwise.

No training revision is admitted. If a future training experiment changes collection targets to split search, label it warm start from compatible M2048 weights with fresh optimizer/replay/RNG and compare against a matched original-search control with the same reset. Do not call this exact continuation. Full-state milestone retention and remote recovery validation remain prerequisites before long training. Historical M1024 missing optimizer/replay/RNG stays missing.

Reject the adapter if grouping/source isolation fails; reject a strength proposal if the fixed arena comparison fails its registered thresholds or its measured total cost cannot fit shared guards. No silent simplification of game, reward or checkpoint recovery to fit resources.

## No-noise controller follow-up

`split_controller.py` explicitly disables root noise on a configuration view and uses independent PCG64 per game/seat, matching the frozen evaluator's RNG ownership convention. Each stream draws four independent root subsets and one averaged-policy action; this consumes a different number of draws than historical search and is a changed algorithm, not trajectory parity with the baseline. Inactive streams consume no RNG. Six combined fixtures pass, including reset and batch-isolation checks.

The real historical-model check additionally compares grouped/separate controllers over three decisions with identical actions. A canonical arena smoke ran two two-player streams for four decisions each with valid actions and unchanged engine sources. It is truncated and produces zero scored matchups. This establishes basic observation/action wiring, not complete evaluator integration, GPU parity or playing strength. Existing tournament main still needs a separate explicit candidate dispatch and protocol metadata before scored games; never silently replace its controller or import results as the old ladder protocol.
