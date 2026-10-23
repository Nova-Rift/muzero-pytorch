# Equal-rollout target diagnostic

Historical M8 source d10af2d72dbf03dfec113c0b6e7e47481c5cc88d, frozen M2048 full state1898f6ad247339ecde16cdae5c4f1e120ec2615003850d1db9c9cd71cdbd8fc1. No model.py, physics, reward, weights or optimizer changes. Iteration18 was owner-interrupted; these completed diagnostic outputs were reviewed and published during iteration19 without relabeling that session.

Protocol: four saved histories from each of three retained blocks, eight target estimates/history. Standard: one32-simulation search with8 forced roots. Split: average four8-simulation searches with2 forced roots each. Both use32 recurrent expansions/estimate; tree depth, independent-root selection and initial-inference overhead differ. Independent32-draw standard-search mean is the reference, not optimal action value. Historical Dirichlet settings retained.

| Batch | History/search seeds | Standard mean squared distance | Split mean squared distance | Split lower/history |
|---|---|---:|---:|---:|
| Initial |180001/180002|0.326452|0.166237|8/12|
| Fixed-setting repeat |180101/180102|0.292781|0.164748|6/12|

Mean measured CPU seconds per eight estimates: initial0.130806 standard versus0.066884 split; repeat0.135466 versus0.073285. These tiny local timings are not GPU throughput estimates. No new games or training updates. Histories are sampled from the same checkpoint's retained replay and may overlap; this is not independent training-seed replication or held-out game confirmation.

The metric favors agreement with a stochastic historical teacher. Reduced variance can improve it while changing search bias or weakening actions. Results justify correctness tests of an isolated split-search candidate, not promotion or a stronger-play claim. Before paid training: measure candidate playing strength under frozen matched opponent/seat settings, choose and document warm start versus restart (changed algorithm is not exact continuation), test full-state retention/recovery, and cost the actual workload. Do not tune on the repeat batch.

Run `equal_rollout_targets.py --source HISTORICAL_SOURCE --state FULL_STATE --catalog CATALOG --output PRIVATE_JSON`; optional --history-seed/--search-seed reproduce the repeat. Three fixtures cover grouping, invalid probabilities and metric calculation. Source hashes are checked against the saved full state. Partial per-history output has complete=false; only completed batches have complete=true.
