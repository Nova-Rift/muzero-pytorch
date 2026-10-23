# M2048 frozen target diagnostics — iteration17

Historical training source: d10af2d72dbf03dfec113c0b6e7e47481c5cc88d. Frozen full-state SHA256: 1898f6ad247339ecde16cdae5c4f1e120ec2615003850d1db9c9cd71cdbd8fc1. No model changes, new games or optimizer updates. This is local CPU diagnostic evidence, not strength evaluation or exact GPU continuation.

## Checks

`audit_replay_fit.py` takes explicit --source, --state, --catalog and --output paths. Fixed seed170001 samples128 examples with replacement from each of3 retained replay blocks. All three reconstructed losses match historical model.loss exactly (maximum error0). Components are recomputed diagnostics, not previously logged training metrics.

`audit_target_variance.py` takes the same paths, --states and --draws; optional --noise-fraction explicitly changes search settings. Historical model/trainer/history/search source hashes are checked against the full state. Fixed replay seed170002 and search seed170003. Root count8, simulations32, frozen M2048 weights. Two fixture tests verify stochastic-mean-fit and fixed-target-mismatch cases.

For a fixed history, mean KL(target||prior) = mean KL(target||mean target) + KL(mean target||prior). The first term is empirical Jensen-Shannon variation. Finite sampled means do not estimate irreducible noise exactly, and saved histories are not a held-out strength benchmark.

## Results

Initial4 histories x32 draws: variation2.491–2.717 nats. Expanded16 histories x64 draws: variation2.693–3.078, mean2.827; averaged-target/prior KL0.141–1.019, mean0.405. Variation accounts for74–95% of individual-target KL on these histories. Broad predicted policies versus sparse single targets alone do not establish underfitting.

A changed-setting zero-Dirichlet control on the same16 histories, retaining stochastic root sampling, yields mean variation2.793 and mean averaged-target/prior KL0.395, versus2.827/0.405 under historical noise fraction0.25. This small descriptive difference does not justify removing exploration. Shared seeds are reproducibility controls; different root choices change search paths, so this is not a variance-free causal estimate.

Independent retained replay census (branch experiment/iteration-17-replay-audit, commit577fb41c78e77ddeb51fe1cebd896121d1b8cc20) covers384 games/87694 valid player transitions. All valid targets sum to1 and have support8. Checked masks are consistent. No replay repair selected.

## Next experiment and stopping criteria

Before a training change, test target aggregation on fixed saved histories with independent discovery/validation search seeds. Compare prior against mean targets using disjoint draws; quantify how averaging changes variance and held-out target fit. Freeze history selection and settings before reading validation outputs. This tests target estimation, not game strength. Reject an aggregation proposal if benefit disappears with independent draws or its extra search cost lacks a bounded training benefit hypothesis.

A real training revision would be a new experiment with explicit warm-start/reset choice, not exact continuation. It still needs source tests, full-state remote recovery validation, sufficient local artifact space, costed throughput, frozen incumbent/E/F comparisons and independent-seed confirmation before promotion. Unchanged training is not justified by these integrity checks.

## Disjoint-draw follow-up

Fresh search seed170004, same16 histories,128 searches/history split64 discovery/64 validation. Fixed estimator =0.5 prior +0.5 discovery mean; no coefficient tuning. Mean validation cross-entropy change versus prior was−0.101142 nats, but improved only7/16 histories (range−0.473749 to+0.036655). Four fixtures now pass, including opposite validation targets detecting overfit. This is internal search-draw validation on development histories, NOT held-out game confirmation.

The64-search estimator costs far more than one historical32-simulation target. Mixed per-history gains and absent training/strength evidence do not justify paid training or changing exploration. Next work must cost-match a practical target estimator and freeze a broader history selection protocol before testing; do not tune on these validation draws. Full-state remote recovery and local storage capacity remain separate admission prerequisites.
