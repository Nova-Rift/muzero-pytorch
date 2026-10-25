# Frozen EGO/MIR restart validation

EGO (`--aim-decoding relative`) changes action semantics. MIR (`--mirror-augment`)
changes replay training samples. Both are fresh 2048-game experiments, not cross-source
continuations. A same-source split-resume correctness gate is still required for each
variant, with the actual flag on every full/partial/resumed trainer invocation.

`evaluate_endpoint.py --declared-difference aim_decoding --source-contract contract.json`
(or `mirror_augment`) requires a reviewed, frozen JSON contract before training:

```json
{"schema":1,"mode":"restart","candidate":{"commit":"<40 hex>","files":{"<all tracked files>":"<SHA256>"}},"paired":{"commit":"<40 hex>","files":{"<all tracked files>":"<SHA256>"}},"arguments":{"aim_decoding":{"candidate":"relative","paired":"absolute"}}}
```

The maps must exactly equal each run's complete source provenance. Candidate files
and commit must additionally match the clean source checkout or verified frozen
receipt. The contract is admission input, never inferred from the candidate after
training. Merely declaring a flag cannot authorize changed source. Frozen maps
are reviewed by the coordinator; this is not a signature/authentication format.
Legacy absent `aim_decoding` and `mirror_augment` mean `absolute` and `false`.
These are the only normalized defaults. Other undeclared argument differences,
configuration/environment/runtime changes, changed initial/checkpoint/opponent bytes,
malformed contracts and cross-source continuation are rejected. Combined EGO+MIR is
not admitted. Seed 52731415, batch 256, baseline loss, 2048 games and one update/game
are enforced for both sides.

`variant_collection_gate.py --variant ego|mir --source SOURCE --output OUTPUT
--baseline BASELINE --source-contract CONTRACT -- TRAINER_ARGUMENTS` wraps the
production trainer. Both variants require archived initial tensor and byte parity
before collection, plus source/argument/GPU binding. EGO collection equality is
inapplicable because action decoding changes. MIR requires equality of the two
archived trajectories and the archived first-block aggregate counters before its
first update; it does not claim all 128 trajectories were retained or compared.

Run `resume_gate.py --source SOURCE --site SITE --output OUTPUT --device cuda
--entrypoint train_loss.py --aim-decoding relative` and the separate equivalent
`--mirror-augment` gate before full training. CPU split-resume and mocked CUDA command
wiring tests are useful checks but do not establish CUDA graph/device-history parity.
