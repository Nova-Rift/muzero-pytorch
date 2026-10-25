# EGO/MIR recovery gate

Use `diagnostics/retention/verify_variant_recovery.py` from the recovery tools checkout,
with `--source` pointing to the separate immutable production model checkout at
`14fece1209067004f4c3936a6171bc40da55577d`. Do not replace production source with
this tools checkout: adding the verifier changes its source provenance.

The wrapper checks the exact session 33 production arguments, the externally
admitted restart source contract and paired baseline, then invokes the unchanged
`verify_compressed_states.verify` once. It checks the full 512/1024/1536/2048
chain, optimizer/replay/RNG/config/source fields, initial NPZ equality and final
NPZ equality, compressed/raw hashes and report final-checkpoint hash. It emits
an evidence manifest, source/contract hashes and checkpoint completion date from
the encoder's final `saved_at_unix` (remote host clock, not recovery file mtime).
No equality-to-resumed-CUDA or strength claim follows from this gate.

Recover before normal paid release:

- manifest.json, report.json, initial.npz, weights.npz,
  variant-collection-gate.json, compressed/manifest.json;
- compressed/games-N-updates-N.pt.gz for N=512,1024,1536,2048;
- preserve encoding-result.json, metrics.jsonl, all checkpoint NPZs and trace NPZs,
  trainer logs, source provenance, source contract and exact launch job;
- preserve CUDA split-resume results and all evaluation artifacts independently.

The first ten files are hashed in the verifier output. The broader archive manifest
must also include the additional provenance/log/evaluation files. Originals remain
intact. The existing decoder cleans only its unique temporary decoded files.

Run each arm sequentially, using the SAME lock path on the host (nonblocking flock
rejects a concurrent decode). Output should be a new filename in the private
recovery workspace; retain both failed and successful receipts.

```sh
python3 "$RECOVERY_TOOLS/diagnostics/retention/verify_variant_recovery.py" \
  --root "$RECOVERED_ARM" --arm ego \
  --source "$PAYLOAD/experiment/model" \
  --source-provenance "$PAYLOAD/experiment/model/.source-provenance.json" \
  --baseline "$PAYLOAD/experiment/baseline2048" \
  --contract "$PAYLOAD/experiment/contracts/ego.json" \
  --output "$RECOVERY_WORKSPACE/ego-recovery-attempt1.json" \
  --lock "$RECOVERY_WORKSPACE/host-decode.lock"
```

For MIR use `--arm mir`, its recovered root, contracts/mir.json and a distinct
output receipt. The gate's production paths refer to the original remote launch
arguments and must not be rewritten to local recovery paths.

Disk: peak temporary decoded bytes = max(raw_bytes in compressed manifest), not
sum of all four states. The wrapper rejects >1,250,000,000 raw bytes until storage
is reassessed; prior observed maximum was 1,015,285,607 bytes. The unchanged decoder
requires peak + 786,432,000 bytes free. Policy admission separately retains 4 GiB
free reserve and accounts for both lanes, compressed recovery, archive copies,
outstanding work and scratch. Do not count the reserve twice as decoded storage.
RAM: torch.load also materializes one full state, plus tensor finite-check scratch;
measure available RAM separately. Disk reserve is not a RAM bound.

Tests are small synthetic production-shape states, including missing/corrupt
compressed state, changed source/flags, wrong source pin, missing gate/date,
oversized decode and final weights hash. They are not real trained CUDA artifacts.
Existing session33 CPU split-resume fixtures remain independent logical evidence,
with pre-commit dirty provenance; they are not admitted production recovery.
