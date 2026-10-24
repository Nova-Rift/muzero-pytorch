# Frozen real replay gradient diagnostic

This CPU-only diagnostic imports the unchanged actual `MuZero.loss` and
`arena.train_selfplay.replay_batch`. It performs no optimizer step, target
regeneration, clipping, search, game, or checkpoint write. No trainer/model edits.
Compare original loss, an explicit baseline reconstruction, and policy/value
root:individual-recurrent ratio 5 weighting at K=5 with per-sample valid
coefficient mass preserved. Reward is unchanged and every hidden transition
retains gradient scale .5. Binary masks are required. Uniform batch mean follows
the actual arena trainer; it has no importance weights.

The CLI verifies compressed/raw SHA and byte count, safely loads with
`weights_only=True`, checks full saved source bindings against supplied source
manifest, and checks live model/replay source files. It streams to one owned raw
temporary file and deletes that file after load (also on ordinary exceptions).
It does not copy full checkpoints. Use a private output directory. Diagnostic
runtime is bounded by an alarm of 1..600 seconds; default 180. SIGKILL can leave
owned temporary output requiring inspection; it never deletes source evidence.

Select retained replay blocks in deterministic round-robin order. Within each,
call actual replay_batch with a dedicated seeded numpy Generator, recording
sample index positions, time/agent pairs, RNG before/after and hashes of all
returned tensors. Write frozen.json before any gradient call. This is stratified
descriptive sampling, not an unbiased uniform replay estimate; four batches over
three blocks oversamples block0. Output includes per-head/depth valid counts and
masked/valid mean losses; original and weighted gradient norms by parameter group,
root/recurrent norms and cosine (total and each group), and ratio to trainer clip
threshold100. Zero-norm cosine is null. Finite tolerance checks bind reconstructed
loss/gradients to original on each real batch. Float64 fixtures require exact
baseline parameter-gradient equality. Parameters and .grad remain unmodified.

Coordinator chose the initial descriptive measurement as four batches of64,
seed52600101, final model weights only, smaller than proposed4x256. Seed registry
and real-state run are coordinator responsibilities. Example (private paths supplied
by coordinator):

```
python3 diagnostics/real_gradients/probe.py --state STATE.pt.gz --manifest COMPRESSED_MANIFEST.json --source-manifest SOURCE.json --output NEW_PRIVATE_DIR --seed 52600101 --batch 64 --count 4 --weights model --max-seconds 180
python3 -m unittest discover -s diagnostics/real_gradients -v
```

Seven CPU fixtures cover actual loss/gradient reconstruction, independent mask
coefficient expectations, empty masks, unchanged reward-only gradients,
identity-chain .5 attenuation at every depth, and reproducible real replay
selection with unchanged fixture arrays. They do not validate trained-state
runtime, GPU parity, resume, clipping effects, learning, or playing strength.
Nonfinite loss/gradients and unknown parameter groups fail closed. Before decoding,
free space must exceed raw size plus750MiB.
Private results and `.evidence` must never be committed or published.

## Frozen replay probe, October24 UTC

Four64-example CPU batches from baseline2048 final replay, diagnostic seed52600101; round-robin over three retained blocks (block0 sampled twice). No optimizer step or parameter mutation. Original and reconstructed loss/parameter gradients matched exactly in all four batches.

| Batch | Original total norm | Root5 total norm | Original representation norm | Root5 representation norm |
|---|---:|---:|---:|---:|
| 0 | 0.861700 | 0.725877 | 0.103924 | 0.303244 |
| 1 | 1.518221 | 1.588251 | 0.321457 | 0.859286 |
| 2 | 0.654685 | 0.796664 | 0.164384 | 0.357074 |
| 3 | 0.807496 | 0.948693 | 0.130115 | 0.318512 |

Every measured total norm was below1.6 versus clip threshold100. Redistribution changed representation gradients without a clipping event in this probe. This is descriptive evidence from one checkpoint and small stratified samples; it does not establish harmful baseline gradients or improved playing strength. CPU Torch2.2 measurement is not CUDA numerical parity. Private tensors/checkpoints remain private.
