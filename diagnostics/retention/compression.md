# Lossless checkpoint transport experiment

The iteration22 fresh640 full-state checkpoint is 1,014,720,295 bytes.
Gzip level1 produced 98,522,357 bytes (9.71% of original) in 5.707 seconds on the
local recovery host. Compression plus streamed decode/hash verification took
7.994 seconds. A separate atomic restore plus rehash took 9.401 seconds and
reproduced the exact original SHA256:
`2ed0065bc207c9b483efac2f9da27b42856f1707c72417855d1088254b34b4fc`.

This is an actual full-state measurement, not a synthetic payload estimate.
The original and original immutable archive remain preserved. No state fields,
optimizer, replay, RNG, tensor precision or serialization bytes were changed.
This does not establish numerical split-resume equivalence or playing strength.
Compression throughput on a future remote host and the size of future checkpoints
must still be measured; this ratio is not a universal bound.

`compressed_checkpoint.compress(source, destination, expected_sha256)` refuses
an existing destination and validates the source and decoded content before atomic
publication. `verify(encoded, expected_sha256, expected_bytes)` streams a bounded
decode without loading tensors. `restore(encoded, destination, expected_sha256,
expected_bytes)` writes a temporary raw file and publishes it only after exact
verification. All operations require a trusted manifest digest; never infer the
expected digest from an untrusted download. Safe tensor loading remains a separate
existing verification step after recovery.

For future recovery, preserve the compressed object and a manifest containing
original byte count/hash, compressed byte count/hash, exact source/config/runtime
bindings, and checkpoint save evidence. Archive both compressed object and manifest.
Restore one raw checkpoint at a time for verification or resume. Include temporary
raw restore space and archive copying in capacity estimates; do not count compressed
storage alone. Do not delete original historical checkpoints to adopt this format.

Three focused tests cover exact roundtrip/restore, immutable destinations, rejected
source hashes, truncated/corrupt gzip, decoded size bounds and mismatched decoded
hashes. Model and trainer behavior are unchanged.
