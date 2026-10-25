# Fresh-seed collection and recovery guard

This extends the conditional independent-seed package with executable adapters,
not an approved experiment. Production trainer commit `14fece1209067004f4c3936a6171bc40da55577d`
and the initializer/evaluator/readout bytes are unchanged. Seed is exactly73142017.
Original arm selection, published source, fresh seed collision check, exact-job
storage/budget/runtime/lease admission and real CUDA evidence remain prerequisites.

`replication_collection.py` imports the separately frozen trainer and wraps its
`collect` function. It creates the trainer arguments from the exact contract;
there is no arbitrary trainer-argument pass-through, resume, altered update ratio,
or short-training substitute. Both reference and training invocations declare
2048games,128parallel, ratio1 and the selected single intervention. A reference
raises `CollectionComplete` after one collection but before returning to the
trainer's update loop; the frozen trainer's `finally: pool.close()` still runs.
No final training report or trained checkpoint is expected for a reference.

Before either invocation, bind these immutable inputs by predeclared SHA256:

- `contract.json`, `initialization.json`, full source receipt, and the coordinator's
  initialization-review JSON. Actual CUDA prefix parity and the initializer source
  hash are checked. Never substitute the CPU diagnostic initial hash.
- Review JSON fields are exactly `schema:1`,
  `kind:"replication-initialization-review-v1"`, `accepted:true`,
  `contract_sha256`, `initialization_sha256`, `parity_sha256`, and timezone-bearing
  `completed_at`. Review must follow initialization and precede trainer start.
  This is an explicit coordinator review artifact, not something the adapter
  fabricates or an alternative to original-selection/admission guards.
- Candidate reference also requires `--paired-reference` and its receipt hash.
  Every training invocation requires `--own-reference` and its receipt hash.
  Candidate training additionally requires the same paired reference binding.

At the first actual trainer collection call the adapter validates the actual
manifest, initial NPZ bytes, full source/recipe/config/environment/runtime, GPU
name matching initialization, actual seed range and collector flags. This happens
before the original collector runs. A complete128-game block (all1408 simulator
slots, including inactive slots) and every non-timing aggregate are checked and
saved. Shapes/dtypes, finite arrays, replay indices and aggregate consistency are
checked; historical two-game trace evidence cannot pass.

MIR's collection is training-only augmentation, so candidate reference and
candidate training require bitwise full128 block and exact non-timing aggregate
equality with paired reference. EGO changes relative-aim commands: paired equality
is explicitly inapplicable, never asserted. Both variants' training invocations
must exactly replay their own same-side reference before their first optimizer
update. Later blocks use the untouched collector. Timing fields are preserved but
excluded from equality. Receipt records the review, actual start/completion times,
implementation and input artifact hashes, comparison scope and zero updates at
the check boundary; that zero is not a claim about the completed training run.

Use `--help` for the exact adapter arguments. The conditional graph names the
reference/train modes and exact implementation hashes but intentionally supplies
no executable supervisor commands. A selected contract and hash-bound runtime
inputs must exist before constructing those commands. Wrap the training command
with the unchanged `train_compressed.py`, preserving its existing deadline and
600MB encoded-state bound. Avoid output directories inside the frozen source.

`replication_recovery.py` binds each recovered side to the same reviewed contract,
first-collection receipt and externally pinned compressed manifest. It calls the
unchanged full-state verifier for exactly512/1024/1536/2048games/updates, baseline
loss, ratio1. The replication-specific manifest check pins seed73142017, the
full frozen recipe and exactly the selected side's intervention; each decoded
state must match those manifest arguments/source/environment/runtime. It verifies
encoded/raw hashes, optimizer/replay/RNG fields, counters, initial/final weights
and encoded storage bound. Relocated recovery directories are supported without
rewriting remote manifests. Preserve all receipts and source/encoded manifests.
This is integrity and configuration evidence, not GPU split-resume numerical
parity or playing strength. Run `replication_contract.py` on the completed pair as
a separate exact-pair check, before any evaluation.

Tests use synthetic CPU collector blocks and compressed synthetic states plus
source-bound CPU initializer-prefix parity. None invokes real training or CUDA.
The GPU-name comparison does not establish UUID-level device identity; the
separate exact-runtime gate and supervisor chronology remain required. Measure
full128 artifacts, serialization overhead, raw state peak and recovery scratch
before exact-job storage/runtime admission. Existing planning estimates remain
unapproved and are not measurements.
