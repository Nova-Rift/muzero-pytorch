# EGO2048 →4096 unchanged full-state continuation

This is an external orchestration package. Train only from the original clean
`14fece1209067004f4c3936a6171bc40da55577d` frozen tree, retaining its source
receipt. Never train from this diagnostic revision. `model.py`, physics, reward,
trainer, loss, replay configuration, original initialization and seed are unchanged.

`package.py --root RESEARCH_ROOT --workspace EXCLUSIVE_WORKSPACE` builds an inert
12-job plan plus protocol, hash-bound payload recipe, opponent catalogs, storage
estimate and runtime/cost estimate. It does not allocate, stage checkpoints, run
training, write shared controls or grant admission. Prepared outputs require a
fresh directory. `materialize.py --recipe ... --verify-only` checks every payload
input. After admission, the coordinator may supply `--destination` and optionally
`--archive` to build the hash-verified payload. Keep new diagnostics outside the
frozen model tree.

The chain is: exact H100/Python/PyTorch runtime → relative-aim CUDA split-resume
comparison → recovered2048 checkpoint identity/Adam/replay/RNG/initial-state gate
→ unchanged2048 additional optimizer updates → all four new milestone recovery
checks → seven matched development comparisons. Every job depends on its
predecessor; failure blocks all descendants. The actual input state is restored
on the GPU host and retained; no weights-only fallback exists.

EGO4096 and frozen EGO2048 each face E1024, F1024 and their own shared relative-aim
initial policy on fresh matched64-game balanced-seat seeds; EGO4096 also faces
EGO2048 directly on separate seeds. Frozen old baseline OwnInitial was absolute;
this new within-EGO comparator explicitly uses relative decoding. Original
development evidence motivates only longer unchanged training. Existing heldout
outcomes are not a selection input. New heldout seeds are reserved separately but
not scheduled or admitted. Development results cannot authorize publication.

Before submission the coordinator must verify the returned handoff and source
revision, publish scoped source history, refresh collision audit/storage/archive
quota, bind live runner/session/deadlines, obtain the current price/budget approval,
and cover automatic lease renewal. Both initial and post-training real CUDA gates
remain unexecuted by preparation. No paid work is authorized by these files.
The source tests that consume private frozen inputs skip when those inputs are
absent; materializer tests are self-contained. Mock resume tests are not CUDA proof.
