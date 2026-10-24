# Bounded serial-versus-overlap engineering check

`diagnostics/retention/overlap_benchmark.py` runs the same two128-game trainer
commands serially and concurrently in four isolated scratch directories. One global
240-second allowance includes startup, checkpoint writes and all children. Each
child owns a new process group; timeout, error and termination clean those groups
only. The coordinator admits this as one bounded GPU supervisor test after productive
work, with270seconds outer timeout including30seconds cleanup, never as two already
approved productionGPU slots.

The JSON plan must bind every source/runtime file hash, exact source commits,
manifest arguments/config, Torch/GPU identity and the intended supervisor-plan
fingerprint. Outputs record actual wallclock times, sampled device memory, complete
counters and checkpoint hashes. Serial/concurrent checkpoints must be identical.
A provisional short-workload eligibility result requires10% end-to-end speedup,
15% device-memory headroom and at least8GiB free. Missing samples, incomplete work,
child failures and hash/config mismatch are inconclusive, never permission.

This cold-start128-game sample does not cover three-block replay occupancy or long
steady-state allocator behavior. `full_plan_authorized` is always false. It must not
be copied directly to a2048-game supervisor concurrency proof. A useful speedup can
justify a later representative peak-memory/steady-state check under the same source,
config/runtime and full command-plan binding. The runner does not modify supervisor
slots, reserve compute or upload anything. No measured result is included with code.

Four private trainer output directories retain manifests and reports. Launcher logs
and measurement JSON are sibling files so trainers can exclusively create their own
output directories; failures before directory creation still preserve evidence. Avoid archiving raw full-state `.pt` scratch files;
coordinator decides evidence retention/cleanup after recovery.

The first actual bounded attempt failed before training with `FileExistsError`: the
launcher created a directory that the trainer creates with `exist_ok=False`. That
attempt produced no throughput evidence and remains inconclusive. The corrected
launcher is covered by a CPU subprocess integration test exercising the same
exclusive-directory contract and complete serial/concurrent report inspection.
This correction does not authorize a GPU retry or imply measured concurrency.
