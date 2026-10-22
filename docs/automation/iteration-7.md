# Iteration 7: finite-horizon target candidate M — failed before GPU training

M tests zero value bootstrap at the finite 256-action training cutoff. This is a target-recipe experiment, not an architecture change or exact continuation of L. Restart with the same initial training seed; preserve original model.py, reward, physics, and two players. L's unsuccessful duration extension and positive predictions on surviving zero-reward cutoffs motivated this hypothesis. No strength conclusion can be drawn from this attempt.

## Engineering checks

32 model tests passed, two CUDA-only checks skipped. CPU uninterrupted versus resumed weights matched exactly, repeat evaluation records matched, and first-collection observations/actions/rewards stayed identical while timeout targets changed as intended. Five lifecycle tests verify success retains the pod and errors/stops/expiry clean it. The prelaunch control suite passed 97 tests. These did not cover the final packaged provenance receipt, which was my launch-preparation error.

## Actual paid outcome

One H100 80GB pod in US-GA-2 at $3.99/hour. Setup and transfer took 154.158 seconds and verified 352 staged files. Training then rejected the inherited L source receipt with `ValueError: Cached source changed: arena/train_selfplay.py`. The outer transfer manifest was current, but the trainer's internal source manifest was stale. Zero GPU training games/updates and zero evaluation games completed; no trained M checkpoint exists from this allocation. No metrics were fabricated or promoted from CPU checks.

Failure recovery saved the remote log and exit record, paused automation, deleted the pod and verified absence. Allocation-to-deletion time 183.658 seconds; GPU estimate $0.2035545, excluding separately priced ephemeral disk and unreconciled invoice details. The full $3 reservation remains consumed; $21 of today's $30 local allowance remains. No retry or renewal occurred.

## Local repair after cleanup

A new source-staging helper builds a fresh snapshot only from a clean committed candidate and generates the trainer's provenance receipt from those exact bytes. It rejects overwrites and uncommitted changes. A regression test covers changed commits and preservation of the old snapshot. The original failed payload remains unchanged and still reproduces the manifest rejection. A separate corrected staged snapshot validates all 228 source files at candidate commit a1038ad52553ca85a784cd73c8f8ec16543d2b35 and completes a two-game CPU engineering smoke, including checkpoint writing. This is not a GPU result or evidence of stronger play.

The lifecycle helper retains resources on successful cycle completion and cleans failures; the enclosing worker retains final-session cleanup. Safe remote deadline extension has not been exercised and is not claimed validated. Do not renew until the remote watchdog update is implemented and verified for a concrete next task.

## Next action

Keep this attempt failed and paused. A future owner-reviewed continuation must use a new assigned iteration/reservation and fresh directory, regenerate both source and transfer manifests, run the actual staged entrypoint locally before allocation, refresh offers and authenticated resource inventory, and pass all launch guards. Do not reuse iteration 7's reservation or silently retry. Retain all scientific and operational evidence, including this failed payload and the separate corrected CPU smoke. The pre-registered M scientific question remains untested on GPU.
