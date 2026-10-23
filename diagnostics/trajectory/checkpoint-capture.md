# Finalized checkpoint capture

`capture_checkpoint.py` runs a command without a shell in a fresh POSIX process group. The command has a required finite timeout; remaining group members are killed on success, failure or timeout, with a five-second bounded reap. Only a successful command permits capture. Use a fresh trainer output directory: successful no-op commands can capture an old source file, so this wrapper does not prove checkpoint freshness.

Example (paths and command supplied by the caller):

```sh
python3 diagnostics/trajectory/capture_checkpoint.py --source OUTPUT/resume-state.pt --target MILESTONES/split-2176.pt --timeout-seconds 180 --capture-seconds 120 -- python3 TRAINER.py ARGS
```

The target parent must exist. Capture streams into a temporary sibling, hashes and fsyncs it, rehashes the source, compares source inode/size/mtime/ctime, verifies copied SHA256, and publishes using an atomic no-replace hard link with mode 0444. Existing targets, even dangling symlinks, are rejected. A concurrently created target is preserved. Temporary files are removed on failure. Successful stdout includes target, byte count and SHA256 (command stdout is inherited).

The command must exclusively own its source, finish saving before exit, and not escape its process group by daemonizing. Metadata/content rechecks detect observed mutation, but cannot prevent an unrelated writer racing after the final check. Read-only mode prevents ordinary accidental modification, not deliberate chmod by the owner. These guarantees require local POSIX filesystem semantics. Copy deadline checks occur between file operations; a kernel/filesystem stall cannot be preempted by this Python loop. Parent supervisor/watchdog deadlines remain required. No directory durability across power loss is claimed.

Eight CPU tests cover success, failed command, descendant timeout cleanup, existing and racing destinations, source mutation, timeout validation, expired capture and symlink rejection. They do not validate checkpoint deserialization, full-size throughput, GPU exact resume or training strength. Trainer source is unchanged.
