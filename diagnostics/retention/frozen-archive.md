# Bounded frozen-artifact archival

`archive_frozen.py` uses the existing private research-history schema1 and its
`objects/<sha256>` and `records/<id>-<revision>.json` layout. It creates no second
archive or hidden retention policy. Pass the existing archive root explicitly.

The coordinator supplies an explicit manifest after verifying closed, exclusively
owned outputs. `frozen_closed_outputs: true` is an authorization assertion, **not**
proof that no writer exists. Do not use for active training files, source worktrees,
original historical baseline models, mutable latest files, or credentials.
Executable artifacts are refused to avoid silently losing executable mode.
No directories are traversed; no old archive records are scanned. Paths remain
present, bytes remain exact, resulting artifacts share a read-only archive inode.

```json
{"frozen_closed_outputs":true,"files":[
  {"path":"/absolute/frozen/checkpoint.pt.gz","sha256":"64 lowercase hex characters","bytes":123}
]}
```

Run the exact tested diagnostic from a pinned source checkout:

```
python3 diagnostics/retention/archive_frozen.py --id candidate-cycle \
  --manifest /private/frozen-files.json --archive /private/research-history
```

Expected hashes/sizes must come from verified checkpoint recovery or immutable
archive records. No hash is invented from a filename. For each new object, stream
and hash one source into an fsynced temporary object, check expected hash and stable
source metadata, publish without overwrite, then atomically replace the original
path with a hardlink. The next file starts after that replacement. Previously
archived objects are reverified without copying their payload. Hash caching exists
only within this invocation and is bound to device/inode/size/mtime/ctime.

Extra payload disk for new single-link frozen files is bounded by the largest file,
not the whole candidate. New multiply-linked files are refused because their old
inodes could retain extra copies outside the explicit manifest. Existing objects
can still accept duplicate paths without increasing payload storage. An open
obsolete source descriptor cannot modify the newly copied archive inode. However,
exclusive writer ownership remains required: a later writer reopening the path
would violate the frozen-output contract.

Records retain original mtime and mode, original filenames, exact hashes and sizes.
Old records and objects are never overwritten. If a batch fails partway through,
previously archived paths/bytes remain intact and objects remain recoverable;
there may be no final batch record. Re-run the same admitted manifest after fixing
the cause. Do not call a failed batch complete. Preserve the manifest itself.

## Recovery integration and admission

Call on each fully recovered/verified closed checkpoint batch, not while transfer
is still writing. Keep dynamic metrics/manifests outside the frozen batch until
closed. Use the existing normal snapshot for remaining metadata/executables. The
returned record is directly readable by `history.py list`; include it and the
input manifest in the session evidence. This opt-in diagnostic has not changed
worker/runner source or existing `history.py` behavior.

The tool reduces duplicate archive copying and repeat hashing. It does **not**
remove compressed output storage, raw decompression scratch, transfer staging,
reserve headroom or final full-state verification. Do not lower a current admission
bound merely because this tool passed tests. Measure the exact planned recovery
sequence and all simultaneous storage obligations first. It does not allocate,
renew, release or admit GPU work.
