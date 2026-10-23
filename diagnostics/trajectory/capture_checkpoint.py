"""Bounded POSIX command runner and immutable finalized-checkpoint capture.

The command must exclusively own its checkpoint, finish saving before exit and
not daemonize outside its process group. This is file integrity, not resume
correctness. No trainer imports or source changes are needed.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import stat
import subprocess
import tempfile
import time


class CaptureError(RuntimeError):
    pass


def _check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("checkpoint capture deadline exceeded")


def _signature(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _hash(stream, deadline, destination=None):
    digest = hashlib.sha256()
    while True:
        _check_deadline(deadline)
        chunk = stream.read(1024 * 1024)
        if not chunk:
            return digest.hexdigest()
        digest.update(chunk)
        if destination is not None:
            destination.write(chunk)


def capture(source, target, deadline):
    """Publish via atomic no-replace hard link; deadline is absolute monotonic."""
    source, target = Path(source), Path(target)
    if os.path.lexists(target):
        raise FileExistsError(target)
    temporary = None
    try:
        # Refuse symlinks, including a link swapped in between checking and open.
        fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as original:
            before = os.fstat(original.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise CaptureError("checkpoint is not a regular file")
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".capture-", delete=False) as copied:
                temporary = Path(copied.name)
                digest = _hash(original, deadline, copied)
                copied.flush()
                os.fsync(copied.fileno())
            original.seek(0)
            if _hash(original, deadline) != digest:
                raise CaptureError("checkpoint contents changed during capture")
            if (_signature(os.fstat(original.fileno())) != _signature(before)
                    or _signature(os.stat(source, follow_symlinks=False)) != _signature(before)):
                raise CaptureError("checkpoint changed or was replaced during capture")
            with temporary.open("rb") as copied:
                if _hash(copied, deadline) != digest:
                    raise CaptureError("copied checkpoint SHA256 mismatch")
            if (_signature(os.fstat(original.fileno())) != _signature(before)
                    or _signature(os.stat(source, follow_symlinks=False)) != _signature(before)):
                raise CaptureError("checkpoint changed before publication")
            _check_deadline(deadline)
            os.chmod(temporary, 0o444)
            os.link(temporary, target)  # Atomic publication, never replaces a target.
        return {"target": str(target), "sha256": digest, "bytes": before.st_size}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_and_capture(command, source, target, timeout_seconds, capture_seconds=120):
    """Command and copy each have explicit finite deadlines; stdout is inherited."""
    for value in (timeout_seconds, capture_seconds):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("timeouts must be positive and finite")
    if not command:
        raise ValueError("command is required")
    if os.path.lexists(target):
        raise FileExistsError(target)
    process = subprocess.Popen(command, start_new_session=True)
    try:
        code = process.wait(timeout=timeout_seconds)
        if code:
            raise subprocess.CalledProcessError(code, command)
    finally:
        # Also remove leftover descendants after normal exit, before reading state.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)
    return capture(source, target, time.monotonic() + capture_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--timeout-seconds", required=True, type=float)
    parser.add_argument("--capture-seconds", required=True, type=float)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    receipt = run_and_capture(command, args.source, args.target,
                              args.timeout_seconds, args.capture_seconds)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
