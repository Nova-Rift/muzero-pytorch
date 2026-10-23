import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import capture_checkpoint as capture


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "resume-state.pt"
        self.target = self.root / "milestone.pt"

    def run_code(self, code, timeout=3):
        return capture.run_and_capture([sys.executable, "-c", code],
                                       self.source, self.target, timeout, 3)

    def test_success(self):
        receipt = self.run_code(f"from pathlib import Path; Path({str(self.source)!r}).write_bytes(b'final checkpoint')")
        self.assertEqual(self.target.read_bytes(), b"final checkpoint")
        self.assertEqual(receipt["sha256"], hashlib.sha256(self.target.read_bytes()).hexdigest())
        self.assertEqual(self.target.stat().st_mode & 0o777, 0o444)
        self.assertEqual(list(self.root.glob(".capture-*")), [])

    def test_failure_never_captures_even_existing_source(self):
        self.source.write_bytes(b"old")
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_code("raise SystemExit(7)")
        self.assertFalse(self.target.exists())

    def test_overwrite_rejected_before_command(self):
        self.target.write_bytes(b"preserve")
        with self.assertRaises(FileExistsError):
            self.run_code(f"open({str(self.source)!r}, 'w').write('ran')")
        self.assertFalse(self.source.exists())
        self.assertEqual(self.target.read_bytes(), b"preserve")

    def test_timeout_kills_descendant(self):
        pidfile = self.root / "child.pid"
        code = ("import subprocess,sys,time; "
                "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
                f"open({str(pidfile)!r},'w').write(str(child.pid)); time.sleep(60)")
        with self.assertRaises(subprocess.TimeoutExpired):
            self.run_code(code, .3)
        self.assertFalse(self.target.exists())
        pid = int(pidfile.read_text())
        # Linux may retain a killed orphan as zombie until its reaper collects it.
        for _ in range(100):
            procstat = Path(f"/proc/{pid}/stat")
            if not procstat.exists() or procstat.read_text().split()[2] == "Z":
                break
            time.sleep(.01)
        else:
            self.fail("descendant still running after timeout")

    def test_source_mutation_and_temp_cleanup(self):
        self.source.write_bytes(b"original")
        original_hash = capture._hash
        calls = 0
        def mutate(*args, **kwargs):
            nonlocal calls
            result = original_hash(*args, **kwargs)
            calls += 1
            if calls == 1:
                self.source.write_bytes(b"modified")
            return result
        with patch.object(capture, "_hash", side_effect=mutate):
            with self.assertRaises(capture.CaptureError):
                capture.capture(self.source, self.target, time.monotonic() + 3)
        self.assertFalse(self.target.exists())
        self.assertEqual(list(self.root.glob(".capture-*")), [])

    def test_racing_destination_not_replaced(self):
        self.source.write_bytes(b"source")
        real_link = os.link
        def race(src, dest):
            self.target.write_bytes(b"other writer")
            real_link(src, dest)
        with patch.object(capture.os, "link", side_effect=race):
            with self.assertRaises(FileExistsError):
                capture.capture(self.source, self.target, time.monotonic() + 3)
        self.assertEqual(self.target.read_bytes(), b"other writer")
        self.assertEqual(list(self.root.glob(".capture-*")), [])

    def test_finite_timeout_required(self):
        for timeout in (0, -1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                self.run_code("pass", timeout)

    def test_expired_capture_and_symlink(self):
        self.source.write_bytes(b"state")
        with self.assertRaises(TimeoutError):
            capture.capture(self.source, self.target, time.monotonic() - 1)
        self.assertFalse(self.target.exists())
        link = self.root / "link"
        link.symlink_to(self.source)
        with self.assertRaises(OSError):
            capture.capture(link, self.target, time.monotonic() + 3)
        self.assertFalse(self.target.exists())


if __name__ == "__main__":
    unittest.main()
