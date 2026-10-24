import gzip
import hashlib
from pathlib import Path
import tempfile
import unittest

from compressed_checkpoint import compress, verify, restore


class CheckpointEncodingTests(unittest.TestCase):
    def test_exact_roundtrip_and_immutable_destination(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'state.pt'
            encoded = Path(folder) / 'state.pt.gz'
            payload = bytes(range(256)) * 8192
            source.write_bytes(payload)
            sha = hashlib.sha256(payload).hexdigest()
            result = compress(source, encoded, sha)
            self.assertEqual(gzip.decompress(encoded.read_bytes()), payload)
            self.assertEqual(result['raw_bytes'], len(payload))
            before = encoded.read_bytes()
            with self.assertRaises(FileExistsError):
                compress(source, encoded, sha)
            self.assertEqual(encoded.read_bytes(), before)
            restored = Path(folder) / 'restored.pt'
            restore(encoded, restored, sha, len(payload))
            self.assertEqual(restored.read_bytes(), payload)
            with self.assertRaises(FileExistsError):
                restore(encoded, restored, sha, len(payload))

    def test_bad_source_hash_is_not_published(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder) / 's', Path(folder) / 't'
            source.write_bytes(b'checkpoint')
            with self.assertRaises(ValueError):
                compress(source, target, '0' * 64)
            self.assertFalse(target.exists())
            self.assertEqual(list(Path(folder).iterdir()), [source])

    def test_incomplete_corrupt_and_oversized_decode(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 't'
            data = b'a' * 10000
            encoded = gzip.compress(data)
            sha = hashlib.sha256(data).hexdigest()
            for bad in [encoded[:-5], encoded[:-8] + bytes(8)]:
                target.write_bytes(bad)
                with self.assertRaises((EOFError, OSError, ValueError)):
                    verify(target, sha, len(data))
                restored = Path(folder) / 'restored.pt'
                with self.assertRaises((EOFError, OSError, ValueError)):
                    restore(target, restored, sha, len(data))
                self.assertFalse(restored.exists())
            target.write_bytes(encoded)
            with self.assertRaises(ValueError):
                verify(target, sha, len(data) - 1)
            with self.assertRaises(ValueError):
                verify(target, '0' * 64, len(data))


if __name__ == '__main__':
    unittest.main()
