import hashlib
from pathlib import Path
import tempfile
import unittest
from deduplicate_checkpoint import deduplicate


class ImmutableDedupTests(unittest.TestCase):
    def test_paths_content_readonly_and_idempotence(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=Path(d)/'checkpoint',Path(d)/'object'
            content=b'exact checkpoint bytes';sha=hashlib.sha256(content).hexdigest()
            a.write_bytes(content);b.write_bytes(content)
            result=deduplicate(a,b,sha)
            self.assertEqual(a.read_bytes(),content);self.assertEqual(b.read_bytes(),content)
            self.assertEqual(a.stat().st_ino,b.stat().st_ino)
            self.assertEqual(a.stat().st_mode&0o222,0)
            self.assertTrue(result['source_path_preserved'])
            self.assertTrue(deduplicate(a,b,sha)['already_shared'])

    def test_mismatch_preserves_both(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=Path(d)/'checkpoint',Path(d)/'object';a.write_bytes(b'aaa');b.write_bytes(b'bbb')
            ino=a.stat().st_ino
            with self.assertRaises(ValueError):deduplicate(a,b,hashlib.sha256(b'aaa').hexdigest())
            self.assertEqual(a.stat().st_ino,ino);self.assertEqual(a.read_bytes(),b'aaa');self.assertEqual(b.read_bytes(),b'bbb')


if __name__=='__main__':unittest.main()
