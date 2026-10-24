import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import archive_frozen as subject


class FrozenArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.archive = self.root / 'archive'

    def manifest(self, values):
        files = []
        for name, data in values:
            p = self.root / name
            p.write_bytes(data)
            files.append(dict(path=str(p), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)))
        return dict(frozen_closed_outputs=True, files=files)

    def test_new_objects_preserve_bytes_paths_and_schema_bounded_peak(self):
        m = self.manifest([('a.zlib', b'a'*10000), ('b.zlib', b'b'*20000)])
        r = subject.snapshot_frozen('candidate', m, self.archive)
        self.assertEqual(r['peak_temporary_bytes'], 20000)
        self.assertEqual(r['copied_bytes'], 30000)
        self.assertEqual(r['hashed_bytes'], 30000)
        record = json.loads(Path(r['record']).read_text())
        self.assertEqual(record['schema'], 1)
        self.assertEqual(record['omitted'], [])
        for f in record['files']:
            p, o = Path(f['path']), self.archive/'objects'/f['sha256']
            self.assertEqual(p.stat().st_ino, o.stat().st_ino)
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), f['sha256'])
            self.assertEqual(p.stat().st_mode & 0o222, 0)
            self.assertIn('mtime_ns', f['original'])

    def test_existing_objects_no_copy_and_shared_inode_hashed_once(self):
        m = self.manifest([('a.zlib', b'a'*5000)])
        subject.snapshot_frozen('candidate', m, self.archive)
        r = subject.snapshot_frozen('candidate', m, self.archive)
        self.assertEqual(r['copied_bytes'], 0)
        self.assertEqual(r['peak_temporary_bytes'], 0)
        self.assertEqual(r['hashed_bytes'], 5000)
        self.assertEqual(r['already_shared'], 1)

    def test_existing_duplicate_uses_no_temporary_payload(self):
        m = self.manifest([('a.zlib', b'a'*5000), ('b.zlib', b'a'*5000)])
        r = subject.snapshot_frozen('candidate', m, self.archive)
        self.assertEqual(r['copied_bytes'], 5000)
        self.assertEqual(r['hashed_bytes'], 10000)
        self.assertEqual((self.root/'a.zlib').stat().st_ino,(self.root/'b.zlib').stat().st_ino)

    def test_corrupt_existing_object_never_overwritten(self):
        m = self.manifest([('a.zlib', b'good')])
        objects = self.archive/'objects';objects.mkdir(parents=True)
        obj = objects/m['files'][0]['sha256'];obj.write_bytes(b'evil')
        with self.assertRaises(ValueError):subject.snapshot_frozen('candidate',m,self.archive)
        self.assertEqual(obj.read_bytes(),b'evil')
        self.assertEqual((self.root/'a.zlib').read_bytes(),b'good')

    def test_wrong_expected_hash_preserves_source_and_cleans_temp(self):
        m=self.manifest([('a.zlib',b'good')]);m['files'][0]['sha256']='0'*64
        with self.assertRaises(ValueError):subject.snapshot_frozen('candidate',m,self.archive)
        self.assertEqual((self.root/'a.zlib').read_bytes(),b'good')
        self.assertEqual(list((self.archive/'objects').iterdir()),[])

    def test_requires_explicit_frozen_admission(self):
        m=self.manifest([('a.zlib',b'good')]);m['frozen_closed_outputs']=False
        with self.assertRaises(ValueError):subject.snapshot_frozen('candidate',m,self.archive)

    def test_executable_sensitive_and_symlink_rejected(self):
        m=self.manifest([('a.zlib',b'good')]);p=self.root/'a.zlib';p.chmod(0o700)
        with self.assertRaises(ValueError):subject.snapshot_frozen('candidate',m,self.archive)
        p.chmod(0o600)
        m=self.manifest([('api-key',b'good')])
        with self.assertRaises(ValueError):subject.snapshot_frozen('candidate',m,self.archive)
        m=self.manifest([('b.zlib',b'good')]);p=self.root/'b.zlib';p.unlink();p.symlink_to('a.zlib')
        with self.assertRaises(ValueError):subject.snapshot_frozen('candidate',m,self.archive)

    def test_source_change_before_link_rejected(self):
        m=self.manifest([('a.zlib',b'good')]);p=self.root/'a.zlib'
        original=subject.sync_dir
        def mutate(directory):
            original(directory)
            if directory.name=='objects':p.write_bytes(b'evil')
        with patch.object(subject,'sync_dir',side_effect=mutate):
            with self.assertRaises(ValueError):subject.snapshot_frozen('candidate',m,self.archive)
        self.assertEqual(p.read_bytes(),b'evil')
        obj=self.archive/'objects'/m['files'][0]['sha256']
        self.assertEqual(obj.read_bytes(),b'good')
        self.assertFalse(list((self.archive/'records').glob('*.json')))

    def test_new_multiply_linked_source_rejected_for_peak_space_bound(self):
        m=self.manifest([('a.zlib',b'good')]);os.link(self.root/'a.zlib',self.root/'alias.zlib')
        with self.assertRaises(ValueError):subject.snapshot_frozen('candidate',m,self.archive)
        self.assertEqual((self.root/'alias.zlib').read_bytes(),b'good')

    def test_existing_open_source_handle_cannot_change_archived_inode_after_replace(self):
        m=self.manifest([('a.zlib',b'good')]);p=self.root/'a.zlib'
        with p.open('r+b') as old:
            r=subject.snapshot_frozen('candidate',m,self.archive)
            old.write(b'evil');old.flush()
        self.assertEqual(p.read_bytes(),b'good')
        self.assertTrue(Path(r['record']).exists())


if __name__=='__main__':unittest.main()
