import gzip,hashlib,json,tempfile,unittest,shutil
from pathlib import Path
from restore_continuation import restore,digest
class TestRestore(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory();self.addCleanup(self.t.cleanup);self.p=Path(self.t.name);self.source=self.p/'source';self.base=self.p/'baseline';self.base.mkdir();d=self.source/'diagnostics/retention';d.mkdir(parents=True)
  shutil.copyfile(Path(__file__).with_name('compressed_checkpoint.py'),d/'compressed_checkpoint.py')
  prov=dict(commit='fixed',dirty=False,files={'diagnostics/retention/compressed_checkpoint.py':digest(d/'compressed_checkpoint.py')})
  (self.source/'.source-provenance.json').write_text(json.dumps(prov));(self.base/'manifest.json').write_text(json.dumps(dict(torch='T',python='P',source_provenance=prov)))
  self.raw=b'exact original checkpoint bytes'*100;self.gz=self.p/'state.gz';self.gz.write_bytes(gzip.compress(self.raw));self.binding=dict(trainer_commit='fixed',baseline_files={'manifest.json':digest(self.base/'manifest.json')},compressed_bytes=self.gz.stat().st_size,compressed_sha256=digest(self.gz),raw_bytes=len(self.raw),raw_sha256=hashlib.sha256(self.raw).hexdigest())
 def run_restore(self,torch='T'):return restore(self.source,self.base,self.gz,self.p/'out',self.binding,torch,'P')
 def test_exact(self):
  self.assertTrue(self.run_restore()['restored']);self.assertEqual((self.p/'out/resume-state.pt').read_bytes(),self.raw)
  with self.assertRaises(FileExistsError):self.run_restore()
 def test_runtime(self):
  with self.assertRaises(ValueError):self.run_restore('other')
  self.assertFalse((self.p/'out').exists())
 def test_source(self):
  (self.source/'diagnostics/retention/compressed_checkpoint.py').write_text('changed')
  with self.assertRaises(ValueError):self.run_restore()
 def test_compressed(self):
  self.gz.write_bytes(b'changed')
  with self.assertRaises(ValueError):self.run_restore()
 def test_raw_binding(self):
  self.binding['raw_sha256']='0'*64
  with self.assertRaises(ValueError):self.run_restore()
  self.assertFalse((self.p/'out/resume-state.pt').exists())
if __name__=='__main__':unittest.main()
