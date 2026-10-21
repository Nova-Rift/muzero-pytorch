import hashlib,json,tempfile,unittest
from pathlib import Path
from arena.provenance import capture_source
class ProvenanceTests(unittest.TestCase):
 def test_cached_receipt_rejects_modified_code(self):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);file=root/'source.py';file.write_text('original')
   receipt={'commit':'example','dirty':False,'files':{'source.py':hashlib.sha256(file.read_bytes()).hexdigest()}}
   (root/'.source-provenance.json').write_text(json.dumps(receipt))
   self.assertEqual(capture_source(root)['kind'],'verified-cache-receipt')
   file.write_text('modified')
   with self.assertRaisesRegex(ValueError,'Cached source changed'):capture_source(root)
if __name__=='__main__':unittest.main()
