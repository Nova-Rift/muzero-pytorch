import json
from pathlib import Path
import tempfile
import unittest
from materialize import stage,validate
from run import digest
class MaterializeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.input=self.root/'input';self.input.write_bytes(b'small test artifact')
        self.recipe={'files':[{'local':str(self.input),'target':'experiment/file','bytes':self.input.stat().st_size,'sha256':digest(self.input)}],'total_bytes':self.input.stat().st_size}
    def test_hash_verified_stage_and_archive(self):
        recipe=self.root/'recipe.json';recipe.write_text(json.dumps(self.recipe))
        result=stage(recipe,self.root/'stage',self.root/'payload.tar.gz')
        self.assertEqual((self.root/'stage/experiment/file').read_bytes(),self.input.read_bytes())
        self.assertEqual(result['archive']['sha256'],digest(self.root/'payload.tar.gz'))
    def test_modified_source_rejected(self):
        self.input.write_bytes(b'bad')
        with self.assertRaises(ValueError):validate(self.recipe)
    def test_traversal_and_duplicate_targets_rejected(self):
        self.recipe['files'][0]['target']='../escape'
        with self.assertRaises(ValueError):validate(self.recipe)
        self.recipe['files'][0]['target']='good';self.recipe['files']*=2
        with self.assertRaises(ValueError):validate(self.recipe)
if __name__=='__main__':unittest.main()
