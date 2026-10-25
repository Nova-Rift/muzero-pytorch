"""Contract mutation tests; no training, large copies or checkpoint decoding."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import package
import run
WORKSPACE=HERE.parents[2]
ROOT=WORKSPACE.parents[2]
FROZEN=ROOT/'.local/research-loop/iteration34-frozen-package'
BASE=FROZEN/'staged/payload/experiment'
ORIGINAL=FROZEN/'runner-ego/artifacts/loop-a/EGO2048'

@unittest.skipUnless(ORIGINAL.exists(), 'Private frozen integration evidence required')
class ContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest=json.loads((ORIGINAL/'manifest.json').read_text())
        cls.receipt=json.loads((BASE/'model/.source-provenance.json').read_text())
    def test_frozen_source_all_files(self):
        run.validate_manifest(self.manifest,self.receipt)
        for name,sha in self.receipt['files'].items():self.assertEqual(run.digest(BASE/'model'/name),sha,name)
    def test_mutated_scientific_contract_rejected(self):
        for key,value in {'loss_contract':'root5-mass-v1','games':4096,'updates_per_game':2,'batch':128,'seed':1,'aim_decoding':'absolute','mirror_augment':True,'resume_state':'weights.npz','device':'cpu','replay_blocks':2}.items():
            with self.subTest(key=key):
                m=copy.deepcopy(self.manifest);m['arguments'][key]=value
                with self.assertRaises(ValueError):run.validate_manifest(m,self.receipt)
    def test_changed_source_runtime_rejected(self):
        for key,value in [('source_commit','0'*40),('torch','2.2.0+cu121'),('python','3.11.1')]:
            m=copy.deepcopy(self.manifest);m[key]=value
            with self.assertRaises(ValueError):run.validate_manifest(m,self.receipt)
    def test_command_preserves_all_scientific_flags(self):
        command=run.training_command(self.manifest,'source','site','output','state')
        args={};i=2
        while i<len(command):
            key=command[i][2:].replace('-','_');i+=1
            if i==len(command) or command[i].startswith('--'):args[key]=True
            else:args[key]=command[i];i+=1
        for key,value in self.manifest['arguments'].items():
            if key in run.ALLOW:continue
            if value is False:self.assertNotIn(key,args)
            else:self.assertEqual(str(args[key]),str(value),key)
        self.assertEqual(args['games'],'4096');self.assertEqual(args['resume_state'],'state')
        self.assertEqual(args['seed'],'52731415');self.assertNotIn('--weights',command)
    def test_complete_strict_dependency_chain(self):
        rows=package.jobs(self.manifest);self.assertEqual(len(rows),12)
        for i,row in enumerate(rows):self.assertEqual(row['depends_on'],[rows[i-1]['id']] if i else [])
        self.assertEqual([x['stage'] for x in rows[:5]],['test','test','test','train','test'])
        self.assertEqual(sum(int(x['command'][1][:-1]) for x in rows),2160)
        self.assertEqual(sum(x['stage']=='evaluate' for x in rows),7)
        for row in rows[5:]:self.assertNotIn('--declared-difference',row['command'])
    def test_matched_seeds_and_separated_direct(self):
        rows=package.jobs(self.manifest)[5:]
        seeds=[int(r['command'][r['command'].index('--seed')+1]) for r in rows]
        self.assertEqual(seeds,[131110000]*2+[131210000]*2+[131310000]*2+[131410000])
    def test_original_checkpoint_identity(self):
        encoded=json.loads((ORIGINAL/'compressed/manifest.json').read_text())['states'][run.NAME]
        self.assertEqual(encoded['raw_sha256'],run.RAW);self.assertEqual(encoded['compressed_sha256'],run.ENCODED)
        self.assertEqual(run.digest(ORIGINAL/'compressed'/(run.NAME+'.gz')),run.ENCODED)
        self.assertEqual(run.digest(ORIGINAL/'initial.npz'),run.SHA_INITIAL)
        self.assertEqual(run.digest(ORIGINAL/'weights.npz'),run.SHA_WEIGHTS)
    def test_payload_recipe_hashes_and_no_raw_state(self):
        recipe=json.loads((WORKSPACE/'prepared/payload-inputs.json').read_text());targets=set()
        for item in recipe['files']:
            self.assertNotIn(item['target'],targets);targets.add(item['target'])
            self.assertEqual(run.digest(item['local']),item['sha256'],item['target'])
            self.assertFalse(item['target'].endswith('.pt'))
        self.assertIn('experiment/ego-continuation/run.py',targets)
        self.assertIn('experiment/ego2048/compressed/'+run.NAME+'.gz',targets)
        self.assertFalse(any('/baseline2048/' in x for x in targets))
    def test_relative_own_initial_and_2048_opponents(self):
        for name,expected in [('OwnInitial',run.SHA_INITIAL),('EGO2048',run.SHA_WEIGHTS)]:
            item=json.loads((WORKSPACE/'prepared'/(name+'.json')).read_text())
            self.assertEqual(item['aim_decoding'],'relative');self.assertEqual(item['sha256'],expected)

if __name__=='__main__':unittest.main()
