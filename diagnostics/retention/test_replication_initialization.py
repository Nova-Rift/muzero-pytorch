"""Initializer-only tests. CPU parity is explicitly not CUDA admission evidence."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import replication_initialization as init


class InitializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not all(os.environ.get(k) for k in ('REPLICATION_FROZEN_SOURCE','REPLICATION_TEMPLATE')):
            raise unittest.SkipTest('Set frozen-source/template paths for source-bound initializer checks')
        cls.source=Path(os.environ['REPLICATION_FROZEN_SOURCE']).resolve()
        cls.template=Path(os.environ['REPLICATION_TEMPLATE']).resolve()
        cls.adapter=Path(init.__file__).parent

    def test_source_bound_cpu_prefix_and_export(self):
        # Separate interpreter prevents the adapter checkout's model from leaking in.
        script='''
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import replication_initialization as i
from evaluate_endpoint import bind_source
s=Path(sys.argv[2]);t=json.loads(Path(sys.argv[3]).read_text())
i.validate_template(t)
bind_source({'source_provenance':t['source'],'source_commit':i.rc.SOURCE_COMMIT},s,('replication',))
b,p=i.initialize_and_check(s,'cpu')
assert p['device']=='cpu' and not p['actual_trainer_export_verified']
assert p['tensor_count']>0 and p['initial_sha256']==hashlib.sha256(b).hexdigest()
print(json.dumps(p))
'''
        result=subprocess.run([sys.executable,'-c',script,str(self.adapter),str(self.source),str(self.template)],
                              text=True,capture_output=True,check=True,timeout=45)
        evidence=json.loads(result.stdout)
        self.assertTrue(evidence['tensor_bytes_equal'])
        self.assertTrue(evidence['npz_bytes_equal'])
        self.assertTrue(evidence['rng_states_equal'])
        print('CPU diagnostic evidence:',json.dumps(evidence,sort_keys=True))

    def test_reviewed_source_extracts(self):
        self.assertEqual(len(init.audited_statements(self.source).body),5)

    def test_source_sequence_mutations_fail_closed(self):
        original=(self.source/'arena/train_loss.py').read_text()
        for old,new in [('torch.manual_seed(a.seed)','torch.manual_seed(a.seed+1)'),
                        ('rng=np.random.default_rng(a.seed)','rng=np.random.default_rng(a.seed+1)'),
                        ('model_class(a.loss_contract)(Config','model_class("root5-mass-v1")(Config'),
                        ('observations=64*a.context+a.context-1','observations=65*a.context+a.context-1'),
                        ("np.savez_compressed(out/'initial.npz'","np.savez(out/'initial.npz'")]:
            with self.subTest(old=old),tempfile.TemporaryDirectory() as d:
                p=Path(d);(p/'arena').mkdir();(p/'arena/train_loss.py').write_text(original.replace(old,new))
                with self.assertRaises(ValueError):init.audited_statements(p)

    def test_cached_foreign_model_rejected(self):
        from types import SimpleNamespace
        with patch.dict(sys.modules,{'model':SimpleNamespace(__file__='/foreign/model.py')}):
            with self.assertRaisesRegex(ValueError,'Foreign cached trainer module'):
                init.frozen_factory(self.source)

    def test_cached_foreign_namespace_rejected(self):
        from types import SimpleNamespace
        with patch.dict(sys.modules,{'arena':SimpleNamespace(__path__=['/foreign/arena'])}):
            with self.assertRaisesRegex(ValueError,'Foreign cached trainer namespace'):
                init.frozen_factory(self.source)

    def test_production_generation_rejects_cpu_runtime(self):
        import torch
        with tempfile.TemporaryDirectory() as d,patch.object(torch,'__version__',init.RUNTIME['torch']),\
                patch.object(init.platform,'python_version',return_value=init.RUNTIME['python']),\
                patch.object(torch.cuda,'is_available',return_value=False):
            output=Path(d)/'result'
            with self.assertRaisesRegex(ValueError,'CPU hash cannot freeze'):
                init.generate(self.template,init.rc.digest(self.template),self.source,output)
            self.assertFalse(output.exists())

    def test_wrong_template_hash_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'result'
            with self.assertRaisesRegex(ValueError,'template digest'):
                init.generate(self.template,'f'*64,self.source,output)
            self.assertFalse(output.exists())

if __name__=='__main__':unittest.main()
