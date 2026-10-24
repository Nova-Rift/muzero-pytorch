import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('overlap',Path(__file__).with_name('overlap_benchmark.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class Tests(unittest.TestCase):
    def test_eligibility(self):
        mem=[dict(total_bytes=80*1024**3,used_bytes=50*1024**3)]
        self.assertTrue(m.assess(30,20,mem,True)['eligible'])
        self.assertFalse(m.assess(20,20,mem,True)['eligible'])
        self.assertFalse(m.assess(30,20,mem,False)['eligible'])
        self.assertFalse(m.assess(30,20,[dict(total_bytes=80*1024**3,used_bytes=75*1024**3)],True)['eligible'])
    def test_memory_requires_single_gpu(self):
        with patch.object(m.subprocess,'run') as run:
            run.return_value.stdout='GPU-A, 80000, 10000\n';self.assertEqual(m.memory_sample()['used_bytes'],10000*1048576)
            run.return_value.stdout='GPU-A, 80000, 10000\nGPU-B, 80000, 0\n'
            with self.assertRaises(ValueError):m.memory_sample()
    def test_group_timeout_reaps_child(self):
        with tempfile.TemporaryDirectory() as root:
            d=Path(root)/'x'
            with self.assertRaises(TimeoutError):
                m.run_group([[sys.executable,'-c','import time;time.sleep(20)']],[d],m.time.monotonic()+.1,lambda:dict(total_bytes=10,used_bytes=1))
            import json
            self.assertTrue(all(x is not None for x in json.loads((d/'measurement.json').read_text())['exit_codes']))
    def test_group_failure_preserves_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            d=Path(root)/'x'
            with self.assertRaises(RuntimeError):
                m.run_group([[sys.executable,'-c','raise RuntimeError("fixture")']],[d],m.time.monotonic()+2,lambda:dict(total_bytes=10,used_bytes=1))
            self.assertTrue((d/'measurement.json').exists());self.assertIn('fixture',(d/'process.log').read_text())
    def test_serial_and_overlap_run_same_work(self):
        with tempfile.TemporaryDirectory() as root:
            dirs=[Path(root)/str(i) for i in range(2)]
            r=m.run_group([[sys.executable,'-c','pass']]*2,dirs,m.time.monotonic()+2,lambda:dict(total_bytes=10,used_bytes=1))
            self.assertEqual(r['exit_codes'],[0,0]);self.assertTrue(r['samples'])
    def test_incomplete_is_ineligible(self):
        with tempfile.TemporaryDirectory() as root:
            bound=Path(root)/'bound';bound.write_text('fixture')
            plan=dict(approved=True,seconds=1,target_plan_sha256='a'*64,files={str(bound):m.digest(bound)},arms=[{'command':['x','{output}']}]*2)
            with patch.object(m,'run_group',side_effect=TimeoutError('fixture')):
                r=m.execute(plan,Path(root)/'result')
            self.assertFalse(r['complete']);self.assertFalse(r['eligible']);self.assertIn('fixture',r['error'])
    def test_wrong_binding_refused(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'bound';p.write_text('actual')
            with self.assertRaises(ValueError):m.execute(dict(approved=True,seconds=10,target_plan_sha256='a'*64,files={str(p):'wrong'},arms=[{}]*2),Path(root)/'out')

if __name__=='__main__':unittest.main()
