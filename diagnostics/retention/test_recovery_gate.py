import tempfile
import unittest
from pathlib import Path
import recovery_gate as gate
RETENTION=Path(__file__).resolve().parents[2]/'arena/state_history.py'
class GateTests(unittest.TestCase):
    def test_two_milestones_safe_load_and_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'out'
            r=gate.generate(out,RETENTION,gate.digest(RETENTION),4096)
            self.assertNotEqual(r['milestones'][0]['sha256'],r['milestones'][1]['sha256'])
            h=gate.digest(out/'manifest.json')
            self.assertTrue(gate.verify(out,h)['verified'])
            p=out/r['milestones'][0]['path']
            with p.open('r+b') as f: f.write(b'BAD')
            with self.assertRaises(ValueError): gate.verify(out,h)
    def test_reject_binding_before_output(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'out'
            with self.assertRaises(ValueError): gate.generate(out,RETENTION,'0'*64,4096)
            self.assertFalse(out.exists())
    def test_reject_manifest_change(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'out'; gate.generate(out,RETENTION,gate.digest(RETENTION),4096)
            with self.assertRaises(ValueError): gate.verify(out,'0'*64)
if __name__=='__main__': unittest.main()
