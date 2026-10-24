import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from evaluate_endpoint import catalog


class EndpointTests(unittest.TestCase):
    def test_frozen_endpoint_and_paired_initial_binding(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=root/'a';b=root/'b';a.mkdir();b.mkdir()
            for p in [a,b]:(p/'initial.npz').write_bytes(b'initial')
            (a/'weights.npz').write_bytes(b'weights');opp=root/'opponent.npz';opp.write_bytes(b'opp')
            sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            report=dict(requested_run_completed=True,games=2048,optimizer_updates=2048,checkpoint_sha256=sha(a/'weights.npz'))
            (a/'report.json').write_text(json.dumps(report));(a/'manifest.json').write_text(json.dumps({'config':{'simulations':32}}))
            opponent=dict(checkpoint=str(opp),sha256=sha(opp))
            result=catalog(a,b,opponent,'candidate');self.assertEqual(result['models'][0]['config']['simulations'],32)
            (b/'initial.npz').write_bytes(b'different')
            with self.assertRaises(ValueError):catalog(a,b,opponent,'candidate')
            (b/'initial.npz').write_bytes(b'initial');report['games']=1024;(a/'report.json').write_text(json.dumps(report))
            with self.assertRaises(ValueError):catalog(a,b,opponent,'candidate')


if __name__=='__main__':unittest.main()
