import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import torch
from resume_gate import compare

HERE=Path(__file__).resolve().parent


class RuntimeGateTests(unittest.TestCase):
    def test_state_comparison_includes_replay_optimizer_rng(self):
        state=dict(arguments={'output':'a','resume_state':None,'games':4},
                   replay=[torch.tensor([1.])],optimizer={'step':torch.tensor(4)},rng=torch.tensor([9]))
        same=dict(state,arguments=dict(state['arguments'],output='b',resume_state='old'))
        self.assertEqual(compare(state,same)['tensors'],3)
        for key,replacement in [('replay',[torch.tensor([2.])]),('optimizer',{'step':torch.tensor(3)}),('rng',torch.tensor([8]))]:
            with self.assertRaises(ValueError):compare(state,dict(same,**{key:replacement}))

    def test_encoder_success_and_storage_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);producer=root/'producer.py'
            producer.write_text("import sys,json\nfrom pathlib import Path\np=Path(sys.argv[1]);(p/'full-state').mkdir(parents=True)\n(p/'full-state/games-4-updates-4.pt').write_bytes(bytes(10000))\n(p/'report.json').write_text(json.dumps({'requested_run_completed':True}))\n")
            for label,budget,success in [('good',10000,True),('bounded',1,False)]:
                out=root/label
                result=subprocess.run([sys.executable,str(HERE/'train_compressed.py'),'--output',str(out),
                                       '--seconds','10','--max-encoded-bytes',str(budget),'--',sys.executable,str(producer),str(out)],
                                      timeout=15,capture_output=True,text=True)
                self.assertEqual(result.returncode==0,success,result.stderr)
                self.assertTrue((out/'compressed/manifest.json').is_file())
                if success:
                    self.assertTrue(json.loads((out/'encoding-result.json').read_text())['pod_retained'])
                else:self.assertFalse((out/'encoding-result.json').exists())


if __name__=='__main__':unittest.main()
