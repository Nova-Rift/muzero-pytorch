import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import torch
from resume_gate import compare, validate_counts, validate_upg2_source, UPG2_TRAINER


class ResumeTests(unittest.TestCase):
    def test_counts_both_supported_ratios(self):
        for ratio in (1, 2):
            totals=dict(games=4, optimizer_updates=4*ratio, training_examples=32*ratio)
            state=dict(totals=totals, arguments=dict(updates_per_game=ratio))
            report=dict(totals, requested_run_completed=True)
            self.assertEqual(validate_counts(state, report, 4, ratio), totals)
            for target in (totals, report):
                for key in ('games','optimizer_updates','training_examples'):
                    old=target[key]
                    for wrong in (old-1, float(old), True):
                        target[key]=wrong
                        with self.assertRaises(ValueError): validate_counts(state, report, 4, ratio)
                    target[key]=old
            report['requested_run_completed']=False
            with self.assertRaises(ValueError): validate_counts(state, report, 4, ratio)

    def test_ratio_mismatch_even_when_counters_match(self):
        totals=dict(games=4,optimizer_updates=8,training_examples=64)
        state=dict(totals=totals,arguments=dict(updates_per_game=1))
        with self.assertRaises(ValueError):
            validate_counts(state,dict(totals,requested_run_completed=True),4,2)

    def test_complete_equality_including_arguments(self):
        a=dict(arguments=dict(output='full',resume_state=None,updates_per_game=2),
               model={'w':torch.tensor([1.])},optimizer={'step':1},replay=[2],rng=[3])
        b=dict(a,arguments=dict(a['arguments'],output='resumed',resume_state='state.pt'))
        self.assertEqual(compare(a,b)['tensors'],1)
        for key,value in [('optimizer',{'step':2}),('replay',[3]),('rng',[4]),
                          ('model',{'w':torch.tensor([1.],dtype=torch.float64)}),
                          ('arguments',dict(b['arguments'],extra=True)),
                          ('arguments',dict(b['arguments'],updates_per_game=1))]:
            with self.subTest(key=key), self.assertRaises(ValueError):compare(a,dict(b,**{key:value}))

    def test_upg2_source_binding(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'trainer.py').write_text('frozen')
            source=dict(commit=UPG2_TRAINER,dirty=False,files={'trainer.py':hashlib.sha256(b'frozen').hexdigest()})
            receipt=root/'.source-provenance.json'; receipt.write_text(json.dumps(source))
            self.assertEqual(validate_upg2_source(root,'train_loss.py','baseline'),source)
            for entry,contract in [('train_selfplay.py','baseline'),('train_loss.py','root5-mass-v1')]:
                with self.assertRaises(ValueError):validate_upg2_source(root,entry,contract)
            for field,value in [('commit','wrong'),('dirty',True),('files',{})]:
                receipt.write_text(json.dumps(dict(source,**{field:value})))
                with self.assertRaises(ValueError):validate_upg2_source(root,'train_loss.py','baseline')
            receipt.write_text(json.dumps(source)); (root/'trainer.py').write_text('changed')
            with self.assertRaises(ValueError):validate_upg2_source(root,'train_loss.py','baseline')

if __name__=='__main__':unittest.main()
