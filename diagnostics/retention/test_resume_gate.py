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

class VariantFlagTests(unittest.TestCase):
    def test_exact_variant_flags_and_default_compatibility(self):
        from resume_gate import variant_flags
        self.assertEqual(variant_flags('train_loss.py'),[])
        self.assertEqual(variant_flags('train_selfplay.py'),[])
        self.assertEqual(variant_flags('train_loss.py','relative'),['--aim-decoding','relative'])
        self.assertEqual(variant_flags('train_loss.py','absolute',True),['--mirror-augment'])
        for entry,aim,mirror in [('train_loss.py','relative',True),('train_selfplay.py','relative',False),
                                 ('train_qcompletion.py','absolute',True),('train_loss.py','typo',False)]:
            with self.assertRaises(ValueError):variant_flags(entry,aim,mirror)


class VariantCommandTests(unittest.TestCase):
    def test_all_split_commands_carry_variant_and_cuda_flags(self):
        from unittest.mock import patch
        import resume_gate
        for flags, aim, mirror in [(['--aim-decoding','relative'],'relative',False),(['--mirror-augment'],'absolute',True)]:
            with self.subTest(flags=flags), tempfile.TemporaryDirectory() as d:
                root=Path(d); commands=[]
                def trainer(command, **kwargs):
                    commands.append(command)
                    out=Path(command[command.index('--output')+1]);out.mkdir()
                    games=int(command[command.index('--games')+1])
                    totals=dict(games=games,optimizer_updates=games,training_examples=games*8)
                    arguments=dict(output=str(out),resume_state='restored' if '--resume-state' in command else None,
                                   games=games,updates_per_game=1,aim_decoding=aim,mirror_augment=mirror)
                    torch.save(dict(totals=totals,arguments=arguments,model={'w':torch.tensor([1.])},optimizer={},replay=[],rng=[]),out/'resume-state.pt')
                    (out/'report.json').write_text(json.dumps(dict(totals,requested_run_completed=True)))
                args=['resume_gate.py','--source',str(root/'source'),'--site',str(root/'site'),
                      '--output',str(root/'output'),'--device','cuda','--entrypoint','train_loss.py']+flags
                with patch('sys.argv',args),patch.object(resume_gate.subprocess,'run',side_effect=trainer):
                    resume_gate.main()
                self.assertEqual(len(commands),3)
                for cmd in commands:
                    for flag in flags+['--device-history','--cuda-graph','--tf32']:
                        self.assertIn(flag,cmd)
                result=json.loads((root/'output/result.json').read_text())
                self.assertEqual(result['aim_decoding'],aim);self.assertEqual(result['mirror_augment'],mirror)

if __name__=='__main__':unittest.main()
