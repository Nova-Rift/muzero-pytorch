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




class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.a=self.root/'a';self.b=self.root/'b';self.source=self.root/'source'
        for p in (self.a,self.b,self.source):p.mkdir()
        for p in (self.a,self.b):(p/'initial.npz').write_bytes(b'initial')
        (self.a/'weights.npz').write_bytes(b'weights')
        self.opp=self.root/'opponent.npz';self.opp.write_bytes(b'opponent')
        self.sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        self.opponent=dict(checkpoint=str(self.opp),sha256=self.sha(self.opp))
        (self.source/'train.py').write_text('frozen trainer')
        provenance=dict(commit='a'*40,dirty=False,files={'train.py':self.sha(self.source/'train.py')})
        (self.source/'.source-provenance.json').write_text(json.dumps(provenance))
        self.manifest=dict(config={'simulations':32},environment={'physics':'hash'},torch='2.8',python='3.12',
                           source_commit='a'*40,source_provenance=provenance,
                           arguments=dict(games=4096,updates_per_game=1,seed=52731415,resume_state='state-2048.pt'))
        original=json.loads(json.dumps(self.manifest));original['arguments'].update(games=2048,resume_state=None)
        (self.b/'manifest.json').write_text(json.dumps(original))
        self.report=dict(requested_run_completed=True,games=4096,optimizer_updates=4096,
                         checkpoint_sha256=self.sha(self.a/'weights.npz'))
        self.save()

    def save(self):
        (self.a/'manifest.json').write_text(json.dumps(self.manifest))
        (self.a/'report.json').write_text(json.dumps(self.report))

    def evaluate(self):
        return catalog(self.a,self.b,self.opponent,'continued',4096,self.source)

    def test_valid_continuation(self):
        self.assertEqual(self.evaluate()['models'][0]['sha256'],self.report['checkpoint_sha256'])

    def test_default_still_rejects_4096(self):
        with self.assertRaises(ValueError):catalog(self.a,self.b,self.opponent,'continued')

    def test_wrong_counts_and_incomplete(self):
        for key,value in [('games',2048),('optimizer_updates',4095),('requested_run_completed',False),
                          ('requested_run_completed','true'),('games',4096.0)]:
            with self.subTest(key=key,value=value):
                previous=self.report[key];self.report[key]=value;self.save()
                with self.assertRaises(ValueError):self.evaluate()
                self.report[key]=previous

    def test_weight_initial_opponent_mismatches(self):
        for path in (self.a/'weights.npz',self.b/'initial.npz',self.opp):
            with self.subTest(path=path.name):
                old=path.read_bytes();path.write_bytes(b'changed')
                with self.assertRaises(ValueError):self.evaluate()
                path.write_bytes(old)

    def test_manifest_bindings(self):
        for field,value in [('config',{'simulations':33}),('environment',{'physics':'changed'}),
                            ('source_commit','b'*40),('torch','other'),('python','other')]:
            with self.subTest(field=field):
                old=self.manifest[field];self.manifest[field]=value;self.save()
                with self.assertRaises(ValueError):self.evaluate()
                self.manifest[field]=old

    def test_argument_contract(self):
        for field,value in [('seed',42),('resume_state',None),('games',2048),('updates_per_game',2)]:
            with self.subTest(field=field):
                old=self.manifest['arguments'][field];self.manifest['arguments'][field]=value;self.save()
                with self.assertRaises(ValueError):self.evaluate()
                self.manifest['arguments'][field]=old

    def test_source_tamper(self):
        (self.source/'train.py').write_text('different trainer')
        with self.assertRaises(ValueError):self.evaluate()

    def test_dirty_provenance(self):
        self.manifest['source_provenance']['dirty']=True;self.save()
        with self.assertRaises(ValueError):self.evaluate()

    def test_missing_source(self):
        with self.assertRaises(ValueError):catalog(self.a,self.b,self.opponent,'continued',4096)

    def test_cli_uses_separate_frozen_source(self):
        from unittest.mock import patch
        import evaluate_endpoint
        opponent=self.root/'opponent.json';opponent.write_text(json.dumps(self.opponent))
        output=self.root/'evaluation'
        args=['evaluate_endpoint.py','--training',str(self.a),'--paired-training',str(self.b),
              '--opponent',str(opponent),'--label','continued','--output',str(output),
              '--source',str(self.source),'--site',str(self.root/'site'),'--seed','92820000',
              '--expected-endpoint-games','4096']
        with patch('sys.argv',args), patch.object(evaluate_endpoint.subprocess,'run') as run:
            evaluate_endpoint.main()
        command=run.call_args.args[0]
        self.assertEqual(command[1],str(self.source/'tournament/run.py'))
        self.assertEqual(command[command.index('--games')+1],'64')
        self.assertEqual(json.loads((output/'catalog.json').read_text())['models'][0]['sha256'],
                         self.report['checkpoint_sha256'])




class UPG2Tests(ContinuationTests):
    def setUp(self):
        super().setUp()
        from evaluate_endpoint import UPG2_TRAINER
        self.manifest['source_commit'] = UPG2_TRAINER
        self.manifest['source_provenance']['commit'] = UPG2_TRAINER
        (self.source/'.source-provenance.json').write_text(json.dumps(self.manifest['source_provenance']))
        self.manifest['arguments'].update(games=2048, updates_per_game=2, resume_state=None,
                                         batch=256, loss_contract='baseline')
        original=json.loads(json.dumps(self.manifest));original['arguments']['updates_per_game']=1
        (self.b/'manifest.json').write_text(json.dumps(original))
        self.report.update(games=2048, optimizer_updates=4096, training_examples=1048576)
        self.save()

    def evaluate(self):
        return catalog(self.a,self.b,self.opponent,'UPG2',2048,self.source,4096)

    def test_valid_upg2(self):
        self.assertEqual(self.evaluate()['models'][0]['sha256'],self.report['checkpoint_sha256'])

    def test_argument_contract(self):
        for field,value in [('seed',42),('resume_state','state.pt'),('games',4096),
                            ('updates_per_game',1),('loss_contract','root5-mass-v1'),('batch',128)]:
            with self.subTest(field=field):
                old=self.manifest['arguments'][field];self.manifest['arguments'][field]=value;self.save()
                with self.assertRaises(ValueError):self.evaluate()
                self.manifest['arguments'][field]=old

    def test_wrong_counts_and_incomplete(self):
        for key,value in [('games',4096),('optimizer_updates',2048),('training_examples',524288),
                          ('requested_run_completed',False),('games',2048.0)]:
            with self.subTest(key=key,value=value):
                old=self.report[key];self.report[key]=value;self.save()
                with self.assertRaises(ValueError):self.evaluate()
                self.report[key]=old

    def test_consistently_wrong_source_and_objective(self):
        for field,value in [('source_commit','b'*40),('loss_contract','first-transition1-v1')]:
            original=json.loads((self.b/'manifest.json').read_text())
            if field=='loss_contract':
                original['arguments'][field]=value;self.manifest['arguments'][field]=value
            else:
                original[field]=value;self.manifest[field]=value
                original['source_provenance']['commit']=value;self.manifest['source_provenance']['commit']=value
            (self.b/'manifest.json').write_text(json.dumps(original));self.save()
            with self.assertRaises(ValueError):self.evaluate()

    def test_arbitrary_count_override_rejected(self):
        for updates in (2047,4095,4096.0,True):
            with self.assertRaises(ValueError):
                catalog(self.a,self.b,self.opponent,'UPG2',2048,self.source,updates)

    def test_cli_uses_separate_frozen_source(self):
        from unittest.mock import patch
        import evaluate_endpoint
        opponent=self.root/'opponent.json';opponent.write_text(json.dumps(self.opponent))
        output=self.root/'evaluation'
        args=['evaluate_endpoint.py','--training',str(self.a),'--paired-training',str(self.b),
              '--opponent',str(opponent),'--label','UPG2','--output',str(output),
              '--source',str(self.source),'--site',str(self.root/'site'),'--seed','93020000',
              '--expected-endpoint-games','2048','--expected-endpoint-updates','4096']
        with patch('sys.argv',args), patch.object(evaluate_endpoint.subprocess,'run') as run:
            evaluate_endpoint.main()
        self.assertEqual(run.call_args.args[0][1],str(self.source/'tournament/run.py'))


if __name__=='__main__':unittest.main()
