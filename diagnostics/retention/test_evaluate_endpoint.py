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

    def test_aim_decoding_bound_from_candidate_manifest_only(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=root/'a';b=root/'b';a.mkdir();b.mkdir()
            for p in [a,b]:(p/'initial.npz').write_bytes(b'initial')
            (a/'weights.npz').write_bytes(b'weights');opp=root/'opponent.npz';opp.write_bytes(b'opp')
            sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            report=dict(requested_run_completed=True,games=2048,optimizer_updates=2048,checkpoint_sha256=sha(a/'weights.npz'))
            (a/'report.json').write_text(json.dumps(report))
            opponent=dict(id='E1024',checkpoint=str(opp),sha256=sha(opp),config={'simulations':32})
            # Historical manifests (no aim_decoding argument) and explicit absolute runs: catalog unchanged, no key.
            for arguments in ({},{'aim_decoding':'absolute'}):
                (a/'manifest.json').write_text(json.dumps({'config':{'simulations':32},'arguments':arguments}))
                result=catalog(a,b,opponent,'candidate')
                self.assertNotIn('aim_decoding',result['models'][0]);self.assertEqual(result['models'][1],opponent)
            # Relative run: only the candidate entry carries its decoder; the opponent entry is untouched.
            (a/'manifest.json').write_text(json.dumps({'config':{'simulations':32},'arguments':{'aim_decoding':'relative'}}))
            result=catalog(a,b,opponent,'EGO2048')
            self.assertEqual(result['models'][0]['aim_decoding'],'relative');self.assertEqual(result['models'][1],opponent)
            self.assertNotIn('aim_decoding',result['models'][1])
            (a/'manifest.json').write_text(json.dumps({'config':{'simulations':32},'arguments':{'aim_decoding':'mirror'}}))
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




class DeclaredDifferenceTests(unittest.TestCase):
    """--declared-difference: fresh 2048 and continuation paths, flag-absent byte identity."""
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.a=self.root/'a';self.b=self.root/'b'
        self.source=self.root/'source';self.paired_source=self.root/'paired-source'
        for p in (self.a,self.b,self.source,self.paired_source):p.mkdir()
        for p in (self.a,self.b):(p/'initial.npz').write_bytes(b'initial')
        for p in (self.a,self.b):(p/'weights.npz').write_bytes(b'weights-'+p.name.encode())
        self.opp=self.root/'opponent.npz';self.opp.write_bytes(b'opponent')
        self.sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        self.opponent=dict(checkpoint=str(self.opp),sha256=self.sha(self.opp))
        (self.paired_source/'train.py').write_text('frozen trainer')
        (self.source/'train.py').write_text('trainer with a new flag')
        self.paired_provenance=dict(commit='a'*40,dirty=False,files={'train.py':self.sha(self.paired_source/'train.py')})
        self.provenance=dict(commit='c'*40,dirty=False,files={'train.py':self.sha(self.source/'train.py')})
        (self.source/'.source-provenance.json').write_text(json.dumps(self.provenance))
        (self.paired_source/'.source-provenance.json').write_text(json.dumps(self.paired_provenance))
        common=dict(games=2048,updates_per_game=1,seed=52731415,resume_state=None,batch=256,loss_contract='baseline',output='x')
        self.original=dict(config={'simulations':32},environment={'physics':'hash'},torch='2.8',python='3.12',
                           source_commit='a'*40,source_provenance=self.paired_provenance,arguments=dict(common))
        self.manifest=dict(config={'simulations':32},environment={'physics':'hash'},torch='2.8',python='3.12',
                           source_commit='c'*40,source_provenance=self.provenance,
                           arguments=dict(common,output='y',aim_decoding='relative',mirror_augment=False))
        self.report=dict(requested_run_completed=True,games=2048,optimizer_updates=2048,checkpoint_sha256=self.sha(self.a/'weights.npz'))
        self.contract=dict(schema=1,mode='restart',candidate={k:self.provenance[k] for k in ('commit','files')},
                           paired={k:self.paired_provenance[k] for k in ('commit','files')},
                           arguments={k:dict(candidate=self.manifest['arguments'].get(k),paired={'aim_decoding':'absolute','mirror_augment':False}.get(k,self.original['arguments'].get(k)))
                                      for k in ('aim_decoding','mirror_augment')})
        self.save()

    def save(self):
        (self.a/'manifest.json').write_text(json.dumps(self.manifest));(self.a/'report.json').write_text(json.dumps(self.report))
        (self.b/'manifest.json').write_text(json.dumps(self.original))
        (self.b/'report.json').write_text(json.dumps(dict(self.report,checkpoint_sha256=self.sha(self.b/'weights.npz'))))

    def evaluate(self,declared=('aim_decoding','mirror_augment'),source=None,**kw):
        return catalog(self.a,self.b,self.opponent,'EGO2048',2048,self.source if source is None else source,2048,declared,kw.pop("source_contract", self.contract),**kw)

    def test_declared_candidate_bound_to_running_source(self):
        result=self.evaluate()
        self.assertEqual(result['models'][0]['sha256'],self.report['checkpoint_sha256'])
        self.assertEqual(result['models'][0]['aim_decoding'],'relative')
        self.assertNotIn('aim_decoding',result['models'][1])
        self.assertEqual(result['declared_differences'],['aim_decoding','mirror_augment'])
        self.assertEqual(result['declared_values'],{'aim_decoding':{'candidate':'relative','paired':'absolute'},
                                                    'mirror_augment':{'candidate':False,'paired':False}})
        self.assertEqual(result['candidate_source_commit'],'c'*40);self.assertEqual(result['paired_source_commit'],'a'*40)

    def test_flag_absent_catalog_is_unchanged(self):
        # Historical fresh-2048 path: no paired-manifest comparison, no provenance keys in the catalog.
        plain=catalog(self.a,self.b,self.opponent,'EGO2048',2048,self.source,2048)
        self.assertEqual(sorted(plain),['models']);self.assertEqual(len(plain['models']),2)
        self.manifest['arguments']['seed']=1;self.save()  # undeclared difference is invisible without the flag (historical)
        self.assertEqual(sorted(catalog(self.a,self.b,self.opponent,'EGO2048',2048,self.source,2048)),['models'])

    def test_undeclared_difference_rejected(self):
        self.manifest['arguments']['seed']=1;self.save()
        with self.assertRaisesRegex(ValueError,'objective differs'):self.evaluate()
        self.manifest['arguments']['seed']=52731415;self.save()
        with self.assertRaisesRegex(ValueError,'declared arguments'):self.evaluate(declared=('aim_decoding',))  # mirror_augment key undeclared
        with self.assertRaisesRegex(ValueError,'Undeclarable'):self.evaluate(declared=('aim_decoding','mirror_augment','seed'))
        with self.assertRaisesRegex(ValueError,'Duplicate'):self.evaluate(declared=('aim_decoding','aim_decoding','mirror_augment'))

    def test_partner_pool_flags_declarable_against_flag_absent_paired_manifest(self):
        # EGO-POOL2048 (research/iteration35-partner-pool) vs EGO2048 (14fece12): the paired manifest predates the flags,
        # so the contract carries their defaults ('' / 0.0); undeclared pool flags are still rejected.
        self.manifest['arguments'].update(partner_pool='initial,512,1024',partner_fraction=0.5);self.save()
        with self.assertRaisesRegex(ValueError,'declared arguments'):self.evaluate()
        declared=('aim_decoding','mirror_augment','partner_pool','partner_fraction')
        contract=dict(self.contract,arguments=dict(self.contract['arguments'],partner_pool=dict(candidate='initial,512,1024',paired=''),partner_fraction=dict(candidate=0.5,paired=0.0)))
        result=self.evaluate(declared=declared,source_contract=contract)
        self.assertEqual(result['declared_values']['partner_pool'],{'candidate':'initial,512,1024','paired':''})
        self.assertEqual(result['declared_values']['partner_fraction'],{'candidate':0.5,'paired':0.0})
        self.assertEqual(result['candidate_source_commit'],'c'*40);self.assertEqual(result['paired_source_commit'],'a'*40)
        wrong=dict(contract,arguments=dict(contract['arguments'],partner_fraction=dict(candidate=0.5,paired=None)))
        with self.assertRaisesRegex(ValueError,'argument value differs'):self.evaluate(declared=declared,source_contract=wrong)
        with self.assertRaisesRegex(ValueError,'Undeclarable'):self.evaluate(declared=declared+('seed',),source_contract=contract)

    def test_candidate_commit_must_match_source_checkout(self):
        # The paired source's receipt/commit does not satisfy the binding; the candidate's does.
        with self.assertRaises(ValueError):self.evaluate(source=self.paired_source)
        self.manifest['source_commit']='d'*40;self.manifest['source_provenance']=dict(self.provenance,commit='d'*40);self.save()
        with self.assertRaisesRegex(ValueError,'Source contract differs'):self.evaluate()

    def test_dirty_or_tampered_source_rejected(self):
        (self.source/'.source-provenance.json').write_text(json.dumps(dict(self.provenance,dirty=True)))
        with self.assertRaises(ValueError):self.evaluate()
        (self.source/'.source-provenance.json').write_text(json.dumps(self.provenance))
        (self.source/'train.py').write_text('edited after training')
        with self.assertRaisesRegex(ValueError,'source changed'):self.evaluate()
        with self.assertRaises(ValueError):self.evaluate(source=None) if False else catalog(self.a,self.b,self.opponent,'EGO2048',2048,None,2048,('aim_decoding','mirror_augment'))

    def test_config_environment_still_bound(self):
        for field,value in [('config',{'simulations':33}),('environment',{'physics':'changed'}),('torch','other'),('python','other')]:
            with self.subTest(field=field):
                old=self.manifest[field];self.manifest[field]=value;self.save()
                with self.assertRaisesRegex(ValueError,field):self.evaluate()
                self.manifest[field]=old

    def test_git_checkout_without_receipt_must_be_clean(self):
        import subprocess
        repo=self.root/'repo';repo.mkdir()
        subprocess.run(['git','init','-q',str(repo)],check=True)
        (repo/'train.py').write_text('trainer with a new flag')
        env=dict(GIT_AUTHOR_NAME='t',GIT_AUTHOR_EMAIL='t@t',GIT_COMMITTER_NAME='t',GIT_COMMITTER_EMAIL='t@t')
        import os;env={**os.environ,**env}
        subprocess.run(['git','-C',str(repo),'add','train.py'],check=True);subprocess.run(['git','-C',str(repo),'commit','-q','-m','x'],check=True,env=env)
        commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
        self.manifest['source_commit']=commit;self.manifest['source_provenance']=dict(self.provenance,commit=commit)
        self.contract['candidate']['commit']=commit;self.save()
        self.assertEqual(self.evaluate(source=repo)['candidate_source_commit'],commit)
        (repo/'train.py').write_text('trainer with a new flag')  # same bytes: still clean
        self.assertEqual(self.evaluate(source=repo)['candidate_source_commit'],commit)
        (repo/'other.py').write_text('tracked edit');subprocess.run(['git','-C',str(repo),'add','other.py'],check=True)
        with self.assertRaisesRegex(ValueError,'dirty'):self.evaluate(source=repo)

    def test_cross_source_continuation_rejected(self):
        self.manifest['arguments'].update(games=4096,resume_state='state-2048.pt')
        self.report.update(games=4096,optimizer_updates=4096);self.save()
        with self.assertRaisesRegex(ValueError,'restart'):
            catalog(self.a,self.b,self.opponent,'c',4096,self.source,4096,('aim_decoding','mirror_augment'),self.contract)

    def test_cli_flag_repeatable(self):
        from unittest.mock import patch
        import evaluate_endpoint
        opponent=self.root/'opponent.json';opponent.write_text(json.dumps(self.opponent));output=self.root/'evaluation'
        args=['evaluate_endpoint.py','--training',str(self.a),'--paired-training',str(self.b),'--opponent',str(opponent),'--label','EGO2048',
              '--output',str(output),'--source',str(self.source),'--site',str(self.root/'site'),'--seed','95010000',
              '--expected-endpoint-games','2048','--expected-endpoint-updates','2048','--declared-difference','aim_decoding','--declared-difference','mirror_augment']
        contract_path=self.root/'contract.json';contract_path.write_text(json.dumps(self.contract))
        args += ['--source-contract',str(contract_path)]
        with patch('sys.argv',args), patch.object(evaluate_endpoint.subprocess,'run') as run:evaluate_endpoint.main()
        written=json.loads((output/'catalog.json').read_text())
        self.assertEqual(written['declared_differences'],['aim_decoding','mirror_augment']);self.assertEqual(written['models'][0]['aim_decoding'],'relative')
        self.assertEqual(run.call_args.args[0][1],str(self.source/'tournament/run.py'))

    def test_missing_and_tampered_contract_rejected(self):
        import copy
        with self.assertRaisesRegex(ValueError,'source contract'):self.evaluate(source_contract=None)
        for side in ('candidate','paired'):
            for field in ('commit','files'):
                bad=copy.deepcopy(self.contract);bad[side][field]='tampered'
                with self.assertRaisesRegex(ValueError,'Source contract differs'):self.evaluate(source_contract=bad)
        bad=copy.deepcopy(self.contract);bad['arguments']['aim_decoding']['candidate']='absolute'
        with self.assertRaisesRegex(ValueError,'argument value'):self.evaluate(source_contract=bad)

    def test_mirror_positive_and_combination_rejected(self):
        self.manifest['arguments'].update(aim_decoding='absolute',mirror_augment=True)
        self.contract['arguments']['aim_decoding']['candidate']='absolute'
        self.contract['arguments']['mirror_augment']['candidate']=True;self.save()
        result=self.evaluate();self.assertNotIn('aim_decoding',result['models'][0])
        self.manifest['arguments']['aim_decoding']='relative'
        self.contract['arguments']['aim_decoding']['candidate']='relative';self.save()
        with self.assertRaisesRegex(ValueError,'Combined'):self.evaluate()

    def test_only_actual_variant_requires_declaration(self):
        self.contract['arguments'].pop('mirror_augment')
        result=self.evaluate(declared=('aim_decoding',))
        self.assertEqual(result['declared_values'],{'aim_decoding':{'candidate':'relative','paired':'absolute'}})

    def test_malformed_contracts_rejected(self):
        import copy
        for key,value in [('schema',True),('mode','resume'),('arguments',[]),('extra','unreviewed')]:
            bad=copy.deepcopy(self.contract);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.evaluate(source_contract=bad)
        bad=copy.deepcopy(self.contract);bad['arguments']['mirror_augment']['candidate']=0
        with self.assertRaises(ValueError):self.evaluate(source_contract=bad)
        for name,digest in [('../outside','a'*64),('/absolute','a'*64),('train.py','bad')]:
            bad=copy.deepcopy(self.contract)
            # Matching malicious provenance still cannot make an unsafe map valid.
            bad['candidate']['files']={name:digest}
            old=self.manifest['source_provenance']['files']
            self.manifest['source_provenance']['files']={name:digest};self.save()
            with self.assertRaises(ValueError):self.evaluate(source_contract=bad)
            self.manifest['source_provenance']['files']=old
        self.save()

    def test_reward_and_object_and_initial_checks_remain(self):
        for field in ('reward_scale','timeout_bootstrap','discount'):
            self.manifest['arguments'][field]='changed';self.save()
            with self.assertRaisesRegex(ValueError,'beyond declared'):self.evaluate()
            del self.manifest['arguments'][field]
        self.save()
        for path in (self.a/'weights.npz',self.b/'initial.npz',self.opp):
            old=path.read_bytes();path.write_bytes(b'changed')
            with self.assertRaises(ValueError):self.evaluate()
            path.write_bytes(old)
        self.manifest['arguments']['resume_state']='other-source.pt';self.save()
        with self.assertRaisesRegex(ValueError,'restart'):self.evaluate()

if __name__=='__main__':unittest.main()
