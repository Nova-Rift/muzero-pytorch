"""Synthetic CPU fixtures test guards, never claim real collection/CUDA parity."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import replication_collection as c
import replication_contract as rc


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.files={'arena/train_loss.py':'a'*64}
        h=hashlib.sha256(json.dumps(self.files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        p=patch.object(rc,'SOURCE_FILES_SHA256',h);p.start();self.addCleanup(p.stop)
        p=patch.object(c,'bind_source');p.start();self.addCleanup(p.stop)
        self.source=dict(commit=rc.SOURCE_COMMIT,files=self.files,dirty=False)
        self.contract=dict(schema=1,mode=rc.MODE,variant='MIR',training_seed=rc.TRAINING_SEED,
            source={k:self.source[k] for k in ('commit','files')},initial_sha256=hashlib.sha256(b'initial').hexdigest(),
            recipe=copy.deepcopy(rc.RECIPE),config=copy.deepcopy(rc.CONFIG),environment=copy.deepcopy(rc.ENVIRONMENT))
        self.initialization={'gpu':'fixture GPU'};self.bound={'contract_sha256':'fixture','initialization_review_completed_at':'2026-01-01T00:00:00+00:00'}
        self.block=dict(observation=np.zeros((2,1408,64),np.float32),actions=np.zeros((1,1408),np.int64),
            rewards=np.zeros((1,1408),np.float32),policies=np.zeros((2,1408,153),np.float32),
            targets=np.zeros((2,1408),np.float32),valid=np.zeros((2,1408),bool),absorbing=np.zeros((2,1408),bool))
        self.block['valid'][0]=True;self.block['indices']=np.argwhere(self.block['valid'][:1])
        self.stats=dict(games=128,collection_steps=1,natural_terminal=128,cutoffs=0,active_transitions=1408,
            action_counts=[1408]+[0]*152,search=1.,environment=1.,history=1.,
            padded_search_roots=256,hits=0,movement_pixels=0.,completed_search_return_mse=None,joint_steps=128)
        self.calls=0

    def manifest(self,path,side):
        path.mkdir();(path/'initial.npz').write_bytes(getattr(self,'initial_payload',b'initial'))
        m=dict(arguments={**rc.RECIPE,**(rc.VARIANTS[self.contract['variant']] if side=='candidate' else {}),
            'site':'/site','output':str(path),'max_seconds':850.},source_commit=rc.SOURCE_COMMIT,
            source_provenance=self.source,config=rc.CONFIG,environment=rc.ENVIRONMENT,**c.init.RUNTIME,gpu='fixture GPU')
        c.write(path/'manifest.json',m);return m

    def install(self,path,side='paired',mode='reference',own=None,paired=None):
        def collect(pool,model,seeds,limit,context,searcher,device_history=False,return_mode='nstep',timeout_bootstrap='continue',aim_decoding='absolute'):
            self.calls+=1;return copy.deepcopy(self.block),copy.deepcopy(self.stats)
        trainer=SimpleNamespace(collect=collect)
        checked=c.install_gate(trainer,self.root,path,self.contract,self.source,side,self.initialization,self.bound,mode,
            own,rc.digest(own/c.RECEIPT) if own else None,paired,rc.digest(paired/c.RECEIPT) if paired else None)
        self.manifest(path,side);return trainer,checked

    def call(self,trainer,side='paired',seed=rc.TRAINING_SEED):
        return trainer.collect(None,None,range(seed,seed+128),256,64,None,True,'episode','zero',
            'relative' if side=='candidate' and self.contract['variant']=='EGO' else 'absolute')

    def make_reference(self,name,side='paired',paired=None):
        path=self.root/name;t,checked=self.install(path,side,paired=paired)
        with self.assertRaises(c.CollectionComplete):self.call(t,side)
        self.assertEqual(len(checked),1);return path

    def test_reference_exits_before_update_and_full_block_train_passes(self):
        ref=self.make_reference('reference');out=self.root/'train'
        t,checked=self.install(out,mode='train',own=ref)
        self.call(t);self.assertEqual(checked[0]['comparisons']['own']['full_128_trajectory_parity'],True)
        self.assertEqual(checked[0]['optimizer_updates'],0)
        self.call(t);self.assertEqual(self.calls,3)

    def test_wrong_export_fails_before_actual_collection(self):
        t,_=self.install(self.root/'bad');(self.root/'bad/initial.npz').write_bytes(b'wrong')
        with self.assertRaisesRegex(ValueError,'initial export'):self.call(t)
        self.assertEqual(self.calls,0)

    def test_every_recipe_mutation_rejected_before_collection(self):
        t,_=self.install(self.root/'bad');path=self.root/'bad/manifest.json';original=json.loads(path.read_text())
        for key in rc.RECIPE:
            m=copy.deepcopy(original);m['arguments'][key]='wrong';c.write(path,m)
            with self.subTest(key=key),self.assertRaises(ValueError):self.call(t)
        self.assertEqual(self.calls,0)

    def test_actual_seed_and_call_flags_rejected(self):
        t,_=self.install(self.root/'bad')
        with self.assertRaises(ValueError):self.call(t,seed=52731415)
        with self.assertRaises(ValueError):t.collect(None,None,range(rc.TRAINING_SEED,rc.TRAINING_SEED+128),256,64,None,False,'episode','zero','absolute')
        self.assertEqual(self.calls,0)

    def test_full128_required_not_two_game_trace(self):
        for key in ('observation','policies','actions'):
            block=copy.deepcopy(self.block);block[key]=block[key][:,:22]
            with self.subTest(key=key),self.assertRaises(ValueError):c.validate_block(block,self.stats)
        bad=copy.deepcopy(self.block);bad['indices']=bad['indices'][1:]
        with self.assertRaises(ValueError):c.validate_block(bad,self.stats)

    def test_nonfinite_dtype_and_signed_zero_mutations(self):
        for key in ('rewards','policies'):
            bad=copy.deepcopy(self.block);bad[key].flat[0]=float('nan')
            with self.assertRaises(ValueError):c.validate_block(bad,self.stats)
        bad=copy.deepcopy(self.block);bad['rewards']=bad['rewards'].astype(np.float64)
        with self.assertRaises(ValueError):c.validate_block(bad,self.stats)
        bad=copy.deepcopy(self.block);bad['rewards'].flat[0]=-0.
        with self.assertRaises(ValueError):c.equal_blocks(bad,self.block)

    def test_mir_paired_exact_and_difference_rejected(self):
        paired=self.make_reference('paired');self.make_reference('mir','candidate',paired)
        self.block['actions'][0,1000]=1
        t,_=self.install(self.root/'bad','candidate',paired=paired)
        with self.assertRaisesRegex(ValueError,'Full128'):self.call(t,'candidate')
        self.assertFalse((self.root/'bad'/c.RECEIPT).exists())

    def test_ego_records_difference_but_requires_own_replay(self):
        self.contract['variant']='EGO';paired=self.make_reference('paired')
        self.block['actions'][0,1000]=1;own=self.make_reference('ego','candidate',paired)
        r=json.loads((own/c.RECEIPT).read_text());self.assertNotIn('paired',r['comparisons'])
        self.assertEqual(r['paired_parity'],'inapplicable-relative-aim-changes-policy')
        t,_=self.install(self.root/'train','candidate','train',own,paired);self.call(t,'candidate')
        self.block['actions'][0,1000]=2
        t,_=self.install(self.root/'bad','candidate','train',own,paired)
        with self.assertRaises(ValueError):self.call(t,'candidate')

    def test_reference_mutations_rejected(self):
        ref=self.make_reference('reference');path=ref/c.RECEIPT;original=json.loads(path.read_text())
        for key,value in [('optimizer_updates',1),('mode','train'),('side','candidate'),('complete',False),
                          ('guard_sha256','0'*64),('bindings',{}),('files',{})]:
            r=copy.deepcopy(original);r[key]=value;c.write(path,r)
            with self.subTest(key=key),self.assertRaises(ValueError):self.install(self.root/'train',mode='train',own=ref)
        c.write(path,original);(ref/'full-block.npz').write_bytes(b'changed')
        with self.assertRaises(ValueError):self.install(self.root/'train',mode='train',own=ref)

    def test_command_contains_frozen_recipe_and_only_selected_intervention(self):
        command=c.trainer_command(self.contract,'candidate','/site','/out',850)
        self.assertIn('--mirror-augment',command);self.assertIn(str(rc.TRAINING_SEED),command)
        self.assertNotIn('--resume-state',command);self.assertNotIn('--mirror-augment',c.trainer_command(self.contract,'paired','/site','/out',850))
        self.contract['variant']='EGO';command=c.trainer_command(self.contract,'candidate','/site','/out',850)
        self.assertEqual(command[command.index('--aim-decoding')+1],'relative');self.assertNotIn('--mirror-augment',command)


class InitializationBindingTests(unittest.TestCase):
    def setUp(self):
        f=self.f=CollectionTests();f.setUp();self.addCleanup(f.doCleanups)
        self.directory=f.root/'initialization';self.directory.mkdir()
        (self.directory/'initial.npz').write_bytes(b'initial')
        c.write(self.directory/'contract.json',f.contract)
        self.receipt=dict(schema=1,kind='no-update-initialization-v1',runtime=c.init.RUNTIME,
            source=f.contract['source'],config=rc.CONFIG,recipe=rc.RECIPE,training_seed=rc.TRAINING_SEED,
            counters=dict(games=0,optimizer_updates=0,training_examples=0),initial_sha256=f.contract['initial_sha256'],
            contract_sha256=rc.digest(self.directory/'contract.json'),started_at='2026-01-01T00:00:00+00:00',
            completed_at='2026-01-01T00:00:01+00:00',generator_sha256=rc.digest(Path(c.init.__file__)),gpu='fixture GPU')
        self.parity=dict(schema=1,kind='frozen-trainer-initial-prefix-parity-v1',device='cuda',
            trainer_main_invoked=False,actual_trainer_export_verified=False,tensor_count=20,
            source_commit=rc.SOURCE_COMMIT,generator_sha256=self.receipt['generator_sha256'],
            trainer_sha256=f.files['arena/train_loss.py'],initial_sha256=f.contract['initial_sha256'],
            torch=c.init.RUNTIME['torch'],python=c.init.RUNTIME['python'],
            tensor_bytes_equal=True,npz_bytes_equal=True,rng_states_equal=True)
        self.review_path=f.root/'review.json'

    def save(self,review_changes=None):
        c.write(self.directory/'initialization.json',self.receipt);c.write(self.directory/'initial-parity.json',self.parity)
        self.review=dict(schema=1,kind='replication-initialization-review-v1',accepted=True,
            contract_sha256=rc.digest(self.directory/'contract.json'),initialization_sha256=rc.digest(self.directory/'initialization.json'),
            parity_sha256=rc.digest(self.directory/'initial-parity.json'),completed_at='2026-01-01T00:00:02+00:00')
        self.review.update(review_changes or {});c.write(self.review_path,self.review)

    def check(self):
        return c.bindings(self.directory,rc.digest(self.directory/'contract.json'),rc.digest(self.directory/'initialization.json'),
            self.review_path,rc.digest(self.review_path),self.f.source)

    def test_bound_review_accepts_only_completed_cuda_prefix(self):
        self.save();contract,receipt,bound=self.check();self.assertEqual(contract,self.f.contract)
        self.assertEqual(bound['initialization_review_sha256'],rc.digest(self.review_path))

    def test_cpu_wrong_source_or_generator_and_rng_parity_rejected(self):
        original=copy.deepcopy(self.parity)
        for key,value in [('device','cpu'),('trainer_sha256','f'*64),('generator_sha256','f'*64),
            ('initial_sha256','f'*64),('rng_states_equal',False),('tensor_count',2),('torch','2.2'),
            ('trainer_main_invoked',True),('actual_trainer_export_verified',True)]:
            self.parity=copy.deepcopy(original);self.parity[key]=value;self.save()
            with self.subTest(key=key),self.assertRaises(ValueError):self.check()

    def test_unreviewed_and_pre_initialization_and_future_review_rejected(self):
        for change in ({'accepted':False},{'contract_sha256':'f'*64},{'parity_sha256':'f'*64},
            {'completed_at':'2025-01-01T00:00:00+00:00'},{'completed_at':'2099-01-01T00:00:00+00:00'},
            {'extra_unreviewed_field':True}):
            self.save(change)
            with self.subTest(change=change),self.assertRaises(ValueError):self.check()

    def test_hash_pin_and_initial_export_mutations_rejected(self):
        self.save();(self.directory/'initial.npz').write_bytes(b'changed')
        with self.assertRaises(ValueError):self.check()
        with self.assertRaises(ValueError):c.pinned(self.review_path,'0'*64)

if __name__=='__main__':unittest.main()
