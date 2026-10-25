"""Small synthetic production-shape fixtures; never production evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from compressed_checkpoint import compress
from verify_variant_recovery import run, expected_arguments, PIN, COUNTS, digest

class VariantRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name);self.root=self.base/'run';self.root.mkdir();(self.root/'compressed').mkdir()
        self.source=self.base/'source';self.source.mkdir();(self.source/'trainer.py').write_text('fixture only\n')
        self.prov=self.source/'.source-provenance.json'
        self.provenance=dict(commit=PIN,dirty=False,files={'trainer.py':digest(self.source/'trainer.py')})
        self.prov.write_text(json.dumps(self.provenance))
        self.baseline=self.base/'baseline';self.baseline.mkdir()
        self.contract=self.base/'contract.json'
        self.arm='ego';self.build()
    def write(self,p,d):p.write_text(json.dumps(d))
    def build(self):
        args=expected_arguments(self.arm)
        oldargs=copy.deepcopy(args);oldargs.update(aim_decoding='absolute',mirror_augment=False)
        original=dict(arguments=oldargs,config={'fixture':True},environment={'fixture':'sha'},torch='fixture',python='fixture',gpu='fixture',source_commit='a'*40,source_provenance=dict(commit='a'*40,dirty=False,files={'old.py':'b'*64}))
        self.manifest=dict(original,arguments=args,source_commit=PIN,source_provenance=self.provenance)
        self.write(self.root/'manifest.json',self.manifest);self.write(self.baseline/'manifest.json',original)
        key='aim_decoding' if self.arm=='ego' else 'mirror_augment'
        contract=dict(schema=1,mode='restart',candidate=dict(commit=PIN,files=self.provenance['files']),paired=dict(commit=original['source_commit'],files=original['source_provenance']['files']),arguments={key:dict(candidate=args[key],paired=oldargs[key])})
        self.write(self.contract,contract)
        np.savez(self.root/'initial.npz',w=np.array([1.],np.float32));(self.baseline/'initial.npz').write_bytes((self.root/'initial.npz').read_bytes())
        np.savez(self.root/'weights.npz',w=np.array([2.],np.float32))
        self.write(self.root/'variant-collection-gate.json',dict(complete=True,checked_before_first_update=True,variant=self.arm,source_contract=contract,initial_sha256=digest(self.root/'initial.npz')))
        self.write(self.root/'report.json',dict(requested_run_completed=True,games=2048,optimizer_updates=2048,training_examples=2048*256,checkpoint_sha256=digest(self.root/'weights.npz')))
        self.states={};records={}
        for n in COUNTS:
            state=dict(arguments=args,environment=self.manifest['environment'],source_files=self.provenance['files'],torch_version='fixture',initial={'w':torch.tensor([1.])},model={'w':torch.tensor([2.])},optimizer={'state':{0:{'step':torch.tensor(float(n)),'exp_avg':torch.tensor([.1])}},'param_groups':[{'params':[0]}]},totals=dict(games=n,optimizer_updates=n,training_examples=n*256),replay=[{'x':torch.tensor([1.])} for _ in range(3)],numpy_rng=np.random.default_rng(5).bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=[torch.tensor([1,2],dtype=torch.uint8)])
            self.states[n]=state;records.update(self.encode(n))
        self.write(self.root/'compressed/manifest.json',dict(states=records,encoded_bytes=sum(v['compressed_bytes'] for v in records.values())))
    def encode(self,n):
        name=f'games-{n}-updates-{n}.pt';raw=self.base/name;torch.save(self.states[n],raw)
        dest=self.root/'compressed'/(name+'.gz');dest.unlink(missing_ok=True)
        r=compress(raw,dest,digest(raw));r.update(raw_name=name,saved_at_unix=1700000000+n);raw.unlink();return {name:r}
    def verify(self):return run(self.root,self.source,self.prov,self.baseline,self.contract,self.arm)
    def test_complete_both_arms(self):
        for arm in ('ego','mir'):
            self.arm=arm;self.build();r=self.verify();self.assertTrue(r['verified']);self.assertEqual(len(r['states']),4);self.assertEqual(len(r['evidence']),10)
    def test_missing_state(self):
        next((self.root/'compressed').glob('*.gz')).unlink()
        with self.assertRaises(FileNotFoundError):self.verify()
    def test_corrupt_state(self):
        p=next((self.root/'compressed').glob('*.gz'));p.write_bytes(p.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError,'Compressed'):self.verify()
    def test_changed_source(self):
        (self.source/'trainer.py').write_text('changed')
        with self.assertRaisesRegex(ValueError,'source changed'):self.verify()
    def test_changed_flags_consistently_rejected(self):
        for key,value in [('aim_decoding','absolute'),('mirror_augment',True),('cuda_graph',False),('tf32',False),('device_history',False),('loss_contract','root5-mass-v1'),('resume_state','prior.pt'),('seed',1)]:
            self.build();self.manifest['arguments'][key]=value;self.write(self.root/'manifest.json',self.manifest)
            with self.assertRaisesRegex(ValueError,'argument/flag'):self.verify()
    def test_missing_date(self):
        p=self.root/'compressed/manifest.json';d=json.loads(p.read_text());next(iter(d['states'].values())).pop('saved_at_unix');self.write(p,d)
        with self.assertRaisesRegex(ValueError,'completion times'):self.verify()
    def test_missing_gate(self):
        (self.root/'variant-collection-gate.json').unlink()
        with self.assertRaises(FileNotFoundError):self.verify()
    def test_final_weight_parity(self):
        np.savez(self.root/'weights.npz',w=np.array([3.],np.float32))
        with self.assertRaisesRegex(ValueError,'checkpoint hash'):self.verify()
    def test_wrong_pin(self):
        self.provenance['commit']='0'*40;self.write(self.prov,self.provenance)
        with self.assertRaisesRegex(ValueError,'source pin'):self.verify()
    def test_oversize_decode(self):
        p=self.root/'compressed/manifest.json';d=json.loads(p.read_text());next(iter(d['states'].values()))['raw_bytes']=1250000001;self.write(p,d)
        with self.assertRaisesRegex(ValueError,'scratch'):self.verify()

if __name__=='__main__':unittest.main()
