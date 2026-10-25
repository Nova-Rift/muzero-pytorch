"""Real compression/restore of synthetic CPU states; no CUDA execution claim."""
import copy
import hashlib
import io
import json
from pathlib import Path
import unittest
import numpy as np
import torch
from compressed_checkpoint import compress
import replication_collection as c
import replication_contract as rc
import replication_recovery as rr
import test_replication_collection as fixture


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        f=self.f=fixture.CollectionTests();f.setUp();self.addCleanup(f.doCleanups)
        out=io.BytesIO();np.savez_compressed(out,w=np.array([1.],np.float32));f.initial_payload=out.getvalue()
        f.contract['initial_sha256']=hashlib.sha256(f.initial_payload).hexdigest()
        self.ref=f.make_reference('ref');self.root=f.root/'train'
        t,_=f.install(self.root,mode='train',own=self.ref);f.call(t)
        self.manifest=json.loads((self.root/'manifest.json').read_text())
        self.source_path=f.root/'source.json';c.write(self.source_path,f.source)
        np.savez_compressed(self.root/'weights.npz',w=np.array([2.],np.float32))
        c.write(self.root/'report.json',dict(requested_run_completed=True,games=2048,optimizer_updates=2048,
            training_examples=2048*256,checkpoint_sha256=rc.digest(self.root/'weights.npz')))
        (self.root/'compressed').mkdir()
        self.state=dict(arguments=self.manifest['arguments'],environment=rc.ENVIRONMENT,source_files=f.files,
            torch_version='2.8.0+cu128',initial={'w':torch.tensor([1.])},model={'w':torch.tensor([2.])},
            optimizer={'state':{0:{'step':torch.tensor(2048.),'exp_avg':torch.tensor([0.1])}},'param_groups':[{'params':[0]}]},
            totals={},replay=[{'observation':torch.tensor([1.])} for _ in range(3)],
            numpy_rng=np.random.default_rng(rc.TRAINING_SEED).bit_generator.state,torch_rng=torch.get_rng_state(),
            cuda_rng=[torch.tensor([1,2],dtype=torch.uint8)])
        self.encode()

    def encode(self, mutate=None):
        states={}
        for n in (512,1024,1536,2048):
            state=copy.deepcopy(self.state);state['totals']=dict(games=n,optimizer_updates=n,training_examples=n*256)
            if mutate:mutate(state,n)
            name=f'games-{n}-updates-{n}.pt';raw=self.f.root/name;torch.save(state,raw)
            target=self.root/'compressed'/(name+'.gz');target.unlink(missing_ok=True)
            record=compress(raw,target,rc.digest(raw));record['raw_name']=name;states[name]=record;raw.unlink()
        c.write(self.root/'compressed/manifest.json',dict(states=states,encoded_bytes=sum(x['compressed_bytes'] for x in states.values())))

    def check(self):
        return rr.verify(self.root,self.source_path,self.f.contract,self.f.source,'paired',self.f.initialization,
            self.f.bound,rc.digest(self.root/c.RECEIPT),rc.digest(self.root/'compressed/manifest.json'))

    def test_all_four_seed73142017_states_decode_and_bind(self):
        result=self.check();self.assertTrue(result['verified']);self.assertEqual(result['training_seed'],73142017)
        self.assertEqual([x['games'] for x in result['states']],[512,1024,1536,2048])
        self.assertFalse(list(self.root.glob('.verify-state-*')))
        self.assertEqual(result['expected_loss_contract'],'baseline')

    def test_saved_historical_seed_rejected_even_valid_compression(self):
        self.encode(lambda s,n:s['arguments'].update(seed=52731415))
        with self.assertRaisesRegex(ValueError,'Checkpoint arguments mismatch'):self.check()

    def test_saved_source_and_update_and_rng_mutations(self):
        for mutate in (lambda s,n:s.update(source_files={}),lambda s,n:s['totals'].update(optimizer_updates=n+1),
                       lambda s,n:s.update(cuda_rng=[]),lambda s,n:s.update(numpy_rng={}),
                       lambda s,n:s.update(replay=[]),lambda s,n:s.update(optimizer={'state':{},'param_groups':[]})):
            self.encode(mutate)
            with self.subTest(mutation=mutate),self.assertRaises(ValueError):self.check()

    def test_recovered_directory_may_move_without_rewriting_manifest(self):
        import shutil
        destination=self.f.root/'recovered';shutil.move(str(self.root),destination);self.root=destination
        self.assertTrue(self.check()['verified'])

    def test_missing_and_corrupt_chain_rejected(self):
        path=self.root/'compressed/manifest.json';m=json.loads(path.read_text());m['states'].pop('games-1024-updates-1024.pt');c.write(path,m)
        with self.assertRaisesRegex(ValueError,'milestone chain'):self.check()
        self.encode();p=self.root/'compressed/games-512-updates-512.pt.gz';p.write_bytes(p.read_bytes()+b'corrupt')
        with self.assertRaisesRegex(ValueError,'hash/size'):self.check()

    def test_guard_parity_and_recipe_mutations_precede_decode(self):
        path=self.root/c.RECEIPT;original=json.loads(path.read_text())
        for key,value in [('mode','reference'),('side','candidate'),('bindings',{}),('comparisons',{}),('files',{})]:
            m=copy.deepcopy(original);m[key]=value;c.write(path,m)
            with self.subTest(key=key),self.assertRaises(ValueError):self.check()
        c.write(path,original)
        m=copy.deepcopy(self.manifest);m['arguments']['seed']=52731415;c.write(self.root/'manifest.json',m)
        with self.assertRaisesRegex(ValueError,'recipe'):self.check()

if __name__=='__main__':unittest.main()
