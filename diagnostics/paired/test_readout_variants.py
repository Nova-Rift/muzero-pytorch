import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import readout_variants as v

H=lambda x:hashlib.sha256(x.encode()).hexdigest()

def write(p,o):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o))

class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.base=self.root/'base';self.train=self.root/'EGO2048'
        files={k:H(k) for k in ('tournament/run.py','tournament/controllers.py','tournament/lana_batch.py','arena/train_loss.py')}
        self.source=dict(commit='a'*40,dirty=False,files=files)
        m=dict(source_commit='b'*40,source_provenance=dict(commit='b'*40,dirty=False,files={'old.py':H('old')}),
               parameters=7,config=dict(discount=1.,simulations=32),environment={'env.ts':H('env')},torch='pinned',python='pinned',
               arguments=dict(games=2048,updates_per_game=1,seed=52731415,batch=256,loss_contract='baseline',resume_state=None,max_seconds=650,timeout_bootstrap='zero'))
        c=copy.deepcopy(m);c.update(source_commit=self.source['commit'],source_provenance=self.source)
        c['arguments'].update(aim_decoding='relative',mirror_augment=False,max_seconds=850)
        self.contract=dict(schema=1,mode='restart',candidate={k:self.source[k] for k in ('commit','files')},paired=dict(commit=m['source_commit'],files=m['source_provenance']['files']),arguments={'aim_decoding':dict(candidate='relative',paired='absolute')})
        for d,manifest in ((self.base,m),(self.train,c)):
            write(d/'manifest.json',manifest);(d/'weights.npz').write_bytes(str(d.name).encode());(d/'initial.npz').write_bytes(b'initial')
            write(d/'report.json',dict(games=2048,optimizer_updates=2048,requested_run_completed=True,checkpoint_sha256=v.digest(d/'weights.npz')))
        self.opponents={}
        for name,d,manifest,wins in [('Baseline2048',self.base,m,20),('EGO2048',self.train,c,25)]:
            for o,seed in v.SEEDS.items():
                models=[dict(id=name,family='independent_muzero',sha256=v.digest(d/'weights.npz'),config=manifest['config'],root_candidates=8,trained=True),dict(id=o,sha256=v.OPP_SHA.get(o,v.digest(self.base/'initial.npz')),config={'discount':.99})]
                if name=='EGO2048':models[0]['aim_decoding']='relative'
                self.opponents[o]=models[1]
                cat={'models':models}
                if name=='EGO2048':cat.update(v.declared_record(c,m,('aim_decoding',)),source_contract=self.contract)
                folder=self.root/f'{name}-{o}';write(folder/'catalog.json',cat)
                p=dict(models=models,catalog_sha256=v.digest(folder/'catalog.json'),seed=seed,sampling=v.SAMPLING,games=64,limit=256,environment=m['environment'],runtime={'torch':'pinned'},source_sha256=files['tournament/run.py'],controller_sha256=files['tournament/controllers.py'],lana_batch_adapter_sha256=files['tournament/lana_batch.py'])
                write(folder/'results/protocol.json',p)
                records=[dict(seed=seed+i,seat_a=i%2,winner=i%2 if i<wins else 1-i%2,finished=True,steps=80) for i in range(64)]
                write(folder/f'results/{name}__{o}-0.json',{'records':records})
                write(folder/'results/summary.json',[dict(a=name,b=o,games=64,wins_a=wins,wins_b=64-wins,draws=0,cutoffs=0)])
    def tearDown(self):self.tmp.cleanup()
    def analyze(self):return v.analyze(self.root,'EGO2048',self.train,self.root,self.base,self.contract,self.source,self.opponents)
    def mutate(self,path,fn):
        o=v.load(path);fn(o);write(path,o)
    def test_valid(self):
        r=self.analyze();self.assertTrue(r['gate']['passed']);self.assertEqual(r['per_opponent']['E1024']['wins_minus_baseline'],5)
    def test_mirror_valid(self):
        old=self.train;new=self.root/'MIR2048';old.rename(new);self.train=new
        m=v.load(new/'manifest.json');m['arguments'].update(aim_decoding='absolute',mirror_augment=True);write(new/'manifest.json',m)
        self.contract['arguments']={'mirror_augment':dict(candidate=True,paired=False)}
        for o in v.SEEDS:
            old=self.root/f'EGO2048-{o}';folder=self.root/f'MIR2048-{o}';old.rename(folder)
            cat=v.load(folder/'catalog.json');cat['models'][0]['id']='MIR2048';cat['models'][0].pop('aim_decoding')
            cat.update(v.declared_record(m,v.load(self.base/'manifest.json'),('mirror_augment',)),source_contract=self.contract);write(folder/'catalog.json',cat)
            pr=v.load(folder/'results/protocol.json');pr['models']=cat['models'];pr['catalog_sha256']=v.digest(folder/'catalog.json');write(folder/'results/protocol.json',pr)
            (folder/f'results/EGO2048__{o}-0.json').rename(folder/f'results/MIR2048__{o}-0.json')
            su=v.load(folder/'results/summary.json');su[0]['a']='MIR2048';write(folder/'results/summary.json',su)
        self.assertTrue(v.analyze(self.root,'MIR2048',new,self.root,self.base,self.contract,self.source,self.opponents)['gate']['passed'])
    def test_negative_controls(self):
        cases=[('source', 'manifest.json',lambda m:m.update(source_commit='c'*40)),
               ('recipe','manifest.json',lambda m:m['arguments'].update(batch=128)),
               ('seed','manifest.json',lambda m:m['arguments'].update(seed=61837209)),
               ('count','report.json',lambda m:m.update(games=2047)),
               ('timeout','report.json',lambda m:m.update(requested_run_completed=False)),
               ('weights','report.json',lambda m:m.update(checkpoint_sha256='0'*64)),
               ('flags','manifest.json',lambda m:m['arguments'].update(mirror_augment=True)),
               ('aim','manifest.json',lambda m:m['arguments'].update(aim_decoding='absolute'))]
        for label,file,fn in cases:
            with self.subTest(label=label):
                p=self.train/file;old=p.read_bytes();self.mutate(p,fn)
                with self.assertRaises((ValueError,KeyError)):self.analyze()
                p.write_bytes(old)
    def test_bout_controls(self):
        folder=self.root/'EGO2048-E1024/results'
        cases=[('seed','protocol.json',lambda p:p.update(seed=1)),('limit','protocol.json',lambda p:p.update(limit=255)),
               ('source','protocol.json',lambda p:p.update(source_sha256='0'*64)),
               ('cataloghash','protocol.json',lambda p:p.update(catalog_sha256='0'*64)),
               ('partial','EGO2048__E1024-0.json',lambda p:p['records'].pop()),
               ('seat','EGO2048__E1024-0.json',lambda p:p['records'][0].update(seat_a=1)),
               ('summary','summary.json',lambda p:p[0].update(wins_a=26))]
        for label,file,fn in cases:
            with self.subTest(label=label):
                p=folder/file;old=p.read_bytes();self.mutate(p,fn)
                with self.assertRaises((ValueError,KeyError)):self.analyze()
                p.write_bytes(old)
    def test_declared_contract(self):
        self.contract['arguments']['aim_decoding']['candidate']='absolute'
        with self.assertRaises(ValueError):self.analyze()
    def test_cutoff_gate(self):
        p={o:dict(arm=dict(wins=21,cutoffs=13),baseline=dict(wins=20)) for o in ('E1024','F1024')}
        self.assertFalse(v.gate(p)['passed'])
        p['E1024']['arm'].update(cutoffs=12,wins=20);p['F1024']['arm']['cutoffs']=12
        self.assertFalse(v.gate(p)['passed'])

if __name__=='__main__':unittest.main()
