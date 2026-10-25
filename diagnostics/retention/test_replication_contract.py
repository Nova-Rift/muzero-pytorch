import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import replication_contract as rc


class ReplicationContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.a=self.root/'candidate';self.b=self.root/'paired'
        self.files={'arena/train_loss.py':'a'*64}
        # Fixture source map only: production code separately requires full pinned map SHA.
        h=hashlib.sha256(json.dumps(self.files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        patcher=patch.object(rc,'SOURCE_FILES_SHA256',h);patcher.start();self.addCleanup(patcher.stop)
        self.source={'commit':rc.SOURCE_COMMIT,'files':self.files,'dirty':False}
        self.contract=dict(schema=1,mode=rc.MODE,variant='EGO',training_seed=rc.TRAINING_SEED,
                           source={k:self.source[k] for k in ('commit','files')},
                           initial_sha256=hashlib.sha256(b'new initial').hexdigest(),
                           recipe=copy.deepcopy(rc.RECIPE),config=copy.deepcopy(rc.CONFIG),
                           environment=copy.deepcopy(rc.ENVIRONMENT))
        self.m={};self.r={}
        for side,p in (('candidate',self.a),('paired',self.b)):
            p.mkdir();(p/'initial.npz').write_bytes(b'new initial');(p/'weights.npz').write_bytes(side.encode())
            self.m[side]=dict(arguments={**rc.RECIPE,**(rc.VARIANTS['EGO'] if side=='candidate' else {}),
                                        'site':'/site','output':str(p),'max_seconds':900.0},
                              config=copy.deepcopy(rc.CONFIG),environment=copy.deepcopy(rc.ENVIRONMENT),
                              source_commit=rc.SOURCE_COMMIT,source_provenance=copy.deepcopy(self.source),
                              torch='2.8.0+cu128',python='3.12.3',parameters=3226691)
            self.r[side]=dict(requested_run_completed=True,games=2048,optimizer_updates=2048,
                              training_examples=524288,checkpoint_sha256=rc.digest(p/'weights.npz'))
        self.save()

    def save(self):
        for side,p in (('candidate',self.a),('paired',self.b)):
            (p/'manifest.json').write_text(json.dumps(self.m[side]));(p/'report.json').write_text(json.dumps(self.r[side]))

    def check(self):
        self.save();return rc.validate_pair(self.a,self.b,self.contract,self.source)

    def test_ego_and_mir(self):
        self.check();self.contract['variant']='MIR'
        self.m['candidate']['arguments'].update(aim_decoding='absolute',mirror_augment=True)
        self.check()

    def test_all_unrelated_arguments_rejected_even_when_both_sides_change(self):
        for key,value in rc.RECIPE.items():
            if key in ('aim_decoding','mirror_augment'):continue
            old=copy.deepcopy(self.m)
            for side in self.m:self.m[side]['arguments'][key]=str(value)+'changed'
            with self.subTest(key=key),self.assertRaises(ValueError):self.check()
            self.m=old
        for side in self.m:self.m[side]['arguments']['new_unreviewed_flag']=1
        with self.assertRaises(ValueError):self.check()

    def test_no_unapproved_flags_or_combination(self):
        for change in ({'mirror_augment':True},{'aim_decoding':'absolute'},{'mirror_augment':1}):
            old=copy.deepcopy(self.m);self.m['candidate']['arguments'].update(change)
            with self.subTest(change=change),self.assertRaises(ValueError):self.check()
            self.m=old
        self.m['paired']['arguments']['aim_decoding']='relative'
        with self.assertRaises(ValueError):self.check()

    def test_schema_seed_source_map_and_initial_binding(self):
        changes=[('schema',True),('mode','restart'),('variant','EGO+MIR'),('training_seed',52731415),
                 ('training_seed',61837209),('initial_sha256',None),('initial_sha256','0'*64),
                 ('source',{'commit':'f'*40,'files':self.files}),
                 ('source',{'commit':rc.SOURCE_COMMIT,'files':{'other.py':'b'*64}}),
                 ('recipe',{**rc.RECIPE,'discount':.99}),('environment',{}),('config',{})]
        for k,v in changes:
            old=copy.deepcopy(self.contract);self.contract[k]=v
            with self.subTest(key=k,value=v),self.assertRaises(ValueError):self.check()
            self.contract=old

    def test_both_endpoint_counts_source_and_runtime(self):
        for side in self.m:
            for key,value in [('games',2047),('optimizer_updates',2048.0),('training_examples',524287),
                              ('requested_run_completed',1),('checkpoint_sha256','f'*64)]:
                old=copy.deepcopy(self.r);self.r[side][key]=value
                with self.subTest(side=side,key=key),self.assertRaises(ValueError):self.check()
                self.r=old
            for key,value in [('source_commit','f'*40),('torch','2.9'),('config',{}),('environment',{}),
                              ('parameters',3226691.0),('python','3.13')]:
                old=copy.deepcopy(self.m);self.m[side][key]=value
                with self.subTest(side=side,key=key),self.assertRaises(ValueError):self.check()
                self.m=old
            self.m[side]['source_provenance']['dirty']=True
            with self.assertRaises(ValueError):self.check()
            self.m[side]['source_provenance']['dirty']=False

    def test_no_self_pair_or_matching_wrong_initial(self):
        with self.assertRaises(ValueError):rc.validate_pair(self.a,self.a,self.contract,self.source)
        for p in (self.a,self.b):(p/'initial.npz').write_bytes(b'matching but wrong')
        with self.assertRaises(ValueError):self.check()

    def test_source_receipt_cannot_drop_hashes(self):
        self.source['files']={}
        with self.assertRaises(ValueError):self.check()

    def test_historical_validator_still_rejects_new_seed(self):
        from evaluate_endpoint import validate_source_contract
        contract={'schema':1,'mode':'restart','candidate':self.contract['source'],'paired':self.contract['source'],
                  'arguments':{'aim_decoding':{'candidate':'relative','paired':'absolute'}}}
        with self.assertRaises(ValueError):validate_source_contract(self.m['candidate'],self.m['paired'],('aim_decoding',),contract)


if __name__=='__main__':unittest.main()
