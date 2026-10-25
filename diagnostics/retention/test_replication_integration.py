import copy
import hashlib
import json
from pathlib import Path
import unittest
import replication_contract as rc
import replication_initialization as init
import evaluate_replication as ev
from test_replication_contract import ReplicationContractTests


class IntegrationTests(ReplicationContractTests):
    def setUp(self):
        super().setUp()
        self.receipt=dict(schema=1,kind='no-update-initialization-v1',started_at='2026-01-01T00:00:00+00:00',
                          completed_at='2026-01-01T00:00:01+00:00',runtime=init.RUNTIME,source=self.contract['source'],
                          config=rc.CONFIG,recipe=rc.RECIPE,training_seed=rc.TRAINING_SEED,
                          counters=dict(games=0,optimizer_updates=0,training_examples=0),
                          initial_sha256=self.contract['initial_sha256'],
                          contract_sha256=hashlib.sha256((json.dumps(self.contract,indent=2)+'\n').encode()).hexdigest())
        self.opp=dict(id='OwnInitial',family='independent_muzero',root_candidates=8,trained=False,
                      checkpoint=str(self.a/'initial.npz'),sha256=self.contract['initial_sha256'],config=rc.CONFIG)

    def test_initial_artifact_and_template(self):
        init.validate_artifact(self.contract,self.receipt,self.a/'initial.npz')
        template=copy.deepcopy(self.contract);template['initial_sha256']=None
        init.validate_template(template)
        with self.assertRaises(ValueError):init.validate_template(self.contract)
        for key,value in [('runtime',{}),('training_seed',52731415),('source',{}),('config',{}),('recipe',{}),
                          ('counters',dict(games=0,optimizer_updates=1,training_examples=256)),
                          ('completed_at','2025-01-01T00:00:00+00:00'),('contract_sha256','f'*64)]:
            bad={**self.receipt,key:value}
            with self.subTest(key=key),self.assertRaises(ValueError):init.validate_artifact(self.contract,bad,self.a/'initial.npz')

    def test_catalog_both_sides_and_forbidden_opponent(self):
        for side in ('candidate','paired'):
            result=ev.catalog(self.a,self.b,side,self.opp,'development',self.contract,self.source,self.receipt,self.a/'initial.npz')
            self.assertEqual(result['models'][0].get('aim_decoding','absolute'),'relative' if side=='candidate' else 'absolute')
        for change in ({'sha256':'f'*64},{'trained':True},{'aim_decoding':'relative'},{'id':'Other'},{'config':{}}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                ev.catalog(self.a,self.b,'candidate',{**self.opp,**change},'development',self.contract,self.source,self.receipt,self.a/'initial.npz')
        with self.assertRaises(ValueError):
            ev.catalog(self.a,self.b,'candidate',self.opp,'confirmation',self.contract,self.source,self.receipt,self.a/'initial.npz')

if __name__=='__main__':unittest.main()
