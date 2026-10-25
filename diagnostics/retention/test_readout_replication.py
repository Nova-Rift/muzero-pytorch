"""Synthetic CPU fixtures only: never evidence of CUDA execution or playing strength."""
import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import replication_contract as rc
import replication_initialization as init
import evaluate_replication as ev
import readout_replication as ro
from test_replication_contract import ReplicationContractTests


def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def binding(path):
    return dict(path=str(path),sha256=rc.digest(path))


class ReadoutTests(ReplicationContractTests):
    def setUp(self):
        super().setUp()
        # A tiny explicitly patched source fixture, not the production source map.
        self.files.update({'tournament/'+f:'b'*64 for f in ('run.py','controllers.py','lana_batch.py')})
        h=hashlib.sha256(json.dumps(self.files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        p=patch.object(rc,'SOURCE_FILES_SHA256',h);p.start();self.addCleanup(p.stop)
        self.contract['source']['files']=self.files
        for side in self.m:
            self.m[side]['gpu']='synthetic GPU'
            self.m[side]['source_provenance']['files']=self.files
        self.save()
        self.request=dict(schema=1,phase='development',candidate=str(self.a),paired=str(self.b))
        template=copy.deepcopy(self.contract);template['initial_sha256']=None
        for name,value in [('template',template),('contract',self.contract),('source_receipt',self.source),
                            ('evaluation_environment',dict(gpu='synthetic GPU',legacy_sources={'scripts/example.py':'c'*64}))]:
            path=self.root/(name+'.json');write(path,value);self.request[name]=binding(path)
        self.initialization=dict(schema=1,kind='no-update-initialization-v1',started_at='2026-01-01T00:00:00+00:00',
            completed_at='2026-01-01T00:00:01+00:00',runtime=init.RUNTIME,source=self.contract['source'],
            config=rc.CONFIG,recipe=rc.RECIPE,training_seed=rc.TRAINING_SEED,
            counters=dict(games=0,optimizer_updates=0,training_examples=0),
            initial_sha256=self.contract['initial_sha256'],contract_sha256=self.request['contract']['sha256'],
            template_sha256=self.request['template']['sha256'],generator_sha256=rc.digest(Path(init.__file__)),gpu='synthetic GPU')
        path=self.root/'initialization.json';write(path,self.initialization);self.request['initialization']=binding(path)
        self.request['initial']=binding(self.a/'initial.npz')
        hashes={}
        self.entries={};self.opponents={}
        for name in ('E1024','F1024','OwnInitial'):
            weight=self.a/'initial.npz' if name=='OwnInitial' else self.root/(name+'.npz')
            if name!='OwnInitial':weight.write_bytes(name.encode());hashes[name]=rc.digest(weight)
            entry=dict(id=name,family='independent_muzero',root_candidates=8,trained=name!='OwnInitial',
                       checkpoint=str(weight),sha256=rc.digest(weight),
                       config={**rc.CONFIG,'discount':1.0 if name=='OwnInitial' else .99})
            path=self.root/(name+'.json');write(path,entry)
            self.entries[name]=entry;self.opponents[name]=dict(entry=binding(path),checkpoint=binding(weight))
        p=patch.object(ev,'OPP_SHA',hashes);p.start();self.addCleanup(p.stop)
        self.make_bouts('development')

    def make_bouts(self,phase,candidate_wins=50,paired_wins=30,cutoffs=0):
        self.request['phase']=phase
        self.request['opponents']={k:self.opponents[k] for k in ev.SEEDS[phase]}
        self.request['bouts']=[]
        for side in ('candidate','paired'):
            for opponent in ev.SEEDS[phase]:
                directory=self.root/(phase+'-'+side+'-'+opponent);directory.mkdir(exist_ok=True)
                cat=ev.catalog(self.a,self.b,side,self.entries[opponent],phase,self.contract,self.source,self.initialization,self.a/'initial.npz')
                write(directory/'catalog.json',cat)
                pr=ro.protocol(cat,rc.digest(directory/'catalog.json'),phase,opponent,self.contract,
                               ro.load(self.request['evaluation_environment']['path']))
                result=directory/'results';result.mkdir(exist_ok=True)
                write(result/'protocol.json',pr)
                wins=candidate_wins if side=='candidate' else paired_wins
                rows=[]
                for i in range(64):
                    cutoff=i>=64-cutoffs
                    rows.append(dict(seed=pr['seed']+i,seat_a=i%2,finished=not cutoff,
                                     winner=-1 if cutoff else i%2 if i<wins else 1-i%2,steps=256 if cutoff else 20))
                label=cat['models'][0]['id']
                write(result/(label+'__'+opponent+'-0.json'),dict(records=rows))
                write(result/'summary.json',[dict(a=label,b=opponent,games=64,wins_a=wins,wins_b=64-wins-cutoffs,draws=0,cutoffs=cutoffs)])
                receipt=dict(schema=1,started_at='2026-01-01T00:05:00+00:00',completed_at='2026-01-01T00:05:40+00:00',
                    catalog_sha256=rc.digest(directory/'catalog.json'),contract_sha256=self.request['contract']['sha256'],
                    initialization_sha256=self.request['initialization']['sha256'],source_receipt_sha256=self.request['source_receipt']['sha256'],
                    opponent_sha256=self.request['opponents'][opponent]['entry']['sha256'],evaluator_sha256=rc.digest(Path(ev.__file__)),
                    result_sha256={p.name:rc.digest(p) for p in result.iterdir()})
                write(directory/'evaluation-receipt.json',receipt)
                self.request['bouts'].append(dict(side=side,opponent=opponent,directory=str(directory)))

    def analyze(self):
        return ro.analyze(self.request)

    def mutate_bout(self,filename,mutate):
        d=Path(self.request['bouts'][0]['directory'])
        path=d/filename
        value=ro.load(path);mutate(value);write(path,value)
        r=ro.load(d/'evaluation-receipt.json')
        r['catalog_sha256']=rc.digest(d/'catalog.json')
        r['result_sha256']={p.name:rc.digest(p) for p in (d/'results').iterdir()}
        write(d/'evaluation-receipt.json',r)

    def test_success_development_gate_and_bootstrap(self):
        result=self.analyze()
        self.assertTrue(result['gate']['passed'])
        self.assertEqual(set(result['per_opponent']),{'E1024','F1024','OwnInitial'})
        self.assertEqual(result['development_paired_bootstrap']['E1024']['paired_win_fraction_difference'],20/64)
        self.assertEqual(len(result['bout_evidence']),6)

    def test_confirmation_absolute_gate_not_baseline_improvement(self):
        self.make_bouts('confirmation',candidate_wins=50,paired_wins=51,cutoffs=12)
        self.assertTrue(self.analyze()['gate']['passed'])
        self.make_bouts('confirmation',candidate_wins=36,paired_wins=30)
        self.assertFalse(self.analyze()['gate']['passed'])
        self.make_bouts('confirmation',candidate_wins=50,paired_wins=30,cutoffs=13)
        self.assertFalse(self.analyze()['gate']['passed'])

    def test_development_own_initial_is_required_to_improve(self):
        # Ties fail including OwnInitial (unlike the inherited candidate readout).
        self.make_bouts('development',candidate_wins=30,paired_wins=30)
        self.assertFalse(self.analyze()['gate']['passed'])
        self.assertFalse(self.analyze()['gate']['checks']['OwnInitial'])

    def test_protocol_mutations_rejected_after_receipt_rehash(self):
        d=Path(self.request['bouts'][0]['directory']);p=d/'results/protocol.json';original=ro.load(p)
        changes={'seed':95010000,'games':True,'parallel':32,'workers':2,'limit':255,'device':'cpu',
                 'source_sha256':'f'*64,'controller_sha256':'f'*64,'lana_batch_adapter_sha256':'f'*64,
                 'environment':{},'runtime':{'torch':'2.8.0+cu128','gpu':'other'},'legacy_sources':{},
                 'sampling':'different','catalog_sha256':'f'*64,'extra':1}
        for key,value in changes.items():
            write(p,original)
            self.mutate_bout('results/protocol.json',lambda obj:obj.update({key:value}))
            with self.subTest(key=key),self.assertRaises(ValueError):self.analyze()
        write(p,original)

    def test_catalog_mutations_rejected_after_receipt_rehash(self):
        d=Path(self.request['bouts'][0]['directory']);p=d/'catalog.json';original=ro.load(p)
        mutations=[lambda c:c.update(side='paired'),lambda c:c.update(phase='confirmation'),
            lambda c:c['replication_contract'].update(training_seed=52731415),
            lambda c:c['initialization'].update(gpu='other'),lambda c:c['models'][0].update(sha256='f'*64),
            lambda c:c['models'][0].update(aim_decoding='absolute'),lambda c:c['models'][0].update(trained=1),
            lambda c:c['models'][1].update(root_candidates=4),lambda c:c['models'][1].update(checkpoint='/other')]
        for i,mutation in enumerate(mutations):
            write(p,original);self.mutate_bout('catalog.json',mutation)
            with self.subTest(i=i),self.assertRaises(ValueError):self.analyze()

    def test_raw_mutations_rejected_after_receipt_rehash(self):
        d=Path(self.request['bouts'][0]['directory']);p=next((d/'results').glob('*__*.json'));original=ro.load(p)
        mutations=[lambda x:x['records'].pop(),lambda x:x['records'][0].update(seed=121110001),
                   lambda x:x['records'][0].update(seat_a=1),lambda x:x['records'][0].update(winner=2),
                   lambda x:x['records'][0].update(finished=1),lambda x:x['records'][0].update(steps=257),
                   lambda x:x['records'][0].update(finished=False,winner=-1,steps=20),
                   lambda x:x['records'][0].update(winner=1)]
        for i,mutation in enumerate(mutations):
            write(p,original);self.mutate_bout(str(p.relative_to(d)),mutation)
            with self.subTest(i=i),self.assertRaises(ValueError):self.analyze()

    def test_receipt_bindings_and_timestamps(self):
        d=Path(self.request['bouts'][0]['directory']);p=d/'evaluation-receipt.json';original=ro.load(p)
        changes={key:'f'*64 for key in ('catalog_sha256','contract_sha256','source_receipt_sha256',
                                      'initialization_sha256','opponent_sha256','evaluator_sha256')}
        changes.update(schema=True,result_sha256={},started_at='2025-01-01T00:00:00+00:00',
                       completed_at='2026-01-01T00:04:00+00:00')
        for key,value in changes.items():
            write(p,{**original,key:value})
            with self.subTest(key=key),self.assertRaises(ValueError):self.analyze()

    def test_missing_duplicate_extra_results_and_bouts(self):
        original=copy.deepcopy(self.request['bouts'])
        for entries in (original[:-1],original+[original[0]],[{**original[0],'side':'other'}]+original[1:]):
            self.request['bouts']=entries
            with self.assertRaises(ValueError):self.analyze()
        self.request['bouts']=original
        (Path(original[0]['directory'])/'results/extra.json').write_text('{}')
        with self.assertRaises(ValueError):self.analyze()

    def test_initialization_generator_template_and_counter_binding(self):
        p=Path(self.request['initialization']['path']);original=ro.load(p)
        for key,value in [('generator_sha256','f'*64),('template_sha256','f'*64),('gpu','other'),
                          ('counters',dict(games=0,optimizer_updates=1,training_examples=0))]:
            write(p,{**original,key:value});self.request['initialization']=binding(p)
            with self.subTest(key=key),self.assertRaises(ValueError):self.analyze()

    def test_mir_positive_readout(self):
        self.contract['variant']='MIR'
        self.m['candidate']['arguments'].update(aim_decoding='absolute',mirror_augment=True)
        self.save()
        template=copy.deepcopy(self.contract);template['initial_sha256']=None
        for name,value in [('template',template),('contract',self.contract)]:
            p=Path(self.request[name]['path']);write(p,value);self.request[name]=binding(p)
        self.initialization.update(contract_sha256=self.request['contract']['sha256'],
                                   template_sha256=self.request['template']['sha256'])
        p=Path(self.request['initialization']['path']);write(p,self.initialization);self.request['initialization']=binding(p)
        # Fresh directories; changing an old catalog must not leave obsolete bouts.
        self.make_bouts('confirmation')
        self.assertTrue(self.analyze()['gate']['passed'])
        self.assertEqual(self.analyze()['variant'],'MIR')

    def test_own_initial_tie_alone_fails_development(self):
        entry=next(x for x in self.request['bouts'] if x['side']=='candidate' and x['opponent']=='OwnInitial')
        d=Path(entry['directory']);p=next((d/'results').glob('*__*.json'));raw=ro.load(p)
        for i,row in enumerate(raw['records']):row['winner']=i%2 if i<30 else 1-i%2
        write(p,raw);p=d/'results/summary.json';summary=ro.load(p)
        summary[0].update(wins_a=30,wins_b=34);write(p,summary)
        p=d/'evaluation-receipt.json';receipt=ro.load(p)
        receipt['result_sha256']={f.name:rc.digest(f) for f in (d/'results').iterdir()};write(p,receipt)
        result=self.analyze()
        self.assertFalse(result['gate']['passed'])
        self.assertEqual(result['gate']['checks'],dict(E1024=True,F1024=True,OwnInitial=False))

    def test_frozen_audit_helpers_checked(self):
        with patch.dict(ro.AUDIT_HELPERS,{'readout.py':'f'*64}):
            with self.assertRaisesRegex(ValueError,'Frozen audit helper'):self.analyze()

    def test_summary_and_result_hash_mismatch(self):
        d=Path(self.request['bouts'][0]['directory'])
        self.mutate_bout('results/summary.json',lambda x:x[0].update(wins_a=49))
        with self.assertRaisesRegex(ValueError,'summary mismatch'):self.analyze()
        p=d/'results/summary.json';p.write_text(p.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'Result receipt differs'):self.analyze()

    def test_request_pin_and_endpoint_counter(self):
        with self.assertRaises(ValueError):ro.bound({'path':self.request['contract']['path'],'sha256':'f'*64})
        self.r['paired']['optimizer_updates']=2047;self.save()
        with self.assertRaises(ValueError):self.analyze()

if __name__=='__main__':unittest.main()
