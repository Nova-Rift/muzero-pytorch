import copy
import json
from pathlib import Path
import tempfile
import unittest
from readout import analyze, audit_bout


class ReadoutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.manifest = {'bouts': []}
        for arm in ('baseline', 'reference-q'):
            for opponent, seed, sha in [('E1024',92320000,'18c18151b820636ef3bf67ea52dbecd3cb47582d9ea6bf63ede36d6c2adb1532'),('F1024',92330000,'4de08ce08adef8344d6cf8707e22a74896c355008ffff18d3b10ce21ec1dcdb7')]:
                directory = Path(self.temp.name)/f'{arm}-{opponent}'
                directory.mkdir()
                model = {'id':arm,'sha256':('a' if arm=='baseline' else 'b')*64,'root_candidates':8,'config':{'discount':1.0,'simulations':32}}
                opp = {'id':opponent,'sha256':sha,'root_candidates':8,'config':{'discount':.99,'simulations':32}}
                p = {'games':64,'limit':256,'seed':seed,'models':[model,opp],'source_sha256':'c'*64,'controller_sha256':'d'*64,'catalog_sha256':'e'*64,'environment':{'test':'f'*64},'sampling':'independent PCG64 per game and seat; no root noise'}
                records = [{'seed':seed+i,'seat_a':i%2,'winner':i%2 if arm=='reference-q' else 1-i%2,'finished':True,'steps':10} for i in range(64)]
                summary = [{'a':arm,'b':opponent,'games':64,'wins_a':64 if arm=='reference-q' else 0,'wins_b':0 if arm=='reference-q' else 64,'draws':0,'cutoffs':0}]
                for name,data in [('protocol.json',p),('summary.json',summary),(f'{arm}__{opponent}-0.json',{'records':records})]:
                    (directory/name).write_text(json.dumps(data))
                self.manifest['bouts'].append({'arm':arm,'opponent':opponent,'directory':str(directory),'protocol':p})

    def mutate_records(self, function):
        entry=self.manifest['bouts'][0]
        path=next(Path(entry['directory']).glob('*__*.json'))
        data=json.loads(path.read_text());function(data['records']);path.write_text(json.dumps(data))

    def test_known_effect_and_determinism(self):
        result=analyze(self.manifest)
        self.assertEqual(result, analyze(self.manifest))
        self.assertEqual(result['secondary_equal_opponent_difference'],1)
        self.assertEqual(result['per_opponent']['E1024']['percentile95'],[1,1])

    def test_partial(self):
        self.mutate_records(lambda r:r.pop())
        with self.assertRaisesRegex(ValueError,'partial'): analyze(self.manifest)

    def test_duplicate_seed(self):
        self.mutate_records(lambda r:r.__setitem__(1,copy.deepcopy(r[0])))
        with self.assertRaises(ValueError): analyze(self.manifest)

    def test_seat(self):
        self.mutate_records(lambda r:r[0].update(seat_a=1))
        with self.assertRaisesRegex(ValueError,'seat'): analyze(self.manifest)

    def test_summary_mismatch(self):
        self.mutate_records(lambda r:r[0].update(winner=0))
        with self.assertRaisesRegex(ValueError,'summary'): analyze(self.manifest)

    def test_unbound_protocol(self):
        self.manifest['bouts'][0]['protocol']['source_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'protocol differs'): analyze(self.manifest)

    def test_pair_mismatch_even_when_expected(self):
        e=self.manifest['bouts'][0];e['protocol']['controller_sha256']='0'*64
        (Path(e['directory'])/'protocol.json').write_text(json.dumps(e['protocol']))
        with self.assertRaisesRegex(ValueError,'incompatible'): analyze(self.manifest)

    def test_wrong_opponent_even_when_expected(self):
        e=self.manifest['bouts'][0];e['protocol']['models'][1]['sha256']='0'*64
        (Path(e['directory'])/'protocol.json').write_text(json.dumps(e['protocol']))
        with self.assertRaisesRegex(ValueError,'opponent weights'): analyze(self.manifest)

    def test_cutoff_and_draw_score_zero(self):
        self.mutate_records(lambda r:(r[0].update(winner=-1,finished=False,steps=256),r[1].update(winner=-1,finished=True)))
        e=self.manifest['bouts'][0];p=Path(e['directory'])/'summary.json';s=json.loads(p.read_text());s[0].update(wins_b=62,cutoffs=1,draws=1);p.write_text(json.dumps(s))
        b=audit_bout(e['directory'],e['protocol'])
        self.assertEqual(b['counts'],dict(wins=0,losses=62,cutoffs=1,draws=1))
        self.assertEqual(sum(b['scores']),0)

    def test_duplicate_bout(self):
        self.manifest['bouts'][1]=self.manifest['bouts'][0]
        with self.assertRaisesRegex(ValueError,'duplicate bout'): analyze(self.manifest)

if __name__=='__main__': unittest.main()
