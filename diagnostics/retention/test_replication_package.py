"""Conditional topology/binding tests; synthetic CPU evidence only."""
import copy
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest

import replication_package as package
import replication_contract as rc
import replication_initialization as init
import evaluate_replication as ev
import readout_replication as ro


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.graph=package.graph()
        self.jobs={j['id']:j for j in self.graph['jobs']}

    def ancestors(self,identity):
        direct=set(self.jobs[identity]['depends_on'])
        return direct | set().union(*(self.ancestors(p) for p in direct))

    def test_selection_and_initialization_precede_every_training(self):
        package.validate_graph(self.graph)
        for side in ('paired','candidate'):
            parents=self.ancestors(side+'-fresh2048')
            self.assertTrue({'original-selection','conditional-admission','no-update-initialization',
                             'initialization-review',side+'-runtime',side+'-resume',side+'-collection'}<=parents)
        self.assertIn('paired-collection',self.ancestors('candidate-collection'))
        self.assertFalse(self.graph['approved']);self.assertFalse(self.graph['executable'])
        self.assertIsNone(self.graph['selected_variant']);self.assertIsNone(self.graph['initial_sha256'])

    def test_pair_acceptance_and_dev_pass_precede_confirmation(self):
        for job in self.graph['jobs']:
            if job['stage']=='evaluate':
                self.assertIn('exact-pair-and-recovery',self.ancestors(job['id']))
                self.assertEqual(job['device_count'],1)
                self.assertEqual(job['games'],64)
                self.assertEqual(job['seed'],ev.SEEDS[job['phase']][job['opponent']])
                if job['phase']=='confirmation':
                    self.assertIn('development-pass-review',self.ancestors(job['id']))
        self.assertEqual(len([j for j in self.graph['jobs'] if j['stage']=='evaluate']),10)
        self.assertEqual(set(self.jobs['exact-pair-and-recovery']['depends_on']),
                         {'paired-fresh2048','candidate-fresh2048'})

    def test_dependency_bypasses_and_unapproved_commands_rejected(self):
        for identity in ('no-update-initialization','paired-fresh2048','candidate-collection',
                         'exact-pair-and-recovery','candidate-confirmation-E1024'):
            changed=copy.deepcopy(self.graph)
            next(j for j in changed['jobs'] if j['id']==identity)['depends_on']=[]
            with self.subTest(identity=identity),self.assertRaises(ValueError):package.validate_graph(changed)
        changed=copy.deepcopy(self.graph);changed['jobs'][0]['command']=['true']
        with self.assertRaises(ValueError):package.validate_graph(changed)

    def test_seed_recipe_and_selection_mutations_rejected(self):
        for key,value in [('seed',52731415),('approved',True),('executable',True),
                          ('selected_variant','EGO'),('initial_sha256','0'*64),('gpu_overlap_assumed',True)]:
            changed=copy.deepcopy(self.graph);changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):package.validate_graph(changed)
        changed=copy.deepcopy(self.graph)
        next(j for j in changed['jobs'] if j['id']=='candidate-fresh2048')['recipe']['updates_per_game']=2
        with self.assertRaises(ValueError):package.validate_graph(changed)

    def test_unresolved_collection_and_operational_bindings(self):
        self.assertEqual(self.jobs['paired-collection']['implementation'],'replication_collection.py --mode reference')
        self.assertEqual(self.jobs['candidate-collection']['implementation'],'replication_collection.py --mode reference')
        self.assertIsNone(self.jobs['candidate-fresh2048']['intervention'])
        self.assertEqual(self.jobs['paired-fresh2048']['recipe'],rc.RECIPE)
        estimate=package.estimates(self.graph)
        self.assertIsNone(estimate['storage']['other_outstanding_bytes'])
        self.assertFalse(estimate['approved'])
        self.assertGreaterEqual(estimate['storage']['scratch_peak_bytes'],2*1016000000)
        self.assertGreaterEqual(estimate['runtime']['total_seconds'],7428)

    def test_stage_binds_integrated_generator_and_preserves_null_templates(self):
        path=os.environ.get('REPLICATION_PREREQUISITES')
        if not path:self.skipTest('Set source-bound prerequisite path for staging integration')
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'package'
            package.stage(Path(path),out)
            manifest=json.loads((out/'package-manifest.json').read_text())
            for name,sha in manifest['files'].items():self.assertEqual(rc.digest(out/name),sha)
            self.assertEqual(manifest['initializer_sha256'],rc.digest(Path(init.__file__)))
            self.assertEqual(manifest['evaluator_sha256'],rc.digest(Path(ev.__file__)))
            self.assertEqual(manifest['readout_sha256'],rc.digest(Path(ro.__file__)))
            for variant in ('ego','mir'):
                template=json.loads((out/'inputs'/f'contract-{variant}.template.json').read_text())
                self.assertIsNone(template['initial_sha256']);init.validate_template(template)
                with self.assertRaises(ValueError):rc.validate_contract(template)
            for script in ('replication_package.py','readout_replication.py','evaluate_replication.py','replication_initialization.py','replication_collection.py','replication_recovery.py'):
                subprocess.run([sys.executable,str(out/'diagnostics'/'retention'/script),'--help'],
                               check=True,capture_output=True,timeout=10)
            self.assertFalse((out/'initial.npz').exists())
            self.assertFalse((out/'initialization.json').exists())
            with self.assertRaises(ValueError):package.stage(Path(path),out)

if __name__=='__main__':unittest.main()
