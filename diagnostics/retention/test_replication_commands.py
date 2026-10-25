import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import replication_commands as c

class CommandTests(unittest.TestCase):
    def setUp(self):
        self.s=c.specification()
        self.b={k:('a'*64 if k.endswith('_sha256') else '/fixture/'+k) for k in self.s['bindings']}
        self.b.update(c.PINNED)
        self.r={j['id']:'b'*64 for j in self.s['jobs']}
    def test_initial_only_and_unknowns(self):
        self.assertFalse(self.s['approved']);self.assertFalse(self.s['executable'])
        self.assertIsNone(self.s['selected_variant']);self.assertIsNone(self.s['initial_sha256'])
        self.assertTrue(all(v is None for v in self.s['bindings'].values()))
        cmds=c.render(self.s,'no-update-initialization',self.b,self.r)
        self.assertEqual(len(cmds),1);self.assertIn('replication_initialization.py',cmds[0][1])
        self.assertNotIn('--games',cmds[0]);self.assertNotIn('--site',cmds[0])
        self.assertEqual(c.ancestors(self.s,'no-update-initialization'),{'original-selection','conditional-admission'})
    def test_dependencies_cannot_be_dropped(self):
        for j in self.s['jobs']:
            if not j.get('commands'):continue
            for ancestor in c.ancestors(self.s,j['id']):
                r=dict(self.r);del r[ancestor]
                with self.assertRaises(ValueError):c.render(self.s,j['id'],self.b,r)
    def test_unresolved_and_bad_hash_rejected(self):
        for identity in ('paired-fresh2048','candidate-fresh2048','exact-pair-and-recovery'):
            job=next(j for j in self.s['jobs'] if j['id']==identity)
            keys={k for cmd in job['commands'] for v in cmd for k in c.TOKEN.findall(v)}
            for key in keys:
                b=dict(self.b);b[key]=None
                with self.assertRaises(ValueError):c.render(self.s,identity,b,self.r)
            b=dict(self.b);b['contract_sha256']='unknown'
            with self.assertRaises(ValueError):c.render(self.s,identity,b,self.r)
    def test_mutations_fail_closed(self):
        for key,value in [('approved',True),('selected_variant','EGO'),('training_seed',52731415)]:
            s=copy.deepcopy(self.s);s[key]=value
            with self.assertRaises(ValueError):c.validate(s)
        s=copy.deepcopy(self.s);s['jobs'][10]['depends_on']=[]
        with self.assertRaises(ValueError):c.validate(s)
    def test_pair_commands_and_exact_checks(self):
        for side in ('paired','candidate'):
            cmd=c.render(self.s,side+'-fresh2048',self.b,self.r)[0]
            self.assertTrue(cmd[1].endswith('train_compressed.py'))
            self.assertIn('600000000',cmd);self.assertIn('960',cmd);self.assertIn('850',cmd)
            self.assertIn('--own-reference-sha256',cmd)
            if side=='candidate':self.assertIn('--paired-reference-sha256',cmd)
            deps=c.ancestors(self.s,side+'-fresh2048')
            self.assertTrue({'initialization-review',side+'-runtime',side+'-resume',side+'-collection'}<=deps)
        pair=c.render(self.s,'exact-pair-and-recovery',self.b,self.r)
        self.assertEqual(len(pair),3)
        self.assertTrue(pair[0][1].endswith('replication_recovery.py'))
        self.assertTrue(pair[2][1].endswith('replication_contract.py'))
    def test_confirmation_gate_and_scientific_review(self):
        evaluations=[j for j in self.s['jobs'] if j['stage']=='evaluate']
        self.assertEqual(len(evaluations),10)
        for j in evaluations:
            self.assertIn('exact-pair-and-recovery',c.ancestors(self.s,j['id']))
            if j['phase']=='confirmation':self.assertIn('development-pass-review',c.ancestors(self.s,j['id']))
        with self.assertRaises(ValueError):c.render(self.s,'development-pass-review',self.b,self.r)
    def test_argv_interfaces(self):
        # Help exits before runtime/CUDA actions. Check every supplied flag against
        # each real adapter parser, including the nested train command separately.
        root=Path(c.__file__).parent
        commands=[cmd for j in self.s['jobs'] for cmd in j.get('commands',[])]
        helptexts={}
        for cmd in commands:
            parts=[cmd]
            if '--' in cmd:parts=[cmd[:cmd.index('--')],cmd[cmd.index('--')+1:]]
            for part in parts:
                script=root/Path(part[1]).name
                if script not in helptexts:
                    r=subprocess.run([sys.executable,str(script),'--help'],capture_output=True,text=True,timeout=20)
                    self.assertEqual(r.returncode,0,r.stderr);helptexts[script]=r.stdout
                for flag in part[2:]:
                    if flag.startswith('--'):self.assertIn(flag,helptexts[script])
    def test_wrong_ego_template_rejected(self):
        self.b['ego_template_sha256']='f'*64
        with self.assertRaises(ValueError):c.render(self.s,'no-update-initialization',self.b,self.r)
    def test_runtime_storage_honesty(self):
        e=c.estimates(self.s)
        self.assertFalse(e['approved']);self.assertIsNone(e['storage']['other_outstanding_bytes'])
        self.assertIsNone(e['storage']['payload_bytes'])
        self.assertEqual(e['runtime']['optimistic_non_admission_pair_plus_10_bouts_seconds'],1083.8)
        self.assertGreater(e['runtime']['total_seconds'],1083.8)
    def test_single_initializer_supervisor_shape(self):
        path=os.environ.get('SUPERVISOR')
        if not path:self.skipTest('SUPERVISOR not supplied')
        sp=importlib.util.spec_from_file_location('supervisor',path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
        with tempfile.TemporaryDirectory() as d:
            plan=c.initializer_plan(self.s,self.b,self.r,d,100,200)
            self.assertEqual(m.validate(plan),plan)

if __name__=='__main__':unittest.main()
