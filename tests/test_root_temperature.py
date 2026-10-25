"""--root-prior-temperature / --action-temperature (research/iteration36-explore): self-play-only exploration flags.
Flag-off byte-identity against the frozen EGO trainer 14fece12, exact scoped-edit pins, search-level behaviour, CLI/manifest
recording and split-resume equality. CPU only, tiny configs. Set ARENA_SITE to the canonical website checkout for trainer runs."""
import json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from model import Config,MuZero
from batched_search_exploration import search_batch
from batched_search import search_batch as original_search
from arena import train_loss
SITE=os.environ.get('ARENA_SITE')
FROZEN='14fece1209067004f4c3936a6171bc40da55577d'
# EGO2048 argument pattern (iteration30 ego2048-jobs.json) reduced to CPU scale, as in the partner-pool contract.
TINY=['--parallel','4','--workers','1','--hidden','16','--context','2','--simulations','2','--root-candidates','2','--replay-blocks','3','--batch','4','--unroll','2','--return-mode','episode','--timeout-bootstrap','zero','--discount','1.0','--limit','3','--updates-per-game','1','--agents','2','--policy-floor','0','--noise-fraction','.25','--cadence','homepage','--seed','52731415','--max-seconds','60','--device','cpu','--aim-decoding','relative']
# Exact scoped edits as (added, original) pairs; reversing them reproduces FROZEN byte-for-byte (test below).
# tests/test_rootweighted.py reverses EXPLORE_EDITS before its own trainer comparison.
EXPLORE_EDITS=[
[
"    def __init__(self,model,count,graph=False,agents=2,root_candidates=0,root_temperature=1.0,action_temperature=1.0):\n        # --root-prior-temperature / --action-temperature (self-play only; 1.0 reproduces the frozen trainer exactly).\n        self.root_temperature=root_temperature;self.action_temperature=action_temperature\n",
"    def __init__(self,model,count,graph=False,agents=2,root_candidates=0):\n"
],
[
"                for _ in range(2):search_batch(model,self.input,root_noise=self.noise,static=True,root_candidates=self.candidates,root_temperature=root_temperature)\n",
"                for _ in range(2):search_batch(model,self.input,root_noise=self.noise,static=True,root_candidates=self.candidates)\n"
],
[
"                self.result=search_batch(model,self.input,root_noise=self.noise,static=True,root_candidates=self.candidates,root_temperature=root_temperature)\n",
"                self.result=search_batch(model,self.input,root_noise=self.noise,static=True,root_candidates=self.candidates)\n"
],
[
"            prior=(1-self.model.config.exploration_fraction)*(logits.softmax(-1) if self.root_temperature==1.0 else (logits/self.root_temperature).softmax(-1))+self.model.config.exploration_fraction*self.noise\n",
"            prior=(1-self.model.config.exploration_fraction)*logits.softmax(-1)+self.model.config.exploration_fraction*self.noise\n"
],
[
"        else:policies,values=search_batch(self.model,self.input,root_noise=self.noise,root_candidates=self.candidates,root_temperature=self.root_temperature)\n        # Acting temperature reshapes only the sampled self-play action; the recorded policy target stays the visit distribution.\n        actions=torch.multinomial(policies if self.action_temperature==1.0 else policies**(1/self.action_temperature),1).squeeze(-1)\n",
"        else:policies,values=search_batch(self.model,self.input,root_noise=self.noise,root_candidates=self.candidates)\n        actions=torch.multinomial(policies,1).squeeze(-1)\n"
],
[
"    p.add_argument('--noise-fraction',type=float,default=.25)\n    p.add_argument('--root-prior-temperature',type=float,default=1.0,help='self-play only: softmax(logits/T) for the root prior used by candidate sampling and root PUCT; 1.0 reproduces the frozen trainer exactly')\n    p.add_argument('--action-temperature',type=float,default=1.0,help='self-play only: sample actions from visits**(1/T); the recorded policy target is unchanged; 1.0 reproduces the frozen trainer exactly')\n",
"    p.add_argument('--noise-fraction',type=float,default=.25)\n"
],
[
"    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)\n    if not (a.root_prior_temperature>0 and a.action_temperature>0):raise ValueError('Temperatures must be positive')\n",
"    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)\n"
],
[
"    manifest=dict(arguments=arguments,exploration=dict(root_prior_temperature=a.root_prior_temperature,action_temperature=a.action_temperature),config=asdict(model.config),",
"    manifest=dict(arguments=arguments,config=asdict(model.config),"
],
[
"        searcher=Search(model,11*a.parallel,a.cuda_graph,a.agents,a.root_candidates,a.root_prior_temperature,a.action_temperature);setup=time.perf_counter()-started\n",
"        searcher=Search(model,11*a.parallel,a.cuda_graph,a.agents,a.root_candidates);setup=time.perf_counter()-started\n"
],
[
"            if count!=a.parallel:searcher=Search(model,11*count,a.cuda_graph,a.agents,a.root_candidates,a.root_prior_temperature,a.action_temperature)\n",
"            if count!=a.parallel:searcher=Search(model,11*count,a.cuda_graph,a.agents,a.root_candidates)\n"
]
]
SEARCH_EDITS=[
[
"def search_batch(model, observations, *, explore=False, legal=None, root_noise=None, static=False, root_candidates=None, root_temperature=1.0):",
"def search_batch(model, observations, *, explore=False, legal=None, root_noise=None, static=False, root_candidates=None):"
],
[
"    # root_temperature>1 flattens the ROOT prior only (self-play exploration flag); 1.0 is the exact historical path.\n    priors[:,0] = policy.softmax(-1) if root_temperature==1.0 else (policy/root_temperature).softmax(-1)\n",
"    priors[:,0] = policy.softmax(-1)\n"
]
]
RUNNER="import subprocess,sys,types\nsource=subprocess.check_output(['git','show',sys.argv[1]+':arena/train_loss.py'],cwd=sys.argv[2],text=True)\nsys.argv=[sys.argv[2]+'/arena/train_loss.py']+sys.argv[3:]\nm=types.ModuleType('__main__');m.__file__=sys.argv[0];exec(compile(source,m.__file__,'exec'),m.__dict__)\n"

def run(output,games,extra=(),frozen=None):
    args=['--site',SITE,'--output',str(output),'--games',str(games)]+TINY+list(extra)
    if frozen:command=[sys.executable,'-B','-c',RUNNER,frozen,str(ROOT)]+args
    else:command=[sys.executable,'-B',str(ROOT/'arena/train_loss.py')]+args
    return subprocess.run(command,capture_output=True,text=True,timeout=240,cwd=ROOT)

def arrays(directory):
    out={}
    for path in sorted(Path(directory).glob('*.npz')):
        with np.load(path) as z:out[path.name]={k:z[k] for k in z.files}
    return out

class SourcePinTests(unittest.TestCase):
    def test_only_the_listed_edits_against_frozen_trainer(self):
        source=(ROOT/'arena/train_loss.py').read_text()
        for added,original in EXPLORE_EDITS:
            self.assertEqual(source.count(added),1,added[:80]);source=source.replace(added,original)
        self.assertEqual(source,subprocess.check_output(['git','show',FROZEN+':arena/train_loss.py'],cwd=ROOT,text=True))
        for name in ('batched_search_exploration.py','batched_search_qcompletion.py'):
            source=(ROOT/name).read_text()
            for added,original in SEARCH_EDITS:
                self.assertEqual(source.count(added),1,name+': '+added[:80]);source=source.replace(added,original)
            self.assertEqual(source,subprocess.check_output(['git','show',FROZEN+':'+name],cwd=ROOT,text=True),name)
        # Untouched by this branch: environment, bridge, evaluator protocol, model, returns, decoding, scalar search.
        for name in ('arena/environment.py','arena/bridge.mjs','arena/aim_decoding.py','arena/returns.py','arena/history.py','model.py','batched_search.py','arena/frozen_search.py','tournament/run.py','tournament/controllers.py','diagnostics/retention/resume_gate.py','diagnostics/retention/train_compressed.py','diagnostics/retention/verify_compressed_states.py'):
            self.assertEqual(subprocess.check_output(['git','show',FROZEN+':'+name],cwd=ROOT),(ROOT/name).read_bytes(),name)

class SearchTests(unittest.TestCase):
    def model(self):
        torch.manual_seed(3);return MuZero(Config(observations=5,actions=9,hidden=16,simulations=8))
    def test_temperature_one_is_the_original_search(self):
        m=self.model();x=torch.randn(3,5)
        a,b=original_search(m,x);c,d=search_batch(m,x,root_temperature=1.0)
        torch.testing.assert_close(a,c);torch.testing.assert_close(b,d)
        torch.manual_seed(5);noise=torch.distributions.Dirichlet(torch.full((3,9),0.3)).sample();roots=torch.tensor([[0,1,2]]*3)
        e,f=search_batch(m,x,root_noise=noise,root_candidates=roots);g,h=search_batch(m,x,root_noise=noise,root_candidates=roots,root_temperature=1.0)
        torch.testing.assert_close(e,g);torch.testing.assert_close(f,h)
    def test_temperature_flattens_only_the_root_prior(self):
        # Collapsed prior on action 0: the forced-candidate floor (one visit each) survives, and the hot root prior lets
        # non-favourite candidates take visits that the cold prior gives to the favourite.
        m=MuZero(Config(observations=5,actions=9,hidden=16,simulations=32))
        with torch.no_grad():m.policy[-1].weight.zero_();m.policy[-1].bias.fill_(0);m.policy[-1].bias[0]=6.3
        x=torch.zeros(4,5);roots=torch.tensor([[0,1,2,3]]*4);noise=torch.full((4,9),1/9)
        cold,_=search_batch(m,x,root_noise=noise,root_candidates=roots);hot,_=search_batch(m,x,root_noise=noise,root_candidates=roots,root_temperature=4.0)
        torch.testing.assert_close(cold.sum(-1),torch.ones(4));torch.testing.assert_close(hot.sum(-1),torch.ones(4))
        self.assertTrue((cold.gather(1,roots)>=1/32).all());self.assertTrue((hot.gather(1,roots)>=1/32).all())
        self.assertGreater(float(cold[:,0].mean()),float(hot[:,0].mean()))
        with self.assertRaises(TypeError):original_search(m,x,root_temperature=2.0)
    def test_action_temperature_changes_sampling_but_not_the_recorded_policy(self):
        m=self.model();cold=train_loss.Search(m,11,False,2,0,1.0,1.0);hot=train_loss.Search(m,11,False,2,0,1.0,3.0)
        x=torch.randn(11,5);agree=[0,0]
        for i in range(150):
            torch.manual_seed(100+i);a,p,v=cold(x);torch.manual_seed(100+i);b,q,w=hot(x)
            np.testing.assert_array_equal(p,q);np.testing.assert_array_equal(v,w)  # same visits and values: the recorded target is untouched
            self.assertEqual(p.shape,(11,9));agree[0]+=int(a[0]==p[0].argmax());agree[1]+=int(b[0]==q[0].argmax())
        # visits**(1/3) sampling picks the top-visit action less often than sampling the visits themselves.
        self.assertGreater(agree[0],agree[1],agree)

@unittest.skipUnless(SITE,'Set ARENA_SITE to canonical website checkout')
class TrainerTests(unittest.TestCase):
    def test_default_flags_are_byte_identical_to_frozen_trainer(self):
        with tempfile.TemporaryDirectory() as d:
            old=run(Path(d)/'frozen',8,frozen=FROZEN);self.assertEqual(old.returncode,0,old.stderr[-2000:])
            new=run(Path(d)/'new',8,['--root-prior-temperature','1.0','--action-temperature','1.0']);self.assertEqual(new.returncode,0,new.stderr[-2000:])
            a,b=arrays(Path(d)/'frozen'),arrays(Path(d)/'new')
            self.assertEqual(sorted(a),sorted(b));self.assertIn('trace-4.npz',a);self.assertIn('weights.npz',a)
            for name in a:
                self.assertEqual(sorted(a[name]),sorted(b[name]),name)
                for key in a[name]:np.testing.assert_array_equal(a[name][key],b[name][key],err_msg=name+':'+key)
            self.assertEqual((Path(d)/'frozen/weights.npz').read_bytes(),(Path(d)/'new/weights.npz').read_bytes())
            volatile={'elapsed_seconds','setup_seconds','training_seconds','games_per_second','batch_seconds'}
            for x,y in zip((Path(d)/'frozen/metrics.jsonl').read_text().splitlines(),(Path(d)/'new/metrics.jsonl').read_text().splitlines()):
                x,y=json.loads(x),json.loads(y);self.assertEqual({k:v for k,v in x.items() if k not in volatile},{k:v for k,v in y.items() if k not in volatile})
                self.assertEqual({k:v for k,v in x['batch_seconds'].items() if k not in ('search','environment','history')},{k:v for k,v in y['batch_seconds'].items() if k not in ('search','environment','history')})
            old_m,new_m=json.loads((Path(d)/'frozen/manifest.json').read_text()),json.loads((Path(d)/'new/manifest.json').read_text())
            self.assertEqual(new_m['exploration'],dict(root_prior_temperature=1.0,action_temperature=1.0))
            self.assertEqual({k:v for k,v in new_m['arguments'].items() if k not in ('output','root_prior_temperature','action_temperature')},{k:v for k,v in old_m['arguments'].items() if k!='output'})
            self.assertEqual(new_m['config'],old_m['config'])
    def test_hot_root_prior_changes_self_play_and_is_recorded(self):
        with tempfile.TemporaryDirectory() as d:
            cold=run(Path(d)/'cold',8);self.assertEqual(cold.returncode,0,cold.stderr[-2000:])
            hot=run(Path(d)/'hot',8,['--root-prior-temperature','2.0']);self.assertEqual(hot.returncode,0,hot.stderr[-2000:])
            m=json.loads((Path(d)/'hot/manifest.json').read_text());self.assertEqual(m['exploration'],dict(root_prior_temperature=2.0,action_temperature=1.0))
            self.assertEqual(m['arguments']['root_prior_temperature'],2.0);self.assertEqual(m['config'],json.loads((Path(d)/'cold/manifest.json').read_text())['config'])
            a,b=arrays(Path(d)/'cold'),arrays(Path(d)/'hot')
            self.assertEqual((Path(d)/'cold/initial.npz').read_bytes(),(Path(d)/'hot/initial.npz').read_bytes(),'same seed, same initial weights')
            self.assertFalse(all(np.array_equal(a['trace-0.npz'][k],b['trace-0.npz'][k]) for k in ('policies','actions')),'hot root prior changes the first self-play block')
            report=json.loads((Path(d)/'hot/report.json').read_text());self.assertTrue(report['requested_run_completed']);self.assertEqual(report['optimizer_updates'],8)
    def test_invalid_temperatures_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertNotEqual(run(Path(d)/'a',4,['--root-prior-temperature','0']).returncode,0)
            self.assertNotEqual(run(Path(d)/'b',4,['--action-temperature','-1']).returncode,0)
    def test_split_resume_equals_uninterrupted_hot_run(self):
        flags=['--root-prior-temperature','2.0']
        with tempfile.TemporaryDirectory() as d:
            full=run(Path(d)/'full',8,flags);self.assertEqual(full.returncode,0,full.stderr[-2000:])
            partial=run(Path(d)/'partial',4,flags);self.assertEqual(partial.returncode,0,partial.stderr[-2000:])
            resumed=run(Path(d)/'resumed',8,flags+['--resume-state',str(Path(d)/'partial/resume-state.pt')]);self.assertEqual(resumed.returncode,0,resumed.stderr[-2000:])
            self.assertEqual((Path(d)/'full/weights.npz').read_bytes(),(Path(d)/'resumed/weights.npz').read_bytes())
            a,b=arrays(Path(d)/'full')['trace-4.npz'],arrays(Path(d)/'resumed')['trace-4.npz']
            for key in a:np.testing.assert_array_equal(a[key],b[key],err_msg=key)
            changed=run(Path(d)/'changed',8,['--resume-state',str(Path(d)/'partial/resume-state.pt')])
            self.assertNotEqual(changed.returncode,0);self.assertIn('Resume setting changed: root_prior_temperature',changed.stderr)
if __name__=='__main__':unittest.main()
