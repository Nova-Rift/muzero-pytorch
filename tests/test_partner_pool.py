"""--partner-pool / --partner-fraction: flag-off byte-identity against the frozen EGO trainer 14fece12, replay masking,
frozen pool weights, CLI/manifest recording and split-resume equality. CPU only, tiny configs."""
import json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from model import Config,MuZero
from arena import train_loss
from arena.environment import Pool
SITE=os.environ.get('ARENA_SITE')
FROZEN='14fece1209067004f4c3936a6171bc40da55577d'
# EGO2048 argument pattern (ego2048-jobs.json) reduced to CPU scale: two blocks of four games so checkpoint-4.npz can join the pool.
TINY=['--parallel','4','--workers','1','--hidden','16','--context','2','--simulations','2','--root-candidates','2','--replay-blocks','3','--batch','4','--unroll','2','--return-mode','episode','--timeout-bootstrap','zero','--discount','1.0','--limit','3','--updates-per-game','1','--agents','2','--policy-floor','0','--noise-fraction','.25','--cadence','homepage','--seed','52731415','--max-seconds','60','--device','cpu','--aim-decoding','relative']
# Exact scoped edits of the partner-pool branch as (added, original) pairs; reversing them reproduces FROZEN byte-for-byte
# (test below) and the other source pins (test_rootweighted, test_mirror_augment) reverse them before their own comparison.
PARTNER_EDITS=[('    def __init__(self,model,count,graph=False,agents=2,root_candidates=0,slots=None):\n        # slots: explicit root slot list (--partner-pool); default keeps the historical per-game agent layout.\n        self.indices=torch.tensor([g*11+a for g in range(count//11) for a in range(agents)] if slots is None else list(slots),dtype=torch.long,device=next(model.parameters()).device)', '    def __init__(self,model,count,graph=False,agents=2,root_candidates=0):\n        self.indices=torch.tensor([g*11+a for g in range(count//11) for a in range(agents)],device=next(model.parameters()).device)'), ('class PartnerPool:\n    """--partner-pool: frozen copies of this run\'s OWN earlier weights (initial.npz / checkpoint-N.npz) play agent 1 in a\n    seeded fraction of every block\'s games. Their slots are traced but never enter the replay indices; the gate opponents\n    E1024/F1024/OwnInitial are foreign weights and can never be members (only own-run files are accepted)."""\n    def __init__(self,model,directories,members,fraction,root_candidates=0):\n        import copy\n        self.model=copy.deepcopy(model).eval()\n        for parameter in self.model.parameters():parameter.requires_grad_(False)\n        self.directories=[Path(d) for d in directories];self.members=members;self.fraction=fraction;self.root_candidates=root_candidates\n    def files(self):\n        # Members whose file exists yet (before checkpoint-N.npz is written the pool is initial.npz only).\n        found=[]\n        for member in self.members:\n            name=\'initial.npz\' if member==\'initial\' else f\'checkpoint-{member}.npz\'\n            for directory in self.directories:\n                if (directory/name).exists():found.append((member,directory/name));break\n        return found\n    def assign(self,games,seed):\n        # Dedicated generator seeded by the block\'s first game seed: no shared RNG state to save, identical on resume.\n        prng=np.random.default_rng(seed);files=self.files();member=np.full(games*11,-1,np.int16)\n        chosen=np.sort(prng.choice(games,int(round(self.fraction*games)),replace=False))\n        member[chosen*11+1]=prng.integers(len(files),size=len(chosen))\n        return member,files\n    def __call__(self,x,member,files):\n        count=len(member);actions=np.zeros(count,np.int64);policies=np.zeros((count,self.model.config.actions),np.float32);values=np.zeros(count,np.float32)\n        for index,(_,path) in enumerate(files):\n            slots=np.flatnonzero(member==index)\n            if not len(slots):continue\n            with np.load(path) as z:self.model.load_state_dict({k:torch.from_numpy(z[k]) for k in z.files})\n            act,pol,value=Search(self.model,count,root_candidates=self.root_candidates,slots=slots)(x)\n            actions[slots],policies[slots],values[slots]=act[slots],pol[slots],value[slots]\n        return actions,policies,values\n\n\ndef history(observations,actions,times,agents,context,mirror=None):', 'def history(observations,actions,times,agents,context,mirror=None):'), ('aim_decoding="absolute",partner=None):\n    # Recorded actions stay model indices (0..152); only the command sent to the bridge carries the decoding offset.\n    aim_offset=wire_offset(aim_decoding)\n    state=pool.reset(seeds);games=len(seeds);count=games*11\n    # --partner-pool: member[slot]>=0 marks a frozen pool agent; learner marks the slots that may enter replay.\n    member,files=partner.assign(games,int(seeds[0])) if partner else (None,[])\n    learner=np.ones(count,bool) if member is None else member<0\n', 'aim_decoding="absolute"):\n    # Recorded actions stay model indices (0..152); only the command sent to the bridge carries the decoding offset.\n    aim_offset=wire_offset(aim_decoding)\n    state=pool.reset(seeds);games=len(seeds);count=games*11\n'), ("        act,pol,value=searcher(x)\n        if member is not None:\n            pa,pp,pv=partner(x,member,files);act[~learner],pol[~learner],value[~learner]=pa[~learner],pp[~learner],pv[~learner]\n        if cache:cache.acted(act)\n        timings['search']+=time.perf_counter()-start\n        action_counts+=np.bincount(act[valid[t]&learner],minlength=153)", "        act,pol,value=searcher(x)\n        if cache:cache.acted(act)\n        timings['search']+=time.perf_counter()-start\n        action_counts+=np.bincount(act[valid[t]],minlength=153)"), ('    completed=valid[:length]&learner[None]&np.repeat(finished,11)[None]\n', '    completed=valid[:length]&np.repeat(finished,11)[None]\n'), ("    indices=np.argwhere((valid&learner[None])[:length])\n    block=dict(observation=obs[:length+1],actions=actions[:length],rewards=rewards[:length],policies=policies[:length+1],targets=targets[:length+1],valid=valid[:length+1],absorbing=absorbing[:length+1],indices=indices)\n    if member is not None:block['partner']=np.repeat(member[None],length+1,0)\n    stats=dict(partner_games=int((~learner).sum()),partner_members={name:int((member==i).sum()) for i,(name,_) in enumerate(files)}) if member is not None else {}\n    stats.update(collection_steps", '    indices=np.argwhere(valid[:length])\n    block=dict(observation=obs[:length+1],actions=actions[:length],rewards=rewards[:length],policies=policies[:length+1],targets=targets[:length+1],valid=valid[:length+1],absorbing=absorbing[:length+1],indices=indices)\n    stats=dict(collection_steps'), ("    p.add_argument('--partner-pool',default='',help='comma-separated OWN-run weights (initial or checkpoint game counts, e.g. initial,512,1024) that play agent 1 in --partner-fraction of each block\\'s games; empty keeps the frozen trainer')\n    p.add_argument('--partner-fraction',type=float,default=0.0,help='seeded fraction of parallel games whose agent 1 is a frozen pool member; 0 reproduces the frozen trainer exactly')\n    p.add_argument('--cadence',choices=['legacy','homepage'],default='homepage')\n", "    p.add_argument('--cadence',choices=['legacy','homepage'],default='homepage')\n"), ("    a=p.parse_args()\n    members=[m for m in a.partner_pool.split(',') if m]\n    if any(m!='initial' and not m.isdigit() for m in members):raise ValueError('--partner-pool accepts initial or own checkpoint game counts only')\n    if not 0<=a.partner_fraction<=1 or (a.partner_fraction and not members):raise ValueError('--partner-fraction requires a pool and lies in [0,1]')\n    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)\n", '    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)\n'), ("gpu=torch.cuda.get_device_name() if a.device=='cuda' else None,partner_pool=dict(members=members,fraction=a.partner_fraction))", "gpu=torch.cuda.get_device_name() if a.device=='cuda' else None)"), ('a.root_candidates);setup=time.perf_counter()-started\n        partner=PartnerPool(model,[out]+([a.resume_state.parent,a.resume_state.parent.parent] if a.resume_state else []),members,a.partner_fraction,a.root_candidates) if a.partner_fraction else None\n', 'a.root_candidates);setup=time.perf_counter()-started\n'), ('a.return_mode,a.timeout_bootstrap,a.aim_decoding,partner)', 'a.return_mode,a.timeout_bootstrap,a.aim_decoding)')]
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

def tiny_model():
    torch.manual_seed(5);return MuZero(Config(observations=64*2+1,actions=153,hidden=16,simulations=2,discount=1.0,support=10,exploration_fraction=.25))

class SourcePinTests(unittest.TestCase):
    def test_only_the_listed_edits_against_frozen_trainer(self):
        source=(ROOT/'arena/train_loss.py').read_text()
        for added,original in PARTNER_EDITS:
            self.assertEqual(source.count(added),1,added[:80]);source=source.replace(added,original)
        self.assertEqual(source,subprocess.check_output(['git','show',FROZEN+':arena/train_loss.py'],cwd=ROOT,text=True))
        # Untouched by this branch: environment, bridge, evaluator, search, model, returns, decoding.
        for name in ('arena/environment.py','arena/bridge.mjs','arena/aim_decoding.py','arena/returns.py','arena/history.py','model.py','batched_search_exploration.py','tournament/run.py','diagnostics/retention/resume_gate.py','diagnostics/retention/train_compressed.py','diagnostics/retention/verify_compressed_states.py'):
            self.assertEqual(subprocess.check_output(['git','show',FROZEN+':'+name],cwd=ROOT),(ROOT/name).read_bytes(),name)

@unittest.skipUnless(SITE,'Set ARENA_SITE to canonical website checkout')
class TrainerTests(unittest.TestCase):
    def test_fraction_zero_is_byte_identical_to_frozen_trainer(self):
        with tempfile.TemporaryDirectory() as d:
            old=run(Path(d)/'frozen',8,frozen=FROZEN);self.assertEqual(old.returncode,0,old.stderr[-2000:])
            new=run(Path(d)/'new',8,['--partner-pool','initial,4','--partner-fraction','0']);self.assertEqual(new.returncode,0,new.stderr[-2000:])
            a,b=arrays(Path(d)/'frozen'),arrays(Path(d)/'new')
            self.assertEqual(sorted(a),sorted(b));self.assertIn('trace-4.npz',a);self.assertIn('weights.npz',a)
            for name in a:
                self.assertEqual(sorted(a[name]),sorted(b[name]),name)  # no partner key when the pool is off
                for key in a[name]:np.testing.assert_array_equal(a[name][key],b[name][key],err_msg=name+':'+key)
            self.assertEqual((Path(d)/'frozen/weights.npz').read_bytes(),(Path(d)/'new/weights.npz').read_bytes())
            volatile={'elapsed_seconds','setup_seconds','training_seconds','games_per_second','batch_seconds'}
            for x,y in zip((Path(d)/'frozen/metrics.jsonl').read_text().splitlines(),(Path(d)/'new/metrics.jsonl').read_text().splitlines()):
                x,y=json.loads(x),json.loads(y);self.assertEqual({k:v for k,v in x.items() if k not in volatile},{k:v for k,v in y.items() if k not in volatile})
                self.assertEqual({k:v for k,v in x['batch_seconds'].items() if k not in ('search','environment','history')},{k:v for k,v in y['batch_seconds'].items() if k not in ('search','environment','history')})
            manifest=json.loads((Path(d)/'new/manifest.json').read_text())
            self.assertEqual(manifest['partner_pool'],dict(members=['initial','4'],fraction=0.0))
    def test_pool_on_marks_agent_one_of_half_the_games_and_grows_with_checkpoints(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'pool';result=run(out,8,['--partner-pool','initial,4','--partner-fraction','0.5']);self.assertEqual(result.returncode,0,result.stderr[-2000:])
            manifest=json.loads((out/'manifest.json').read_text());self.assertEqual(manifest['partner_pool'],dict(members=['initial','4'],fraction=0.5))
            self.assertEqual(manifest['arguments']['partner_pool'],'initial,4');self.assertEqual(manifest['arguments']['partner_fraction'],0.5)
            rows=[json.loads(x)['batch_seconds'] for x in (out/'metrics.jsonl').read_text().splitlines()]
            self.assertEqual([r['partner_games'] for r in rows],[2,2])
            self.assertEqual(rows[0]['partner_members'],{'initial':2})  # checkpoint-4.npz does not exist during the first block
            self.assertEqual(set(rows[1]['partner_members']),{'initial','4'});self.assertEqual(sum(rows[1]['partner_members'].values()),2)
            for name in ('trace-0.npz','trace-4.npz'):
                with np.load(out/name) as z:
                    partner=z['partner'];self.assertEqual(partner.shape[1],22)
                    marked=np.flatnonzero(partner[0]>=0);self.assertTrue(all(s%11==1 for s in marked))
                    self.assertTrue((partner==partner[0]).all(),'membership is fixed for the whole block')
            report=json.loads((out/'report.json').read_text());self.assertTrue(report['requested_run_completed']);self.assertEqual(report['optimizer_updates'],8)
    def test_invalid_flags_rejected_and_default_off(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertNotEqual(run(Path(d)/'a',4,['--partner-fraction','0.5']).returncode,0)  # fraction without a pool
            self.assertNotEqual(run(Path(d)/'b',4,['--partner-pool','initial,E1024','--partner-fraction','0.5']).returncode,0)  # foreign weights can never be members
            self.assertNotEqual(run(Path(d)/'c',4,['--partner-pool','initial','--partner-fraction','1.5']).returncode,0)
            result=run(Path(d)/'d',4);self.assertEqual(result.returncode,0,result.stderr[-2000:])
            manifest=json.loads((Path(d)/'d/manifest.json').read_text());self.assertEqual(manifest['partner_pool'],dict(members=[],fraction=0.0))
            with np.load(Path(d)/'d/trace-0.npz') as z:self.assertNotIn('partner',z.files)
    def test_split_resume_equals_uninterrupted_pool_run(self):
        flags=['--partner-pool','initial,4','--partner-fraction','0.5']
        with tempfile.TemporaryDirectory() as d:
            full=run(Path(d)/'full',8,flags);self.assertEqual(full.returncode,0,full.stderr[-2000:])
            partial=run(Path(d)/'partial',4,flags);self.assertEqual(partial.returncode,0,partial.stderr[-2000:])
            resumed=run(Path(d)/'resumed',8,flags+['--resume-state',str(Path(d)/'partial/resume-state.pt')]);self.assertEqual(resumed.returncode,0,resumed.stderr[-2000:])
            self.assertEqual((Path(d)/'full/weights.npz').read_bytes(),(Path(d)/'resumed/weights.npz').read_bytes())
            a,b=arrays(Path(d)/'full')['trace-4.npz'],arrays(Path(d)/'resumed')['trace-4.npz']
            for key in a:np.testing.assert_array_equal(a[key],b[key],err_msg=key)
            second=json.loads((Path(d)/'resumed/metrics.jsonl').read_text().splitlines()[0])['batch_seconds']
            self.assertEqual(set(second['partner_members']),{'initial','4'},'resumed block finds checkpoint-4.npz beside the resume state')
            changed=run(Path(d)/'changed',8,['--partner-pool','initial','--partner-fraction','0.5','--resume-state',str(Path(d)/'partial/resume-state.pt')])
            self.assertNotEqual(changed.returncode,0);self.assertIn('Resume setting changed: partner_pool',changed.stderr)

@unittest.skipUnless(SITE,'Set ARENA_SITE to canonical website checkout')
class CollectTests(unittest.TestCase):
    def test_masked_slots_never_enter_replay_and_pool_weights_stay_frozen(self):
        model=tiny_model();optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
        with tempfile.TemporaryDirectory() as d:
            initial={k:v.detach().clone() for k,v in model.state_dict().items()}
            np.savez_compressed(Path(d)/'initial.npz',**{k:v.numpy() for k,v in initial.items()})
            pool=Pool(SITE,1,agents=2,cadence='homepage')
            try:
                games=6;partner=train_loss.PartnerPool(model,[d],['initial','512'],0.5,2)
                self.assertTrue(all(not p.requires_grad for p in partner.model.parameters()))
                self.assertFalse({p.data_ptr() for p in partner.model.parameters()}&{p.data_ptr() for p in model.parameters()},'pool holds its own storage')
                member,files=partner.assign(games,900);self.assertEqual([m for m,_ in files],['initial'],'checkpoint-512 is not a member until it exists')
                np.testing.assert_array_equal(member,train_loss.PartnerPool(model,[d],["initial","512"],0.5,2).assign(games,900)[0])  # seeded by the first game seed, reproducible
                self.assertEqual(int((member>=0).sum()),3);self.assertTrue(all(s%11==1 for s in np.flatnonzero(member>=0)))
                searcher=train_loss.Search(model,11*games,False,2,2)
                torch.manual_seed(11);block,stats=train_loss.collect(pool,model,range(900,900+games),3,2,searcher,False,'episode','zero','relative',partner)
                self.assertEqual(stats['partner_games'],3);self.assertEqual(stats['partner_members'],{'initial':3})
                pooled=set(np.flatnonzero(member>=0).tolist())
                self.assertEqual(len(block['indices']),int((block['valid'][:3]&(member<0)[None]).sum()))
                self.assertFalse(pooled&set(block['indices'][:,1].tolist()),'pool slots are masked from replay')
                self.assertTrue(set(block['indices'][:,1].tolist())<=set(np.flatnonzero((member<0)&block['valid'][0]).tolist()))
                np.testing.assert_array_equal(block['partner'][0],member);self.assertEqual(block['partner'].shape,(4,66))
                # pool decisions were taken: masked slots carry live actions/policies recorded in the trace
                self.assertTrue((block['policies'][0][sorted(pooled)].sum(-1)>0.99).all())
                # a learner update leaves the pool model untouched and equal to the file it was loaded from
                tensors=train_loss.replay_batch(block,np.random.default_rng(0),4,2,2,'cpu');loss=model.loss(*tensors).mean();optimizer.zero_grad();loss.backward();optimizer.step()
                self.assertFalse(all(torch.equal(model.state_dict()[k],initial[k]) for k in initial),'learner moved')
                for k,v in partner.model.state_dict().items():self.assertTrue(torch.equal(v,initial[k]),k)
                # flag off: collect() ignores the pool entirely and produces the frozen block layout
                torch.manual_seed(11);plain,plain_stats=train_loss.collect(pool,model,range(900,900+games),3,2,searcher,False,'episode','zero','relative')
                self.assertNotIn('partner',plain);self.assertNotIn('partner_games',plain_stats);self.assertEqual(len(plain['indices']),int(plain['valid'][:3].sum()))
            finally:pool.close()

if __name__=='__main__':unittest.main()
