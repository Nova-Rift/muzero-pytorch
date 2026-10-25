"""Egocentric (relative) aim decoding: Python mirror, JS bridge, and trainer flag."""
import json,math,os,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np
from arena.aim_decoding import ACTIONS,MODES,bearing_from_observation,decode,wire_action,wire_offset
from arena.environment import Pool

ROOT=Path(__file__).resolve().parents[1]
SITE=os.environ.get('ARENA_SITE')
PRE_CHANGE_BRIDGE='c069f5efe596f95367bf90f9f7eff1ee75af47e9'

def legacy_decode(a):
    """Verbatim pre-change bridge.mjs controller arithmetic (absolute aim)."""
    move=a%9;aim=a//9;angle=(aim-1)*math.pi/8
    return dict(dx=move%3-1,dy=move//3-1,fire=aim>0,aimX=math.cos(angle),aimY=math.sin(angle))

def synthetic_observation(self_xy,others):
    """64-float observation: self at pixel self_xy; others = [(x,y,hp)] in fighter-id order (ten slots)."""
    v=[self_xy[0]/1000,self_xy[1]/460,1.0,0.0]
    for x,y,hp in (others+[(0,0,0)]*10)[:10]:
        v+=[(x-self_xy[0])/1000,(y-self_xy[1])/460,hp/3,float(hp>0)]
    v+=[0.0]*20
    assert len(v)==64
    return v

SEAT0=(180,156.4);SEAT1=(340,156.4)

class PythonDecoderTests(unittest.TestCase):
    def test_offset_bin_zero_points_at_opponent_from_both_seats(self):
        for me,other,expected in [(SEAT0,SEAT1,(1.0,0.0)),(SEAT1,SEAT0,(-1.0,0.0))]:
            command=decode(1*9+4,'relative',synthetic_observation(me,[other+(3,)]))
            self.assertTrue(command['fire']);self.assertEqual((command['dx'],command['dy']),(0,0))
            self.assertAlmostEqual(command['aimX'],expected[0],places=9);self.assertAlmostEqual(command['aimY'],expected[1],places=9)
    def test_absolute_bin_one_is_east_from_both_seats(self):
        for me,other in [(SEAT0,SEAT1),(SEAT1,SEAT0)]:
            command=decode(1*9+4,'absolute',synthetic_observation(me,[other+(3,)]))
            self.assertAlmostEqual(command['aimX'],1.0);self.assertAlmostEqual(command['aimY'],0.0)
    def test_wraparound(self):
        # Seat 1 (bearing 180 deg) with offset bin 15 (337.5 deg) -> 157.5 deg, equal to offset bin -1 mod 16.
        obs=synthetic_observation(SEAT1,[SEAT0+(3,)])
        command=decode(16*9+4,'relative',obs);angle=math.atan2(command['aimY'],command['aimX'])
        self.assertAlmostEqual((angle-math.radians(157.5)+math.pi)%(2*math.pi)-math.pi,0.0,places=9)
        for aim in range(1,17):
            c=decode(aim*9+4,'relative',obs)
            self.assertAlmostEqual(math.hypot(c['aimX'],c['aimY']),1.0,places=12)
    def test_nearest_live_opponent_skips_dead_and_breaks_ties_by_order(self):
        # Nearer fighter is dead (hp 0): bearing must go to the live one further away, due south.
        obs=synthetic_observation(SEAT0,[(200,156.4,0),(180,400,3)])
        self.assertAlmostEqual(bearing_from_observation(obs),math.pi/2,places=9)
        # Two live fighters at equal distance: the first in id order wins (strict < keeps the first).
        obs=synthetic_observation(SEAT0,[(280,156.4,3),(80,156.4,3)])
        self.assertAlmostEqual(bearing_from_observation(obs),0.0,places=9)
        # No live opponent: bearing 0, i.e. relative equals absolute.
        self.assertEqual(bearing_from_observation(synthetic_observation(SEAT0,[])),0.0)
    def test_move_component_is_absolute_in_both_modes(self):
        obs=synthetic_observation(SEAT1,[SEAT0+(3,)])
        for a in range(ACTIONS):
            r=decode(a,'relative',obs);l=legacy_decode(a)
            self.assertEqual((r['dx'],r['dy'],r['fire']),(l['dx'],l['dy'],l['fire']))
            if not l['fire']:self.assertEqual((r['aimX'],r['aimY']),(l['aimX'],l['aimY']))
    def test_absolute_mode_reproduces_legacy_arithmetic_for_all_actions(self):
        for a in range(ACTIONS):self.assertEqual(decode(a,'absolute'),legacy_decode(a))
    def test_wire_encoding(self):
        self.assertEqual(MODES,('absolute','relative'));self.assertEqual(wire_offset('absolute'),0);self.assertEqual(wire_offset('relative'),153)
        self.assertEqual(wire_action(7,'absolute'),7);self.assertEqual(wire_action(7,'relative'),160)
        with self.assertRaises(ValueError):wire_offset('mirror')
        with self.assertRaises(ValueError):wire_action(153,'absolute')
        with self.assertRaises(ValueError):decode(13,'relative')


def pre_change_pool(site,workers):
    """Pool whose Node workers run the committed pre-change bridge (same canonical site)."""
    source=subprocess.check_output(['git','-C',str(ROOT),'show',PRE_CHANGE_BRIDGE+':arena/bridge.mjs'],text=True)
    directory=tempfile.mkdtemp();old=Path(directory)/'bridge.mjs';old.write_text(source)
    pool=Pool(site,workers,agents=2,cadence='homepage')
    for process in pool.processes:process.stdin.close();process.wait(timeout=10);process.stdout.close()
    pool.processes=[subprocess.Popen(['node','--experimental-strip-types',str(old),str(Path(site).resolve()),'2','homepage'],stdin=subprocess.PIPE,stdout=subprocess.PIPE) for _ in range(workers)]
    return pool

def shot_seen_by(state,game,viewer):
    """Nearest enemy shot direction (vx,vy)/500 observed by `viewer`; None when no shot is visible."""
    row=state['observation'][game,viewer]
    return None if row[48]==0 else (float(row[46]),float(row[47]))

@unittest.skipUnless(SITE,'Set ARENA_SITE to canonical website checkout')
class BridgeTests(unittest.TestCase):
    def test_absolute_wire_actions_reproduce_pre_change_bridge(self):
        # Recorded-style decisions: fixed-seed random model indices for both seats, 48 decisions, 3 worlds.
        new=Pool(SITE,1,agents=2,cadence='homepage');old=pre_change_pool(SITE,1)
        try:
            seeds=[95010000,95010001,95010002];a=new.reset(seeds);b=old.reset(seeds);rng=np.random.default_rng(30)
            for _ in range(48):
                for key in ('observation','alive','reward','finished','winner'):np.testing.assert_array_equal(a[key],b[key])
                actions=np.zeros((3,11),np.int64);actions[:,:2]=rng.integers(ACTIONS,size=(3,2))
                a=new.step(actions);b=old.step(actions)
            for key in ('observation','alive','reward','finished','winner'):np.testing.assert_array_equal(a[key],b[key])
        finally:new.close();old.close()
    def test_relative_bin_one_fires_at_the_opponent_from_either_seat(self):
        pool=Pool(SITE,1,agents=2,cadence='homepage')
        try:
            pool.reset([1,2,3]);actions=np.zeros((3,11),np.int64);actions[:,:2]=4  # stay, no fire
            actions[0,0]=153+13   # seat 0 relative bin 1 -> at seat 1 (east)
            actions[1,1]=153+13   # seat 1 relative bin 1 -> at seat 0 (west)
            actions[2,1]=13       # seat 1 absolute bin 1 -> east, away from seat 0
            state=pool.step(actions)
            vx,vy=shot_seen_by(state,0,1);self.assertAlmostEqual(vx,1.0,places=5);self.assertAlmostEqual(vy,0.0,places=5)
            vx,vy=shot_seen_by(state,1,0);self.assertAlmostEqual(vx,-1.0,places=5);self.assertAlmostEqual(vy,0.0,places=5)
            vx,vy=shot_seen_by(state,2,0);self.assertAlmostEqual(vx,1.0,places=5)
        finally:pool.close()
    def test_bridge_and_python_mirror_agree_on_every_relative_aim_bin(self):
        pool=Pool(SITE,1,agents=2,cadence='homepage')
        try:
            for seat in (0,1):
                for aim in range(1,17):
                    state=pool.reset([7]);observation=state['observation'][0,seat].tolist()
                    actions=np.full((1,11),4,np.int64);actions[0,seat]=153+aim*9+4
                    following=pool.step(actions);seen=shot_seen_by(following,0,1-seat);self.assertIsNotNone(seen)
                    expected=decode(aim*9+4,'relative',observation)
                    self.assertAlmostEqual(seen[0],expected['aimX'],places=5);self.assertAlmostEqual(seen[1],expected['aimY'],places=5)
        finally:pool.close()
    def test_out_of_range_wire_action_rejected(self):
        pool=Pool(SITE,1,agents=2,cadence='homepage')
        try:
            pool.reset([5]);actions=np.full((1,11),4,np.int64);actions[0,0]=306
            with self.assertRaises(RuntimeError):pool.step(actions)
        finally:
            for p in pool.processes:p.kill()
            pool.pool.shutdown()


def tiny_training(output,mode,games=4):
    """Baseline2048 argument pattern reduced to CPU scale (iteration28 smoke_variants), through the external encoder wrapper."""
    trainer=[sys.executable,'-B',str(ROOT/'arena/train_loss.py'),'--site',SITE,'--output',str(output),'--games',str(games),'--parallel','2','--workers','1','--hidden','16','--context','2','--simulations','2','--root-candidates','2','--replay-blocks','3','--batch','4','--unroll','2','--return-mode','episode','--timeout-bootstrap','zero','--discount','1.0','--limit','3','--updates-per-game','1','--agents','2','--policy-floor','0','--noise-fraction','.25','--cadence','homepage','--seed','52731415','--max-seconds','60','--device','cpu','--aim-decoding',mode]
    command=[sys.executable,'-B',str(ROOT/'diagnostics/retention/train_compressed.py'),'--output',str(output),'--seconds','170','--max-encoded-bytes','50000000','--']+trainer
    return subprocess.run(command,capture_output=True,text=True,timeout=200)

@unittest.skipUnless(SITE,'Set ARENA_SITE to canonical website checkout')
class TrainerFlagTests(unittest.TestCase):
    def test_tiny_training_in_both_modes_differs_only_in_the_decoding_key(self):
        with tempfile.TemporaryDirectory() as d:
            manifests={};initials={}
            for mode in MODES:
                out=Path(d)/mode;result=tiny_training(out,mode)
                self.assertEqual(result.returncode,0,result.stderr[-2000:])
                report=json.loads((out/'report.json').read_text());self.assertTrue(report['requested_run_completed']);self.assertEqual(report['games'],4);self.assertEqual(report['optimizer_updates'],4)
                self.assertTrue((out/'encoding-result.json').exists())
                manifests[mode]=json.loads((out/'manifest.json').read_text());initials[mode]=(out/'initial.npz').read_bytes()
                self.assertEqual(manifests[mode]['arguments']['aim_decoding'],mode)
            a,b=manifests['absolute'],manifests['relative']
            self.assertEqual({k:v for k,v in a.items() if k!='arguments'},{k:v for k,v in b.items() if k!='arguments'})
            differing={k for k in a['arguments'] if a['arguments'][k]!=b['arguments'][k]}
            self.assertEqual(differing,{'aim_decoding','output'})
            self.assertEqual(initials['absolute'],initials['relative'],'initial weights depend on seed/config only, not on the decoder')
    def test_default_is_absolute(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'default'
            command=[sys.executable,'-B',str(ROOT/'arena/train_loss.py'),'--site',SITE,'--output',str(out),'--games','2','--parallel','2','--workers','1','--hidden','16','--context','2','--simulations','2','--batch','4','--unroll','2','--limit','2','--device','cpu','--max-seconds','60']
            result=subprocess.run(command,capture_output=True,text=True,timeout=120);self.assertEqual(result.returncode,0,result.stderr[-2000:])
            self.assertEqual(json.loads((out/'manifest.json').read_text())['arguments']['aim_decoding'],'absolute')

if __name__=='__main__':unittest.main()
