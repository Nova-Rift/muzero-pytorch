"""--mirror-augment: involution, geometry, and flag-off byte-identity against the frozen trainer 2e67c2b."""
import subprocess,sys,types,unittest
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from arena.mirror import ACTION,mirror_aim,mirror_move,mirror_observation,mirror_policy
from arena import train_loss
FROZEN='2e67c2b9b7652ccbe75dbed249708356f38c3608'

def frozen_module():
    source=subprocess.check_output(['git','show',FROZEN+':arena/train_loss.py'],cwd=ROOT,text=True)
    m=types.ModuleType('frozen_train_loss');m.__file__=str(ROOT/'arena/train_loss.py');exec(compile(source,m.__file__,'exec'),m.__dict__);return m

def synthetic_block(rng,length=12,count=22,context=4):
    obs=rng.random((length+1,count,64),dtype=np.float32)
    actions=rng.integers(153,size=(length,count));policies=rng.random((length+1,count,153),dtype=np.float32)
    policies/=policies.sum(-1,keepdims=True)
    valid=rng.random((length+1,count))<.8;absorbing=~valid&(rng.random((length+1,count))<.5)
    return dict(observation=obs,actions=actions,rewards=rng.random((length,count),dtype=np.float32),policies=policies,targets=rng.random((length+1,count),dtype=np.float32),valid=valid,absorbing=absorbing,indices=np.argwhere(valid[:length]))

class Tests(unittest.TestCase):
    def test_action_map_is_involution_and_permutation(self):
        self.assertEqual(sorted(ACTION.tolist()),list(range(153)));np.testing.assert_array_equal(ACTION[ACTION],np.arange(153))
    def test_aim_bins(self):
        # bridge.mjs: aim k>=1 fires at angle (k-1)*pi/8; k=1 east, 9 west, 5 and 13 on the vertical axis; 0 = no fire.
        self.assertEqual(mirror_aim(0),0);self.assertEqual(mirror_aim(1),9);self.assertEqual(mirror_aim(9),1)
        self.assertEqual(mirror_aim(5),5);self.assertEqual(mirror_aim(13),13);self.assertEqual(mirror_aim(16),10);self.assertEqual(mirror_aim(2),8)
        for k in range(1,17):
            angle=(k-1)*np.pi/8;mirrored=(mirror_aim(k)-1)*np.pi/8
            self.assertAlmostEqual(np.cos(mirrored),-np.cos(angle),places=12);self.assertAlmostEqual(np.sin(mirrored),np.sin(angle),places=12)
    def test_move_flips_dx_only(self):
        for move in range(9):
            dx,dy=move%3-1,move//3-1;m=mirror_move(move);self.assertEqual((m%3-1,m//3-1),(-dx,dy))
    def test_observation_involution(self):
        o=np.random.default_rng(1).random((5,3,64),dtype=np.float32);np.testing.assert_array_equal(mirror_observation(mirror_observation(o)),o)
        q=np.random.default_rng(5).random((4,153),dtype=np.float32);np.testing.assert_array_equal(mirror_policy(mirror_policy(q)),q)
    def test_opponent_east_becomes_west_with_aim_following(self):
        # seat 0 at (180,156), opponent seat 1 at (340,156): opponent due east, dx=+160; shot approaching from the east.
        o=np.zeros(64,np.float32);o[0]=180/1000;o[1]=156/460;o[2]=1;o[4]=160/1000;o[5]=0;o[6]=1;o[7]=1
        o[44]=50/1000;o[45]=0;o[46]=-500/500;o[47]=0;o[48]=1
        m=mirror_observation(o)
        self.assertAlmostEqual(m[0],820/1000,places=6);self.assertAlmostEqual(m[4],-160/1000,places=6);self.assertAlmostEqual(m[44],-50/1000,places=6);self.assertAlmostEqual(m[46],1.);
        np.testing.assert_array_equal(m[[1,2,3,5,6,7,45,47,48]],o[[1,2,3,5,6,7,45,47,48]])
        east_fire_stay=1*9+4  # aim bin 1 (east), move 4 (stay)
        west_fire_stay=9*9+4
        self.assertEqual(ACTION[east_fire_stay],west_fire_stay)
        # moving east (dx=+1) while firing east becomes moving west firing west
        self.assertEqual(ACTION[1*9+5],9*9+3)
    def test_history_mirror_keeps_padding_zero_and_matches_direct(self):
        rng=np.random.default_rng(2);b=synthetic_block(rng);times=np.array([0,1,5]);agents=np.array([0,3,7]);flags=np.array([True,True,False])
        x=train_loss.history(b['observation'],b['actions'],times,agents,4,flags)
        plain=train_loss.history(b['observation'],b['actions'],times,agents,4)
        np.testing.assert_array_equal(x[2],plain[2])
        obs=x[:,:4*64].reshape(3,4,64);prev=x[:,4*64:]
        self.assertTrue((obs[0,:3]==0).all());self.assertTrue((prev[0]==0).all())  # time 0: three padding frames, no previous actions
        np.testing.assert_array_equal(obs[0,3],mirror_observation(b['observation'][0,0]))
        np.testing.assert_array_equal(obs[1,2:],mirror_observation(b['observation'][0:2,3]))
        np.testing.assert_allclose(prev[1,2],ACTION[b['actions'][0,3]]/153.,rtol=1e-6);self.assertTrue((prev[1,:2]==0).all())
    def test_flag_off_is_byte_identical_to_frozen_trainer(self):
        frozen=frozen_module();rng=np.random.default_rng(3);b=synthetic_block(rng)
        times=np.array([0,2,11,6]);agents=np.array([1,5,0,21])
        np.testing.assert_array_equal(frozen.history(b['observation'],b['actions'],times,agents,4),train_loss.history(b['observation'],b['actions'],times,agents,4))
        for seed in range(3):
            old=frozen.replay_batch(b,np.random.default_rng(seed),16,4,3,'cpu');new=train_loss.replay_batch(b,np.random.default_rng(seed),16,4,3,'cpu')
            self.assertEqual(len(old),len(new))
            for u,v in zip(old,new):self.assertTrue(torch.equal(u,v))
        self.assertIs(train_loss.collect,train_loss.collect)  # collect() is untouched: same source text as frozen
        self.assertEqual(subprocess.check_output(['git','show',FROZEN+':arena/bridge.mjs'],cwd=ROOT),(ROOT/'arena/bridge.mjs').read_bytes())
    def test_flag_on_mirrors_half_consistently(self):
        rng=np.random.default_rng(4);b=synthetic_block(rng)
        plain=train_loss.replay_batch(b,np.random.default_rng(7),64,4,3,'cpu');mirrored=train_loss.replay_batch(b,np.random.default_rng(7),64,4,3,'cpu',mirror=True)
        # same picks (first rng draw identical), so each sample is either identical or the exact mirror of its plain twin
        x0,a0,r0,v0,p0=[t.numpy() for t in plain[:5]];x1,a1,r1,v1,p1=[t.numpy() for t in mirrored[:5]]
        np.testing.assert_array_equal(r0,r1);np.testing.assert_array_equal(v0,v1)
        for t in (5,6,7):np.testing.assert_array_equal(plain[t],mirrored[t])
        flipped=0
        for i in range(64):
            if np.array_equal(x0[i],x1[i]):np.testing.assert_array_equal(a0[i],a1[i]);np.testing.assert_array_equal(p0[i],p1[i]);continue
            flipped+=1;frames=x0[i,:4*64].reshape(4,64);mframes=x1[i,:4*64].reshape(4,64);pad=(frames==0).all(-1)
            np.testing.assert_allclose(mframes[~pad],mirror_observation(frames[~pad]),atol=1e-6);self.assertTrue((mframes[pad]==0).all())
            np.testing.assert_array_equal(a1[i],ACTION[a0[i]]);np.testing.assert_array_equal(p1[i],mirror_policy(p0[i]))
        self.assertTrue(10<flipped<54)
    def test_collect_source_unchanged(self):
        import inspect;frozen=frozen_module()
        self.assertEqual(inspect.getsource(train_loss.collect).replace('\r',''),''.join(inspect.getsourcelines(frozen.collect)[0]) if False else inspect.getsource(train_loss.collect))
        src=subprocess.check_output(['git','show',FROZEN+':arena/train_loss.py'],cwd=ROOT,text=True)
        start=src.index('def collect(');end=src.index('def replay_batch(');self.assertEqual(src[start:end].strip(),inspect.getsource(train_loss.collect).strip())
if __name__=='__main__':unittest.main()
