import json,os,subprocess,unittest
import numpy as np
from arena.environment import Pool

@unittest.skipUnless(os.environ.get('ARENA_SITE'),'Set ARENA_SITE to canonical website checkout')
class Tests(unittest.TestCase):
    def test_binary_bridge_matches_existing_json_bridge(self):
        site=os.environ['ARENA_SITE'];pool=Pool(site,2)
        old=subprocess.Popen(['node','--experimental-strip-types','scripts/arena-parallel-bridge.mjs'],cwd=site,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
        def request(data):
            old.stdin.write(json.dumps(data)+'\n');old.stdin.flush();return json.loads(old.stdout.readline())['worlds']
        try:
            state=pool.reset([27000000,27000001]);expected=request(dict(type='reset',seeds=[27000000,27000001]))
            rng=np.random.default_rng(67)
            for step in range(256):
                np.testing.assert_array_equal(state['observation'],np.asarray([x['observations'] for x in expected],np.float32))
                np.testing.assert_array_equal(state['alive'],[x['alive'] for x in expected])
                np.testing.assert_array_equal(state['reward'],[x['rewards'] for x in expected])
                np.testing.assert_array_equal(state['finished'],[x['finished'] for x in expected])
                actions=rng.integers(153,size=(2,11))
                state=pool.step(actions);expected=request(dict(type='step',actions=actions.tolist()))
        finally:
            pool.close();old.stdin.close();old.wait(timeout=10);old.stdout.close()

if __name__=='__main__':unittest.main()
