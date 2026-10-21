import unittest
import numpy as np
import torch
from arena.history import DeviceHistory
from arena.train import history

class Tests(unittest.TestCase):
    def test_padding_actions_and_rollover_match_replay_history(self):
        rng=np.random.default_rng(72)
        observations=rng.standard_normal((80,3,64)).astype(np.float32)
        actions=rng.integers(153,size=(80,3))
        for context in [1,4,64]:
            cache=DeviceHistory(3,context,'cpu')
            for t in range(80):
                actual=cache.observe(observations[t])
                expected=history(observations,actions,np.full(3,t),np.arange(3),context)
                torch.testing.assert_close(actual,torch.from_numpy(expected),atol=1e-7,rtol=1e-7)
                cache.acted(actions[t])

if __name__=='__main__':unittest.main()
