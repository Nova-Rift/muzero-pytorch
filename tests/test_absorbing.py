import unittest
import numpy as np
import torch
from arena.train import replay_batch
from model import Config,MuZero

class Tests(unittest.TestCase):
    def block(self,terminal):
        return dict(observation=np.zeros((2,1,64),np.float32),actions=np.array([[2]]),rewards=np.ones((1,1),np.float32),targets=np.array([[1],[0]],np.float32),policies=np.ones((2,1,153),np.float32)/153,valid=np.array([[True],[False]]),absorbing=np.array([[False],[terminal]]),indices=np.array([[0,0]]))

    def test_terminal_extends_zero_values_rewards_without_policy_targets(self):
        batch=replay_batch(self.block(True),np.random.default_rng(9),1,1,3,'cpu')
        _,_,rewards,values,_,policy_mask,value_mask,reward_mask=batch
        torch.testing.assert_close(rewards,torch.tensor([[1.,0.,0.]]))
        torch.testing.assert_close(values,torch.tensor([[1.,0.,0.,0.]]))
        torch.testing.assert_close(policy_mask,torch.tensor([[1.,0.,0.,0.]]))
        torch.testing.assert_close(value_mask,torch.ones(1,4))
        torch.testing.assert_close(reward_mask,torch.ones(1,3))

    def test_cutoff_is_not_an_absorbing_death(self):
        batch=replay_batch(self.block(False),np.random.default_rng(9),1,1,3,'cpu')
        torch.testing.assert_close(batch[-2],torch.tensor([[1.,0.,0.,0.]]))
        torch.testing.assert_close(batch[-1],torch.tensor([[1.,0.,0.]]))

    def test_model_uses_absorbing_value_reward_masks(self):
        torch.set_num_threads(1);torch.manual_seed(7)
        m=MuZero(Config(observations=64,actions=153,hidden=16))
        batch=replay_batch(self.block(True),np.random.default_rng(9),1,1,3,'cpu')
        full=m.loss(*batch).mean()
        skipped=m.loss(*batch[:6]).mean()
        self.assertGreater(full.item(),skipped.item())
        full.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))

if __name__=='__main__':unittest.main()
