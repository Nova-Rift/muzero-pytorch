import unittest
import torch
from replay_census import census_block


def fixture():
    valid=torch.tensor([[1,1,0],[1,0,0],[0,0,0]],dtype=torch.bool)
    absorbing=torch.tensor([[0,0,0],[0,1,1],[1,1,1]],dtype=torch.bool)
    policies=torch.full((3,3,2),99.) # Invalid padded policies must not enter census.
    policies[0,0]=torch.tensor([1.,0.]);policies[0,1]=.5;policies[1,0]=torch.tensor([.25,.75])
    return dict(valid=valid,absorbing=absorbing,policies=policies,
                actions=torch.zeros(2,3,dtype=torch.long),indices=valid[:-1].nonzero(),
                targets=torch.tensor([[1.,0.,99.],[1.,0.,99.],[0.,0.,99.]]),
                rewards=torch.tensor([[0.,0.,0.],[1.,0.,0.]]))


class CensusTest(unittest.TestCase):
    def test_mask_units_and_absorbing_padding(self):
        out=census_block(fixture(),1,3)
        self.assertEqual(out['active_player_transitions'],3)
        self.assertEqual(out['games_with_valid_decisions'],1)
        self.assertEqual(out['player_trajectories'],2)
        self.assertEqual(out['policy']['support_histogram'],{'1':1,'2':2})
        self.assertAlmostEqual(out['policy']['maximum_mass']['mean'],.75)
        self.assertEqual(out['active_absorbing_value_targets']['maximum'],0.)
        self.assertEqual(out['consistency']['active_absorbing_rows'],3)
        self.assertEqual(out['consistency']['raw_absorbing_rows'],5)
        self.assertTrue(out['consistency']['indices_equal_valid_nonzero'])
    def test_chunk_equivalence(self):
        self.assertEqual(census_block(fixture(),1,3),census_block(fixture(),100,3))
    def test_corruption_reported(self):
        b=fixture();b['policies'][0,0]=torch.tensor([1.2,-.1]);b['rewards'][0,2]=1
        b['indices']=b['indices'].flip(0)
        out=census_block(b,2,3)
        self.assertEqual(out['policy']['negative_elements'],1)
        self.assertEqual(out['policy']['unnormalized_rows_at_1e_minus_6'],1)
        self.assertEqual(out['consistency']['invalid_nonzero_rewards'],1)
        self.assertFalse(out['consistency']['indices_equal_valid_nonzero'])
    def test_bad_layout(self):
        with self.assertRaises(ValueError):census_block(fixture(),1,2)


if __name__=='__main__':unittest.main()
