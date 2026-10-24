import inspect
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import model
from q_completion_probe import algebra_fixture, loaded_search


class CompletionContractTests(unittest.TestCase):
    def test_source_bound_positive_negative_mixed_and_equal_bounds(self):
        rows = algebra_fixture(ROOT/'batched_search_exploration.py')
        for row, expected in zip(rows, [-1.,2.,.5,0.,0.]):
            self.assertAlmostEqual(row['current'][0][0], expected)
            self.assertEqual(row['reference'][0][0], 0.)
            self.assertEqual(row['current'][0][1],row['reference'][0][1])

    def test_reference_completion_scalar_batch_static_and_independence(self):
        source = inspect.getsource(model.search)
        line = '                q = (q-minimum)/(maximum-minimum) if maximum > minimum else q'
        self.assertEqual(source.count(line),1)
        source = source.replace(line,line+'\n                q = q if child.visits else 0.')
        namespace = dict(vars(model))
        exec(compile(source,'<scalar-reference-completion>','exec'),namespace)
        scalar_search = namespace['search']
        batch_search = loaded_search(ROOT/'batched_search_exploration.py',True)
        torch.set_num_threads(1);torch.manual_seed(18)
        network = model.MuZero(model.Config(observations=3,actions=4,hidden=16,simulations=20))
        observations = torch.randn(5,3)
        legal = torch.tensor([[True,True,False,True]]*5)
        policy,value = batch_search(network,observations,legal=legal)
        for i in range(5):
            expected,sv = scalar_search(network,observations[i],legal=legal[i])
            torch.testing.assert_close(policy[i],expected)
            self.assertAlmostEqual(float(value[i]),sv,places=3)
        static,sv = batch_search(network,observations,legal=legal,static=True)
        torch.testing.assert_close(static,policy);torch.testing.assert_close(sv,value)
        single,_ = batch_search(network,observations[2:3],legal=legal[2:3])
        torch.testing.assert_close(single[0],policy[2])


if __name__ == '__main__':
    unittest.main()
