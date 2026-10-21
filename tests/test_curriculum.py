import unittest
from pathlib import Path
import numpy as np
from arena.environment import Pool
SITE=Path('/home/clark/research-lab/code/sites-trial')
class Curriculum(unittest.TestCase):
 def test_two_agents_and_legacy(self):
  for count in (2,3,11):
   p=Pool(SITE,agents=count)
   try:
    state=p.reset([41000001])
    self.assertEqual(state['alive'].sum(),count)
    self.assertEqual(state['observation'].shape,(1,11,64))
    for _ in range(5):state=p.step(np.full((1,11),4))
    self.assertEqual(state['alive'].sum(),count)
    self.assertEqual(float(state['reward'].sum()),0)
   finally:p.close()
