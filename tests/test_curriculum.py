import unittest,os
from pathlib import Path
import numpy as np
from arena.environment import Pool
SITE=os.environ.get('ARENA_SITE')
@unittest.skipUnless(SITE,'Set ARENA_SITE to canonical website checkout')
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

 def test_homepage_cadence_exactly_five_frames(self):
  p=Pool(SITE,agents=2,cadence='homepage')
  try:
   state=p.reset([41000002]);start=state['observation'][0,0,0]
   actions=np.full((1,11),4);actions[0,0]=5
   state=p.step(actions)
   self.assertAlmostEqual(float((state['observation'][0,0,0]-start)*1000),55,places=4)
  finally:p.close()
