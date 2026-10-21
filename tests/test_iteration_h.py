import unittest
from scripts.run_iteration_h import select,dev_pass,wilson
class Tests(unittest.TestCase):
 def row(self,e,w,t=0):return {'checkpoint':e,'results':[{'wins':v,'games':64,'cutoffs':t} for v in w]}
 def test_weakest_opponent_controls_selection(self):
  a=self.row(512,[60,60,20]);b=self.row(1024,[35,35,35]);self.assertIs(select([a,b]),b)
 def test_ties_and_gates(self):
  a=self.row(512,[40]*3);b=self.row(1024,[40]*3)
  self.assertIs(select([b,a]),a);self.assertTrue(dev_pass(a))
  self.assertFalse(dev_pass(self.row(512,[40,40,38])))
  self.assertFalse(dev_pass(self.row(512,[40]*3,13)))
 def test_confirmation(self):
  self.assertLess(wilson(110,200),.5);self.assertGreater(wilson(130,200),.5)
if __name__=='__main__':unittest.main()
