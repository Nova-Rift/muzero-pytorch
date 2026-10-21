import unittest
from tournament.run import wilson
class TournamentStatisticsTests(unittest.TestCase):
 def test_zero_wins_still_has_uncertainty(self):
  lower,upper=wilson(0,128);self.assertAlmostEqual(lower,0);self.assertGreater(upper,0)
 def test_complement_intervals(self):
  a=wilson(18,32);b=wilson(14,32)
  self.assertAlmostEqual(a[0],1-b[1]);self.assertAlmostEqual(a[1],1-b[0])
