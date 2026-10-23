import copy,unittest
from evaluate_split import summarize
class Tests(unittest.TestCase):
 def rows(self):return [dict(seed=i,seat_a=i%2,winner=w,finished=f,steps=s) for i,(w,f,s) in enumerate([(0,True,2),(0,True,3),(-1,True,4),(-1,False,8)])]
 def test_all_outcomes(self):self.assertEqual(summarize(self.rows(),list(range(4)),8),dict(games=4,wins=1,opponent_wins=1,draws=1,timeouts=1))
 def test_reject_malformed(self):
  for field,value in [('seed',99),('seat_a',1),('winner',2),('finished',1),('steps',0)]:
   rows=self.rows();rows[0][field]=value
   with self.assertRaises(ValueError):summarize(rows,list(range(4)),8)
 def test_incomplete_and_cutoff(self):
  with self.assertRaises(ValueError):summarize(self.rows()[:3],list(range(4)),8)
  rows=self.rows();rows[-1]['steps']=7
  with self.assertRaises(ValueError):summarize(rows,list(range(4)),8)
if __name__=='__main__':unittest.main()
