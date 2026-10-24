import unittest
from check_contract import check

class Contracts(unittest.TestCase):
    def test_all_intended_contracts(self):
        for contract in ('baseline','root5-mass-v1','first-transition1-v1'):
            a={'arguments':{'loss_contract':contract,'seed':52731415}}
            self.assertTrue(check(a,a,contract))
    def test_consistent_but_wrong_arm_rejected(self):
        wrong={'arguments':{'loss_contract':'baseline'}}
        with self.assertRaises(ValueError):check(wrong,wrong,'root5-mass-v1')
    def test_first_transition_wrong_but_consistent_rejected(self):
        for intended,actual in [('first-transition1-v1','baseline'),('first-transition1-v1','root5-mass-v1'),('baseline','first-transition1-v1'),('root5-mass-v1','first-transition1-v1')]:
            with self.subTest(intended=intended,actual=actual):
                wrong={'arguments':{'loss_contract':actual}}
                with self.assertRaisesRegex(ValueError,'Manifest objective differs'):
                    check(wrong,wrong,intended)
    def test_missing_serialized_contract_rejected(self):
        with self.assertRaises(KeyError):check({'arguments':{}},{'arguments':{}},'baseline')
    def test_checkpoint_contract_changed_rejected(self):
        with self.assertRaises(ValueError):check({'arguments':{'loss_contract':'baseline'}},{'arguments':{'loss_contract':'root5-mass-v1'}},'baseline')
    def test_other_arguments_changed_rejected(self):
        with self.assertRaises(ValueError):check({'arguments':{'loss_contract':'baseline','seed':1}},{'arguments':{'loss_contract':'baseline','seed':2}},'baseline')
if __name__=='__main__':unittest.main()
