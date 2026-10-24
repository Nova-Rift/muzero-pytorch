from pathlib import Path
import argparse
import tempfile
import unittest
from unittest.mock import patch
import sys
import torch
from model import Config, MuZero
from arena.first_transition import FirstTransitionMuZero, model_class
from arena.rootweighted import RootWeightedMuZero
from arena import train_first_transition, train_loss
from diagnostics.real_gradients.probe import gradients

ROOT = Path(__file__).resolve().parents[1]

class FirstTransitionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)
        self.base = MuZero(Config(observations=4, actions=3, hidden=8, support=2)).double()
        self.new = FirstTransitionMuZero(self.base.config).double()
        self.new.load_state_dict(self.base.state_dict())
        self.batch = (torch.randn(3,4,dtype=torch.float64), torch.randint(3,(3,5)),
                      torch.randn(3,5,dtype=torch.float64), torch.randn(3,6,dtype=torch.float64),
                      torch.randn(3,6,3,dtype=torch.float64).softmax(-1),
                      torch.ones(3,6,dtype=torch.float64),torch.ones(3,6,dtype=torch.float64),torch.ones(3,5,dtype=torch.float64))

    def compare(self, ratio):
        a,b = self.base.loss(*self.batch), self.new.loss(*self.batch)
        self.assertTrue(torch.equal(a,b))
        ga,gb = gradients(a.sum(),self.base), gradients(b.sum(),self.new)
        nonzero = False
        for name in ga:
            factor = ratio if name.startswith('representation.') else 1
            torch.testing.assert_close(gb[name],ga[name]*factor,rtol=0,atol=0)
            if name.startswith('representation.') and ga[name].abs().sum(): nonzero = True
        self.assertTrue(nonzero)

    def test_exact_forward_dense_sparse_and_default_masks(self):
        for sparse in (False,True):
            if sparse:
                self.batch[5][1,1::2]=0;self.batch[6][2,::2]=0;self.batch[7][0,2:]=0
            self.assertTrue(torch.equal(self.base.loss(*self.batch),self.new.loss(*self.batch)))
        self.assertTrue(torch.equal(self.base.loss(*self.batch[:5]),self.new.loss(*self.batch[:5])))
        self.assertEqual(set(self.base.state_dict()),set(self.new.state_dict()))

    def test_root_only_all_parameter_gradients_unchanged(self):
        self.batch[5][:,1:]=0;self.batch[6][:,1:]=0;self.batch[7].zero_()
        self.compare(1)

    def test_isolated_recurrent_representation_double_heads_dynamics_same(self):
        for head in (5,6,7):
            for depth in range(5):
                for mask in self.batch[5:]: mask.zero_()
                self.batch[head][:,depth if head==7 else depth+1]=1
                self.compare(2)

    def test_later_transition_attenuation_same(self):
        class Chain:
            config=Config(observations=1,actions=2,hidden=1,support=1)
            def initial(self,x): return x,torch.cat((x,torch.zeros_like(x)),-1),x.expand(-1,3)*0
            def recurrent(self,x,a):
                h,p,v=self.initial(x)
                return h,v,p,v
        for depth in range(6):
            x=torch.zeros(1,1,dtype=torch.float64,requires_grad=True)
            mask=torch.zeros(1,6,dtype=torch.float64);mask[:,depth]=1
            b=(x,torch.zeros(1,5,dtype=torch.long),torch.zeros(1,5),torch.zeros(1,6),torch.tensor([[[1.,0.]]*6]),mask,torch.zeros_like(mask),torch.zeros(1,5))
            grad=torch.autograd.grad(FirstTransitionMuZero.loss(Chain(),*b).sum(),x)[0]
            self.assertAlmostEqual(float(grad),-.5*.5**max(0,depth-1)/6,places=12)

    def test_sparse_rewards_align_to_preceding_action_and_empty_masks(self):
        for mask in self.batch[5:]: mask.zero_()
        self.batch[7][1,3]=1
        self.compare(2)
        changed=list(self.batch);changed[2]=self.batch[2].clone();changed[2][:,:3]+=100;changed[2][:,4]+=100
        self.assertTrue(torch.equal(self.new.loss(*self.batch),self.new.loss(*changed)))
        self.batch[7].zero_()
        self.assertTrue(torch.equal(self.new.loss(*self.batch),torch.zeros(3,dtype=torch.float64)))

    def test_only_intended_loss_and_trainer_differences(self):
        import inspect
        a=inspect.getsource(FirstTransitionMuZero.loss).replace('1.0 if t == 0 else .5','.5')
        self.assertEqual(a,inspect.getsource(MuZero.loss))
        a=(ROOT/'arena/train_first_transition.py').read_text().replace('from arena.first_transition import model_class','from arena.rootweighted import model_class').replace("['baseline','root5-mass-v1','first-transition1-v1']","['baseline','root5-mass-v1']")
        self.assertEqual(a,(ROOT/'arena/train_loss.py').read_text())
        self.assertIs(train_first_transition.search_batch,train_loss.search_batch)
        for contract,cls in [('baseline',MuZero),('root5-mass-v1',RootWeightedMuZero),('first-transition1-v1',FirstTransitionMuZero)]:
            self.assertIs(model_class(contract),cls)

    def test_fixed_entrypoint_dispatch_and_override_rejection(self):
        import runpy
        with patch.object(sys,'argv',['entry']), patch.object(train_first_transition,'main') as main:
            runpy.run_path(str(ROOT/'arena/train_first_transition_fixed.py'),run_name='__main__')
            self.assertEqual(sys.argv[-2:],['--loss-contract','first-transition1-v1'])
            main.assert_called_once()
        for arg in ('--loss-contract','--loss-contract=baseline'):
            with patch.object(sys,'argv',['entry',arg]):
                with self.assertRaisesRegex(ValueError,'fixes first-transition1-v1'):
                    runpy.run_path(str(ROOT/'arena/train_first_transition_fixed.py'),run_name='__main__')

    def test_every_cross_contract_resume_rejected(self):
        original=argparse.ArgumentParser.parse_args
        captured={}
        def parse(parser,*a,**kw):
            result=original(parser,*a,**kw);captured.update(vars(result));return result
        contracts=('baseline','root5-mass-v1','first-transition1-v1')
        for requested in contracts:
            for saved_contract in contracts:
                if requested==saved_contract: continue
                def restored(*a,**kw):
                    saved=dict(captured);saved['loss_contract']=saved_contract
                    return {'arguments':saved}
                with tempfile.TemporaryDirectory() as tmp:
                    args=['trainer','--site',tmp,'--output',tmp+'/out','--loss-contract',requested,'--device','cpu','--hidden','8','--context','2','--resume-state',tmp+'/state.pt']
                    with patch.object(sys,'argv',args),patch.object(argparse.ArgumentParser,'parse_args',parse),patch.object(torch,'load',restored):
                        with self.assertRaisesRegex(ValueError,'Resume setting changed: loss_contract'):
                            train_first_transition.main()

if __name__=='__main__': unittest.main()
