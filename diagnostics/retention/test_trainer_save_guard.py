"""Execute actual pinned trainer save closure; no model/environment construction."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
SOURCE=Path(__file__).resolve().parents[2]/'arena/train_selfplay.py'

def closure(save):
    tree=ast.parse(SOURCE.read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='save_state')
    factory=ast.FunctionDef(name='factory',args=ast.arguments(posonlyargs=[],args=[],kwonlyargs=[],kw_defaults=[],defaults=[]),
        body=[ast.parse('last_saved = None').body[0],node,ast.Return(value=ast.Name(id='save_state',ctx=ast.Load()))],decorator_list=[])
    module=ast.fix_missing_locations(ast.Module(body=[factory],type_ignores=[]))
    totals={'games':512,'optimizer_updates':512}
    ns=dict(arguments={},source={},provenance={'files':{}},torch=torch,initial={},
        model=SimpleNamespace(state_dict=lambda:{}),optimizer=SimpleNamespace(state_dict=lambda:{}),
        totals=totals,replay=[],rng=SimpleNamespace(bit_generator=SimpleNamespace(state={})),
        a=SimpleNamespace(device='cpu'),save_milestone=save,out=None)
    exec(compile(module,str(SOURCE),'exec'),ns)
    return ns['factory'](),totals

class SaveGuard(unittest.TestCase):
    def test_exact_boundary_final_save_deduplicated(self):
        seen=[]
        save,totals=closure(lambda state,*args:seen.append(dict(state['totals'])))
        save(); save()  # periodic512, unconditional final512
        self.assertEqual(len(seen),1)
        totals.update(games=2048,optimizer_updates=2048)
        save(); save()
        self.assertEqual([r['games'] for r in seen],[512,2048])
    def test_failed_save_not_marked_success(self):
        calls=[]
        def failing(*args):
            calls.append(1)
            if len(calls)==1: raise OSError('injected write failure')
        save,_=closure(failing)
        with self.assertRaises(OSError): save()
        save(); save()
        self.assertEqual(len(calls),2)
if __name__=='__main__': unittest.main()
