"""Source-bound restart gate. MIR compares available archived first collection; EGO initial only."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from first_block_gate import bind_archive,equal_arrays,compare_block,ARCHIVE
from evaluate_endpoint import validate_declared

def install_gate(trainer,source,output,baseline,contract,variant):
 original=trainer.collect;checked=[]
 def collect(*args,**kwargs):
  if checked:return original(*args,**kwargs)
  manifest=json.loads((output/'manifest.json').read_text())
  declared='aim_decoding' if variant=='ego' else 'mirror_augment'
  validate_declared(manifest,baseline,source,[declared],contract)
  old=json.loads((baseline/'manifest.json').read_text())
  if not manifest.get('gpu') or manifest['gpu']!=old.get('gpu'):raise ValueError('Archived GPU runtime differs')
  with np.load(output/'initial.npz',allow_pickle=False) as a,np.load(baseline/'initial.npz',allow_pickle=False) as b:
   initial=equal_arrays(a,b)
  # Exact archive bytes additionally match the downstream evaluator initialization rule.
  sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
  if sha(output/'initial.npz')!=sha(baseline/'initial.npz'):raise ValueError('Initial checkpoint bytes differ')
  result=dict(variant=variant,initial_tensors=initial,initial_sha256=sha(output/'initial.npz'),checked_before_first_update=True,source_contract=contract,archive_sha256=ARCHIVE,diagnostic_sha256=sha(Path(__file__)),collection_parity_required=variant=='mir')
  block,stats=original(*args,**kwargs)
  if variant=='mir':result.update(compare_block(block,stats,baseline))
  else:result.update(full_128_trajectory_parity=False,collection_parity_applicability='inapplicable: relative aim changes actions')
  result['complete']=True
  (output/'variant-collection-gate.json').write_text(json.dumps(result,indent=2)+'\n');checked.append(result)
  return block,stats
 trainer.collect=collect
 return checked

def main():
 p=argparse.ArgumentParser()
 for key in ('source','output','baseline','source-contract'):p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--variant',choices=['ego','mir'],required=True);p.add_argument('trainer_args',nargs=argparse.REMAINDER);a=p.parse_args()
 bind_archive(a.baseline);source=a.source.resolve();sys.path.insert(0,str(source))
 from arena import train_loss
 if Path(train_loss.__file__).resolve()!=source/'arena/train_loss.py':raise ValueError('Wrong trainer import')
 command=a.trainer_args[1:] if a.trainer_args[:1]==['--'] else a.trainer_args
 if any(v=='--output' or v.startswith('--output=') for v in command):raise ValueError('Supply output only to gate wrapper')
 checked=install_gate(train_loss,source,a.output,a.baseline,json.loads(a.source_contract.read_text()),a.variant)
 sys.argv=[str(source/'arena/train_loss.py'),'--output',str(a.output),*command];train_loss.main()
 if not checked:raise ValueError('No successful variant collection gate')
if __name__=='__main__':main()
