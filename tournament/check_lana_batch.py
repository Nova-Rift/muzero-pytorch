"""Real-checkpoint CPU parity/contract checks for opt-in Lana microbatching."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tournament.controllers import load_model,controller

def main():
 p=argparse.ArgumentParser();p.add_argument('--catalog',type=Path,required=True);p.add_argument('--site',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise ValueError('Preserve evidence')
 torch.set_num_threads(1);entry=next(e for e in json.loads(a.catalog.read_text())['models'] if e['id']=='lana-100')
 assert hashlib.sha256(Path(entry['checkpoint']).read_bytes()).hexdigest()==entry['sha256']
 model=load_model(entry,a.site,'cpu')
 def digest():return hashlib.sha256(b''.join(t.detach().numpy().tobytes() for t in model.state_dict().values())).hexdigest()
 before=digest();labels=[(1000+i,i%2) for i in range(65)];old=controller(entry,model,labels,a.site,32);new=controller(entry,model,labels,a.site,128);rng=np.random.default_rng(13)
 for step in range(4):
  obs=rng.normal(size=(65,64)).tolist();alive=[i%7!=0 for i in range(65)]
  assert old.decide(obs,alive)==new.decide(obs,alive)
 from survival_agent.hybrid.model import TransformerLatent
 state=new.state
 def rejected(s):
  try:model.world_model._check_state(s)
  except ValueError:return
  raise AssertionError('Invalid latent accepted')
 bad=state.clone();bad.deter[40,0]=float('nan');rejected(bad)
 bad=state.clone();bad.lengths[40]=65;rejected(bad)
 bad=TransformerLatent(state.deter,state.stoch[:-1],state.history,state.lengths);rejected(bad)
 tensors=[state.deter,state.stoch,state.history,state.lengths]
 rejected(TransformerLatent(*(x[:1].repeat(129,*([1]*(x.ndim-1))) for x in tensors)))
 model.train();rejected(state);model.eval();assert before==digest()
 report={'checkpoint_sha256':entry['sha256'],'parameter_buffer_bytes_unchanged':True,'streams':65,'decisions':4,'actions_exact':True,'invalid_state_cases_rejected':['NaN after first32rows','invalid history length','mismatched leading dimensions','batch129 exceeds128bound','training mode'],'torch':str(torch.__version__),'scope':'CPU numerical/contract check; GPU game parity and speed not measured'}
 a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
