"""Registered G checkpoint selection and held-out gates; never tunes on final data."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np

def validate(result,games,seed):
 if result['games']!=games or result['seed']!=seed:raise ValueError('Wrong evaluation games/seed')
 if result['decision']!='search' or result['reference_decision']!='search':raise ValueError('Wrong action selection mode')
 rows=result['records']
 if len(rows)!=games or any(r['seat']!=i%2 for i,r in enumerate(rows)):raise ValueError('Missing or unbalanced seat records')
 counts={'wins':0,'losses':0,'draws':0,'cutoffs':0}
 for r in rows:
  if not r['finished']:key='cutoffs'
  elif r['winner']==r['seat']:key='wins'
  elif r['winner']==1-r['seat']:key='losses'
  elif r['winner']==-1:key='draws'
  else:raise ValueError('Invalid winner')
  counts[key]+=1
 if any(result[k]!=v for k,v in counts.items()):raise ValueError('Totals disagree with records')
 return counts

def development_pass(wins,cutoffs,games):return wins/games>.60 and cutoffs/games<=.20

def wilson(wins,games):
 z=1.96;p=wins/games;den=1+z*z/games
 center=(p+z*z/(2*games))/den;half=z*math.sqrt(p*(1-p)/games+z*z/(4*games*games))/den
 return [center-half,center+half]

def confirmation(result,seed):
 validate(result,200,seed);bounds=wilson(result['wins'],200)
 return {'passed':bounds[0]>.5 and result['cutoffs']/200<=.2,'wilson95':bounds,'games':200,'wins':result['wins'],'cutoffs':result['cutoffs']}

def freeze(run,evaluations,output,seed=41000000):
 run=Path(run);output=Path(output)
 if output.exists():raise ValueError('Selection already frozen; never overwrite')
 report=json.loads((run/'report.json').read_text())
 if report['games']!=8192 or not report['requested_run_completed']:raise ValueError('Requested training run incomplete')
 expected={'checkpoint-2048.npz':2048,'checkpoint-4096.npz':4096,'checkpoint-8192.npz':8192};rows=[];files={}
 for path in evaluations:
  path=Path(path);r=json.loads(path.read_text());validate(r,32,seed)
  name=r['challenger']
  if name not in expected or name in files:raise ValueError('Unexpected/duplicate development checkpoint')
  if r['reference']!='initial.npz':raise ValueError('Wrong initialization reference')
  digest=hashlib.sha256((run/name).read_bytes()).hexdigest()
  if digest!=r['checkpoint_sha256']:raise ValueError('Checkpoint changed after evaluation')
  files[name]=hashlib.sha256(path.read_bytes()).hexdigest();rows.append(r)
 if set(files)!=set(expected):raise ValueError('Not all registered checkpoints evaluated')
 chosen=min(rows,key=lambda r:(-r['wins'],r['cutoffs'],expected[r['challenger']]))
 with np.load(run/chosen['challenger'],allow_pickle=False) as weights:
  if not all(np.isfinite(weights[k]).all() for k in weights.files):raise ValueError('Nonfinite checkpoint')
 result={'checkpoint':chosen['challenger'],'sha256':chosen['checkpoint_sha256'],'rule':'most wins, fewest timeouts, earliest checkpoint','development_seed':seed,'development_scores':[{k:r[k] for k in ('challenger','wins','losses','cutoffs','draws')} for r in rows],'evaluation_sha256':files,'initial_sha256':hashlib.sha256((run/'initial.npz').read_bytes()).hexdigest(),'eligible_for_confirmation':development_pass(chosen['wins'],chosen['cutoffs'],32),'status':'frozen before final tests; no final result implied'}
 with output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--evaluations',nargs='+',required=True);p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=41000000);a=p.parse_args()
 print(json.dumps(freeze(a.run,a.evaluations,a.output,a.seed),indent=2))
