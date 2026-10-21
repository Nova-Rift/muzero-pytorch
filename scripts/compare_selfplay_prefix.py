"""Compare an extended self-play run to its preserved shorter-run prefix."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def compare_traces(reference,challenger,through,parallel):
 rows=[]
 for offset in range(0,through,parallel):
  name=f'trace-{offset}.npz';p=Path(reference)/name;q=Path(challenger)/name
  with np.load(p,allow_pickle=False) as x,np.load(q,allow_pickle=False) as y:
   if set(x.files)!=set(y.files):raise ValueError('Different trace fields')
   fields={}
   for key in x.files:
    a,b=x[key],y[key]
    equal=a.shape==b.shape and np.array_equal(a,b)
    fields[key]={'equal':equal,'reference_shape':list(a.shape),'challenger_shape':list(b.shape)}
    if a.shape==b.shape and np.issubdtype(a.dtype,np.number):fields[key]['max_abs_difference']=float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64))))
   rows.append({'file':name,'all_fields_equal':all(v['equal'] for v in fields.values()),'fields':fields})
 return {'all_equal':all(r['all_fields_equal'] for r in rows),'blocks':rows,'scope':'Saved first two games (22 slots) per 128-game collection; not every training trajectory.'}

def compare(reference,challenger,through=2048):
 reference=Path(reference);challenger=Path(challenger)
 a=json.loads((reference/'manifest.json').read_text());b=json.loads((challenger/'manifest.json').read_text())
 differences={k:[a['arguments'].get(k),b['arguments'].get(k)] for k in set(a['arguments'])|set(b['arguments']) if a['arguments'].get(k)!=b['arguments'].get(k)}
 unexpected={k:v for k,v in differences.items() if k not in ('output','games','max_seconds')}
 def rows(root):return [json.loads(line) for line in (root/'metrics.jsonl').read_text().splitlines() if json.loads(line)['games']<=through]
 ar,br=rows(reference),rows(challenger)
 if not ar or not br or ar[-1]['games']!=through or br[-1]['games']!=through:raise ValueError('Requested prefix incomplete')
 counts=['games','natural_terminal','cutoffs','joint_steps','active_transitions','optimizer_updates','training_examples']
 exact=len(ar)==len(br) and all(all(x[k]==y[k] for k in counts) for x,y in zip(ar,br))
 action_exact=len(ar)==len(br) and all(x['batch_seconds']['action_counts']==y['batch_seconds']['action_counts'] for x,y in zip(ar,br))
 p=reference/f'checkpoint-{through}.npz';q=challenger/f'checkpoint-{through}.npz'
 with np.load(p,allow_pickle=False) as x,np.load(q,allow_pickle=False) as y:
  if set(x.files)!=set(y.files):raise ValueError('Different checkpoint structure')
  maximum=max(float(np.max(np.abs(x[k]-y[k]))) for k in x.files)
 result={'through_games':through,'argument_differences':differences,'unexpected_learning_argument_differences':unexpected,'config_equal':a['config']==b['config'],'physics_hashes_equal':a['environment']==b['environment'],'torch_equal':a['torch']==b['torch'],'gpu_equal':a['gpu']==b['gpu'],'initial_equal':sha(reference/'initial.npz')==sha(challenger/'initial.npz'),'prefix_work_counts_equal':exact,'prefix_action_histograms_equal':action_exact,'checkpoint_hash_equal':sha(p)==sha(q),'checkpoint_max_abs_difference':maximum,'reference_checkpoint_sha256':sha(p),'challenger_checkpoint_sha256':sha(q),'loss_max_abs_difference':max(abs(x['loss']-y['loss']) for x,y in zip(ar,br)),'scope':'Prefix repeatability and setup audit, not learning improvement.'}
 result['saved_trace_comparison']=compare_traces(reference,challenger,through,a['arguments']['parallel'])
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('reference',type=Path);p.add_argument('challenger',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
 if a.output.exists():raise ValueError('Preserve earlier prefix audit')
 r=compare(a.reference,a.challenger);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
