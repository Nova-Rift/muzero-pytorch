"""Independently recount saved games and verify the frozen evaluation payload."""
import argparse,hashlib,itertools,json,math
from pathlib import Path

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def audit(directory,stage=None):
 directory=Path(directory);protocol=json.loads((directory/'protocol.json').read_text());models=protocol['models'];names=[m['id'] for m in models]
 if len(set(names))!=len(names):raise ValueError('Duplicate model ids')
 games=protocol['games'];parallel=protocol['parallel'];seed=protocol['seed'];limit=protocol['limit']
 summaries=json.loads((directory/'summary.json').read_text());by_pair={}
 for r in summaries:
  key=(r['a'],r['b'])
  if key in by_pair:raise ValueError('Duplicate pair summary')
  by_pair[key]=r
 completed=0;records_total=0;files={}
 pairs=list(itertools.combinations(names,2))
 if not set(by_pair)<=set(pairs):raise ValueError('Unknown/reversed pair summary')
 for index,(a,b) in enumerate(pairs):
  records=[];seconds=0;all_batches=True
  for offset in range(0,games,parallel):
   path=directory/f'{a}__{b}-{offset}.json'
   if not path.exists():all_batches=False;continue
   files[path.name]=sha(path);batch=json.loads(path.read_text());rows=batch['records'];count=min(parallel,games-offset)
   if len(rows)!=count:raise ValueError('Wrong batch length')
   for i,r in enumerate(rows):
    if r['seed']!=seed+index*100000+offset+i or r['seat_a']!=(offset+i)%2:raise ValueError('Seed/seat mismatch')
    if r['winner'] not in (-1,0,1) or not 1<=r['steps']<=limit:raise ValueError('Invalid outcome or length')
    if not r['finished'] and (r['steps']!=limit or r['winner']!=-1):raise ValueError('Invalid time-limit outcome')
   for name in ['setup_seconds','rollout_seconds','inference_seconds','environment_seconds']:
    if not math.isfinite(batch[name]) or batch[name]<0:raise ValueError('Invalid timing')
   seconds+=batch['setup_seconds']+batch['rollout_seconds'];records+=rows
  records_total+=len(records)
  if not all_batches:
   if (a,b) in by_pair:raise ValueError('Summary claims incomplete pair')
   continue
  if (a,b) not in by_pair:raise ValueError('Completed pair missing summary')
  r=by_pair[a,b];wins=sum(x['finished'] and x['winner']==x['seat_a'] for x in records);losses=sum(x['finished'] and x['winner']==1-x['seat_a'] for x in records);cutoffs=sum(not x['finished'] for x in records)
  expected={'games':games,'wins_a':wins,'wins_b':losses,'cutoffs':cutoffs,'draws':games-wins-losses-cutoffs}
  if any(r[k]!=v for k,v in expected.items()):raise ValueError('Summary disagrees with raw records')
  if abs(r['seconds']-seconds)>1e-6 or abs(r['games_per_second']-games/seconds)>1e-6:raise ValueError('Timing summary disagrees')
  completed+=1
 payload=None
 if stage is not None:
  stage=Path(stage);manifest=json.loads((stage/'cache-manifest.json').read_text())
  for name,digest in manifest.items():
   path=(stage/name).resolve()
   if not path.is_relative_to(stage.resolve()) or sha(path)!=digest:raise ValueError('Staged payload changed: '+name)
  checks={'catalog.json':protocol['catalog_sha256'],'model/tournament/run.py':protocol['source_sha256'],'model/tournament/controllers.py':protocol['controller_sha256']}
  if 'lana_batch_adapter_sha256' in protocol:checks['model/tournament/lana_batch.py']=protocol['lana_batch_adapter_sha256']
  checks.update({'site/'+n:h for n,h in protocol['environment'].items()});checks.update({'site/'+n:h for n,h in protocol['legacy_sources'].items()})
  checks.update({'weights/'+m['id']+'.npz':m['sha256'] for m in models})
  for name,digest in checks.items():
   if sha(stage/name)!=digest:raise ValueError('Protocol payload mismatch: '+name)
  payload={'manifest_sha256':sha(stage/'cache-manifest.json'),'verified_files':len(manifest),'protocol_checks':len(checks)}
 return {'complete':completed==len(pairs),'completed_pairs':completed,'expected_pairs':len(pairs),'validated_game_records':records_total,'protocol_sha256':sha(directory/'protocol.json'),'batch_sha256':files,'payload':payload,'scope':'Record/seed/seat/timing/source integrity audit; not independent gameplay reproduction or a learning-success gate.'}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--stage',type=Path);a=p.parse_args();r=audit(a.directory,a.stage);(a.directory/'audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='batch_sha256'},indent=2))
