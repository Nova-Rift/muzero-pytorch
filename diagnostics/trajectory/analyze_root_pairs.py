"""Frozen paired-seed descriptive comparison; no promotion or tuning."""
import json,random,hashlib
from pathlib import Path
import argparse
p=argparse.ArgumentParser();p.add_argument('--run-directory',type=Path,required=True);a=p.parse_args();r=a.run_directory;out={}
for opponent in ['E1024','F1024']:
 groups=[]
 for candidate in ['M2048-root8','M2048-root16']:
  d=r/'artifacts/loop-a'/f'{candidate}-{opponent}';files=list(d.glob('*__*.json'));assert len(files)==1
  p=json.loads((d/'protocol.json').read_text());assert p['games']==64 and p['seed']==91520000 and p['limit']==256
  groups.append(json.loads(files[0].read_text())['records'])
 assert len(groups[0])==len(groups[1])==64
 delta=[];tables={}
 for x,y in zip(*groups):
  assert (x['seed'],x['seat_a'])==(y['seed'],y['seat_a'])
  delta.append(int(y['finished'] and y['winner']==y['seat_a'])-int(x['finished'] and x['winner']==x['seat_a']))
 for name,rows in zip(['M2048-root8','M2048-root16'],groups):
  tables[name]={str(seat):{'games':sum(x['seat_a']==seat for x in rows),'wins':sum(x['seat_a']==seat and x['finished'] and x['winner']==seat for x in rows),'cutoffs':sum(x['seat_a']==seat and not x['finished'] for x in rows)} for seat in [0,1]}
 rng=random.Random(150001);boot=sorted(sum(rng.choices(delta,k=64))/64 for _ in range(10000))
 out[opponent]={'games_per_candidate':64,'win_fraction_delta_root16_minus_root8':sum(delta)/64,'paired_seed_bootstrap95':[boot[249],boot[9749]],'discordant_gains':delta.count(1),'discordant_losses':delta.count(-1),'by_seat':tables}
report={'iteration':16,'scope':'Development diagnosis; paired-seed resampling includes ties/cutoffs as non-wins; one training seed, no promotion or generalization claim.','comparisons':out}
(r/'paired-analysis.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
