"""Paper-style comparison figures from completed frozen tournament batches."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def report(directory):
 protocol=json.loads((directory/'protocol.json').read_text())
 rows=json.loads((directory/'summary.json').read_text())
 ids=[m['id'] for m in protocol['models']];index={name:i for i,name in enumerate(ids)}
 n=len(ids);win=np.full((n,n),np.nan);cut=np.full((n,n),np.nan)
 totals={name:dict(wins=0,games=0,cutoffs=0) for name in ids}
 for row in rows:
  a,b=index[row['a']],index[row['b']];games=row['games']
  win[a,b]=row['wins_a']/games;win[b,a]=row['wins_b']/games
  cut[a,b]=cut[b,a]=row['cutoffs']/games
  for name,key in [(row['a'],'wins_a'),(row['b'],'wins_b')]:
   totals[name]['wins']+=row[key];totals[name]['games']+=games;totals[name]['cutoffs']+=row['cutoffs']
 complete=len(rows)==n*(n-1)//2 and all(r['games']==protocol['games'] for r in rows)
 title='Frozen two-agent comparison' if complete else 'INCOMPLETE frozen comparison — no overall ranking'
 with (directory/'pairwise.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else []);writer.writeheader();writer.writerows(rows)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
 fig,axes=plt.subplots(1,2,figsize=(18,8.5),layout='constrained')
 for ax,data,label,cmap in [(axes[0],win,'Row model wins / all games (cells: %)' ,'viridis'),(axes[1],cut,'Time-limit fraction (cells: %)' ,'magma')]:
  im=ax.imshow(data,vmin=0,vmax=1,cmap=cmap)
  ax.set_xticks(range(n),ids,rotation=65,ha='right');ax.set_yticks(range(n),ids)
  ax.set_title(label)
  for i in range(n):
   for j in range(n):
    if np.isfinite(data[i,j]):ax.text(j,i,f'{100*data[i,j]:.0f}',ha='center',va='center',fontsize=7,color='white' if data[i,j]<.55 else 'black')
  fig.colorbar(im,ax=ax,shrink=.65,label='Fraction')
 fig.suptitle(title+'\n128-game pairs; equal seats; fixed geometry; terminal winner reward only' if protocol['games']==128 else title)
 for suffix in ['pdf','svg','png']:fig.savefig(directory/('comparison.'+suffix),dpi=180)
 plt.close(fig)
 matches={'lana-100':'lana-untrained','lana-1000':'lana-untrained','muzero-100':'muzero-untrained','muzero-1000':'muzero-untrained','dreamer4-100':'dreamer4-untrained','dreamer4-1000':'dreamer4-untrained','independent-10000':'independent8-untrained','selfplay-E1024':'selfplay32-untrained','selfplay-F1024':'selfplay32-untrained'}
 matched=[]
 for trained,initial in matches.items():
  for row in rows:
   if {row['a'],row['b']}=={trained,initial}:
    side='a' if row['a']==trained else 'b';other='b' if side=='a' else 'a'
    matched.append((trained,row['wins_'+side]/row['games'],row['wilson95_'+side],f"{row['wins_'+side]}W / {row['wins_'+other]}L / {row['cutoffs']}T / {row['draws']}D"))
 if matched:
  fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
  for i,(name,rate,bounds,counts) in enumerate(matched):
   ax.errorbar(rate,i,xerr=[[rate-bounds[0]],[bounds[1]-rate]],fmt='o',color='#245d80',capsize=3)
   ax.text(1.02,i,counts,va='center',fontsize=8)
  ax.set_yticks(range(len(matched)),[m[0] for m in matched]);ax.invert_yaxis();ax.set_xlim(0,1);ax.axvline(.5,color='gray',linestyle='--',linewidth=.8)
  ax.set_xlabel('Win fraction against matching initialization; descriptive 95% Wilson interval')
  ax.set_title(('Matched initialization comparisons' if complete else 'INCOMPLETE matched initialization comparisons')+'\nTimeouts earn zero; T = time limit, D = simultaneous draw')
  ax.grid(axis='x',alpha=.2)
  for suffix in ['pdf','svg','png']:fig.savefig(directory/('matched-initializations.'+suffix),dpi=180,bbox_inches='tight')
  plt.close(fig)
 table=[]
 if complete:
  for name,t in totals.items():table.append(dict(model=name,**t,win_rate=t['wins']/t['games'],cutoff_rate=t['cutoffs']/t['games']))
  table.sort(key=lambda t:-t['win_rate'])
 result={'complete':complete,'completed_pairs':len(rows),'expected_pairs':n*(n-1)//2,'games':sum(r['games'] for r in rows),'aggregate':table,'limits':['Exploratory fixed-layout comparison; not a held-out promotion test.','One training run per entry; unequal historical training budgets.','Cutoffs score zero and are reported separately from actual draws.','Win rates are against this fixed opponent collection; no superhuman claim.']}
 (directory/'comparison-report.json').write_text(json.dumps(result,indent=2))
 print(json.dumps(result,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('directory',type=Path);report(p.parse_args().directory)
