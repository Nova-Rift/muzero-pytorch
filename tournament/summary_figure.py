"""Compact result and measured-throughput summary from frozen public data."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1];out=root/'docs/tournament-001-results'
rows=json.loads((out/'summary.json').read_text());speed=json.loads((root/'docs/tournament-benchmark-001.json').read_text())
matched=[('Lana · 100','lana-100','lana-untrained'),('Lana · 1,000','lana-1000','lana-untrained'),('Legacy MuZero · 100','muzero-100','muzero-untrained'),('Legacy MuZero · 1,000','muzero-1000','muzero-untrained'),('Unofficial Dreamer4 · 100','dreamer4-100','dreamer4-untrained'),('Unofficial Dreamer4 · 1,000','dreamer4-1000','dreamer4-untrained'),('Old independent MuZero · 10,000','independent-10000','independent8-untrained'),('New MuZero · E1024','selfplay-E1024','selfplay32-untrained'),('New MuZero · F1024','selfplay-F1024','selfplay32-untrained')]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
fig,axes=plt.subplots(2,1,figsize=(12,10),gridspec_kw={'height_ratios':[2.2,1]},layout='constrained')
ax=axes[0]
for i,(label,trained,initial) in enumerate(matched):
 r=next(r for r in rows if {r['a'],r['b']}=={trained,initial});side='a' if r['a']==trained else 'b';other='b' if side=='a' else 'a';rate=100*r['wins_'+side]/128;low,high=np.array(r['wilson95_'+side])*100
 ax.errorbar(rate,i,xerr=[[rate-low],[high-rate]],fmt='o',capsize=4,color='#196c82')
 ax.text(103,i,f"{r['wins_'+side]} / {r['wins_'+other]} / {r['cutoffs']} / {r['draws']}",va='center',fontsize=9)
ax.text(103,-.8,'Win / loss / timeout / draw',fontsize=9,fontweight='bold')
ax.set_yticks(range(len(matched)),[x[0] for x in matched]);ax.invert_yaxis();ax.set_xlim(0,100);ax.set_ylim(8.6,-1.3);ax.axvline(50,color='#777777',ls='--',lw=1);ax.grid(axis='x',alpha=.2)
ax.set_xlabel('Wins against matching untrained model (%) · descriptive 95% Wilson interval')
ax.set_title('Trained does not consistently beat untrained\n128 games per pairing, balanced seats; timeouts earn zero',loc='left',fontsize=14,pad=15)
ax=axes[1];labels=['Lana + Dreamer4','Legacy MuZero + new MuZero','New MuZero + untrained'];y=np.arange(3)
ax.barh(y-.17,[r['seconds32'] for r in speed],height=.32,color='#b9c3c8',label='32 parallel games')
ax.barh(y+.17,[r['seconds128'] for r in speed],height=.32,color='#196c82',label='128 parallel games')
for i,r in enumerate(speed):ax.text(r['seconds32']+3,i-.17,f"{r['speedup']:.2f}× faster",va='center',fontsize=10)
ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,210);ax.set_xlabel('Seconds to finish 128 games (lower is better)');ax.legend(loc='upper right',frameon=False);ax.grid(axis='x',alpha=.15)
ax.set_title('Batching made evaluation faster with identical outcomes\nH100 SXM 80 GB · all 384 paired game records matched',loc='left',fontsize=14,pad=12)
fig.suptitle('Research results: 11,648 comparison games + 768 speed-test games',fontsize=16,fontweight='bold')
for suffix in ['png','pdf','svg']:fig.savefig(out/f'research-summary.{suffix}',dpi=170,bbox_inches='tight')
