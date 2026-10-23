"""Copy canonical combat into an isolated observer module without altering physics."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('site',type=Path);p.add_argument('output',type=Path);a=p.parse_args();base=a.site.resolve();src=(base/'src/sandboxCombat.ts').read_text()
x=src.replace("'./lana/arena.ts'",repr(str(base/'src/lana/arena.ts'))).replace("'./sandboxMotion.ts'",repr(str(base/'src/sandboxMotion.ts')))
x='let observer=(w,event)=>{};export function setObserver(fn){observer=fn}\n'+x
changes=[('w.shots.push({id:w.nextShot++,owner:id,x:f.x,y:f.y,vx:(x-f.x)/length*500,vy:(y-f.y)/length*500})',"w.shots.push({id:w.nextShot++,owner:id,x:f.x,y:f.y,vx:(x-f.x)/length*500,vy:(y-f.y)/length*500});observer(w,{type:'fire',shot:{...w.shots[w.shots.length-1]},target:{...w.fighters[1-id]}})"),('if(hit){hit.hp--}',"if(hit){hit.hp--;observer(w,{type:'hit',shot:{...shot},target_id:hit.id})}"),('if(shot.x>=0&&shot.x<=1000&&shot.y>=0&&shot.y<=460)retained.push(shot)',"if(shot.x>=0&&shot.x<=1000&&shot.y>=0&&shot.y<=460)retained.push(shot);else observer(w,{type:'exit',shot:{...shot}})")]
for old,new in changes:
 if x.count(old)!=1:raise ValueError('Canonical source contract changed; inspect before instrumenting')
 x=x.replace(old,new)
with a.output.open('x') as f:f.write(x)
print(json.dumps({'canonical_sha256':hashlib.sha256(src.encode()).hexdigest(),'observer_sha256':hashlib.sha256(x.encode()).hexdigest(),'source_unchanged':True}))
