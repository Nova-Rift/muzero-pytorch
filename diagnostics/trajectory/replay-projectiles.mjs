import fs from 'node:fs';
import {pathToFileURL} from 'node:url';
const base=process.argv[2];
const {newWorld,stepWorld,setObserver}=await import(pathToFileURL(process.argv[3]));
let events=[];setObserver((w,event)=>events.push(event));
const {observation}=await import(pathToFileURL(base+'/src/lana/observation.ts'));
const angular=(a,b)=>Math.abs(Math.atan2(Math.sin(a-b),Math.cos(a-b)));
if(angular(0,2*Math.PI)>1e-12||Math.abs(angular(0,Math.PI)-Math.PI)>1e-12)throw Error('Geometry test');
const candidateIndex=Number(process.argv[4]);if(![0,1].includes(candidateIndex))throw Error('Specify candidate catalog index 0 or 1');
const results=[];let checks=0,maxError=0;
for(const filename of process.argv.slice(5)){
 const rows=fs.readFileSync(filename,'utf8').trim().split('\n').map(JSON.parse),first=rows[0];
 for(let i=0;i<first.seeds.length;i++){
  const w=newWorld(0,first.seeds[i]);w.phase='playing';events=[];
  w.fighters.forEach((f,j)=>f.hp=first.alive[i][j]?3:0);
  const counts=[0,0],errors=[[],[]],near=[0,0];let steps=0;
  const check=row=>{for(let j=0;j<11;j++){const actual=observation(w,j),expected=row.observation[i][j];actual.forEach((v,k)=>{const e=Math.abs(v-expected[k]);maxError=Math.max(maxError,e);if(e>2e-6)throw Error(`Replay mismatch ${first.seeds[i]} ${steps} ${j} ${k} ${e}`);checks++})}if((w.phase==='finished')!==row.finished[i]||(w.winner??-1)!==row.winner[i])throw Error('Outcome mismatch')};
  check(first);
  for(const row of rows.slice(1)){
   if(w.phase==='finished')break;
   const actions=row.actions[i];const controller=(_,id)=>{let a=actions[id],aim=Math.floor(a/9),angle=(aim-1)*Math.PI/8;return{dx:a%9%3-1,dy:Math.floor(a%9/3)-1,fire:aim>0,aimX:Math.cos(angle),aimY:Math.sin(angle)}};
   w.decisionClock=0;
   for(let frame=0;frame<5;frame++){
    if(w.phase==='finished')break;
    const before=w.nextShot,old=w.fighters.map(f=>({hp:f.hp,cooldown:f.cooldown}));
    stepWorld(w,.05,0,0,controller,true);
    let emitted=0;
    for(let j=0;j<2;j++)if(old[j].hp>0&&old[j].cooldown<=0&&w.actions[j].fire){
     emitted++;counts[j]++;
     const f=w.fighters[j],o=w.fighters[1-j],bearing=Math.atan2(o.y-f.y,o.x-f.x),aim=Math.atan2(w.actions[j].aimY,w.actions[j].aimX),error=angular(aim,bearing);errors[j].push(error*180/Math.PI);near[j]+=error<=Math.PI/16?1:0;
    }
    if(w.nextShot-before!==emitted)throw Error('Shot accounting mismatch');
   }
   steps++;check(row);
  }
  const fired=events.filter(e=>e.type==='fire'),hit=events.filter(e=>e.type==='hit'),exit=events.filter(e=>e.type==='exit');
  const attributed=[0,1].map(owner=>({emitted:fired.filter(e=>e.shot.owner===owner).length,hits:hit.filter(e=>e.shot.owner===owner).length,exited:exit.filter(e=>e.shot.owner===owner).length}));
  for(let j=0;j<2;j++){if(attributed[j].emitted!==counts[j])throw Error('Observer shot accounting');attributed[j].unresolved=attributed[j].emitted-attributed[j].hits-attributed[j].exited;if(attributed[j].unresolved<0)throw Error('Negative unresolved');}
  results.push({projectiles:attributed,file:filename.split('/').slice(-2).join('/'),seed:first.seeds[i],candidate_seat:(candidateIndex+i)%2,steps,finished:w.phase==='finished',winner:w.winner,actual_shots:counts,mean_shot_angle_error_degrees:errors.map(a=>a.reduce((s,v)=>s+v,0)/a.length),shots_within_half_aim_bin:near});
 }
}
console.log(JSON.stringify({schema:1,iteration:15,kind:'deterministic_saved_action_replay',new_evaluation_games:0,observation_scalar_checks:checks,max_observation_error:maxError,limitations:'Aim relative to opponent post-movement at each emission; no projectile lead or dodge model. Descriptive only, not causal strength evidence.',games:results},null,2));
