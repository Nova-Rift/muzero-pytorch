// Binary transport around the unchanged canonical TypeScript arena.
import {pathToFileURL} from 'node:url'
import path from 'node:path'
const site=process.argv[2],agents=Number(process.argv[3]??11),cadence=process.argv[4]??'legacy'
if(!Number.isInteger(agents)||agents<2||agents>11)throw Error('Invalid agent count')
if(!site)throw Error('Pass canonical website checkout path')
const {newWorld,stepWorld}=await import(pathToFileURL(path.join(site,'src/sandboxCombat.ts')))
const {observation}=await import(pathToFileURL(path.join(site,'src/lana/observation.ts')))
let worlds=[],pending=Buffer.alloc(0)
const stride=11*64+11+11+3
function respond(){
 const result=new Float32Array(worlds.length*stride)
 worlds.forEach((w,k)=>{
  let i=k*stride
  for(let id=0;id<11;id++){result.set(observation(w,id),i);i+=64}
  for(const f of w.fighters)result[i++]=f.hp>0?1:0
  for(const r of w.rewards)result[i++]=r
  result[i++]=w.phase==='finished'?1:0;result[i++]=w.winner??-1;result[i++]=w.shots.length
 })
 const bytes=Buffer.from(result.buffer),header=Buffer.alloc(4);header.writeUInt32LE(bytes.length)
 process.stdout.write(header);process.stdout.write(bytes)
}
// Bearing (radians, atan2 convention of the arena: +x east, +y south) from fighter `id` to the nearest live
// opponent by Euclidean pixel distance; ties keep the lowest fighter id. Recomputed from the current world
// state at every simulated frame. With no live opponent the offset is applied to bearing 0 (absolute decoding).
function bearing(w,id){
 const self=w.fighters[id];let best=null,bestDistance=Infinity
 for(const other of w.fighters){if(other.id===id||other.hp<=0)continue;const d=Math.hypot(other.x-self.x,other.y-self.y);if(d<bestDistance){bestDistance=d;best=other}}
 return best?Math.atan2(best.y-self.y,best.x-self.x):0
}
function request(bytes){
 const operation=bytes.readInt32LE(0),count=bytes.readInt32LE(4)
 if(operation===0){
  if(count<1||count>1024||bytes.length!==8+4*count)throw Error('Invalid reset')
  worlds=Array.from({length:count},(_,i)=>{const w=newWorld(0,bytes.readInt32LE(8+i*4));if(agents<11)for(const f of w.fighters)if(f.id>=agents)f.hp=0;w.phase='playing';return w})
 }else if(operation===1){
  if(count!==worlds.length||bytes.length!==8+count*44)throw Error('Invalid step')
  worlds.forEach((w,i)=>{
   const actions=Array.from({length:11},(_,j)=>bytes.readInt32LE(8+(i*11+j)*4))
   if(actions[0]===-1)return
   if(actions.some(a=>a<0||a>=306))throw Error('Invalid action')
   // Wire actions 0..152 decode exactly as before (absolute aim). Wire actions 153..305 are the same
   // 153 model indices with egocentric aim: the aim bin is an offset from the bearing to the nearest live opponent.
   const controller=(w,id)=>{const wire=actions[id],relative=wire>=153,a=relative?wire-153:wire,move=a%9,aim=Math.floor(a/9);let angle=(aim-1)*Math.PI/8;if(relative&&aim>0)angle+=bearing(w,id);return {dx:move%3-1,dy:Math.floor(move/3)-1,fire:aim>0,aimX:Math.cos(angle),aimY:Math.sin(angle)}}
   if(cadence==='homepage'){
    w.decisionClock=0
    for(let frame=0;frame<5;frame++)stepWorld(w,.05,0,0,controller,true)
   }else{
    do{stepWorld(w,.05,0,0,controller,true)}while(w.phase!=='finished'&&w.decisionClock>0)
   }
  })
 }else throw Error('Unknown operation')
 respond()
}
for await(const chunk of process.stdin){
 pending=Buffer.concat([pending,chunk])
 while(pending.length>=4){
  const size=pending.readUInt32LE(0)
  if(size>1024*1024)throw Error('Oversized request')
  if(pending.length<size+4)break
  const payload=pending.subarray(4,size+4);pending=pending.subarray(size+4);request(payload)
 }
}
