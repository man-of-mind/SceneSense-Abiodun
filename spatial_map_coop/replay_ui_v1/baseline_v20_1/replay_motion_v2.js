/* Private causal presentation state. No scientific association or raw edits. */
(function(root){'use strict';
const P=typeof module!=='undefined'&&module.exports?require('./presentation_predictor_v1.js'):root.ScenesenseMapPresentation;
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
function box(e){const c=e.upstream?.candidate;return c?[c.bbox_x0,c.bbox_y0,c.bbox_x1,c.bbox_y1]:null;}
function overlap(a,b){if(!a||!b)return {iou:0,contain:0};const area=b=>Math.max(0,b[2]-b[0])*Math.max(0,b[3]-b[1]);
 const inter=Math.max(0,Math.min(a[2],b[2])-Math.max(a[0],b[0]))*Math.max(0,Math.min(a[3],b[3])-Math.max(a[1],b[1]));
 return {iou:inter/(area(a)+area(b)-inter||1),contain:inter/(Math.min(area(a),area(b))||1)};}
function distance(a,b){return Math.hypot(a.raw.location.x-b.raw.location.x,a.raw.location.y-b.raw.location.y);}
function duplicate(a,b){if(a.source!==b.source||a.frame_id!==b.frame_id||a.raw.type!==b.raw.type||!box(a)||!box(b))return false;
 const ov=overlap(box(a),box(b)),ad=a.raw.dimensions,bd=b.raw.dimensions;
 return ov.iou>=.45&&ov.contain>=.85&&distance(a,b)<=Math.min(ad.width,bd.width)*.5&&
  ['length','width','height'].every(k=>Math.min(ad[k],bd[k])/Math.max(ad[k],bd[k])>=.65)&&
  a.heading_valid&&b.heading_valid&&Math.abs(P.angleDeltaDeg(a.heading_yaw_deg,b.heading_yaw_deg))<=25;
}
function temporalSupport(a,b,prior){
 if(prior===null)return true;
 return prior.some(x=>x.raw.type===a.raw.type&&distance(a,x)<3&&overlap(box(a),box(x)).iou>.25&&prior.some(y=>y!==x&&duplicate(x,y)&&distance(b,y)<3&&overlap(box(b),box(y)).iou>.25));
}
function consolidate(entries,enabled=true,prior=null){const groups=[],sorted=entries.slice().sort((a,b)=>b.raw.score-a.raw.score||a.raw.id.localeCompare(b.raw.id));
 for(const e of sorted){const group=enabled?groups.find(g=>g.every(m=>duplicate(e,m)&&temporalSupport(e,m,prior))):null;if(group)group.push(e);else groups.push([e]);}
 return groups.map(members=>({...members[0],members,display_group:members.length>1?'OVERLAPPING_IMAGE_HYPOTHESES':null}));
}
class Filter {
 constructor(options={}){this.states=Object.create(null);this.ego=!!options.ego;this.metrics={created:0,resets:0,reacquired:0,updates:0,corrections:[],innovations:[],filterOffsets:[],correctionClips:0};}
 observe(key,m,arrival){let s=this.states[key];if(s&&m.measuredMs<=s.measuredMs)return m.measuredMs===s.measuredMs?'REPEATED':'OUT_OF_ORDER';
  if(!Number.isFinite(m.x)||!Number.isFinite(m.y)||!Number.isFinite(m.measuredMs))return 'INVALID';
  const dt=s?(m.measuredMs-s.measuredMs)/1000:0;
  if(!s||dt>.5){const old=s?this.sample(key,arrival):null,delta=old?Math.hypot(old.x-m.x,old.y-m.y):0;
   const reacquired=!!old&&dt<.75&&delta<=3;
   this.states[key]={...m,rawX:m.x,rawY:m.y,vx:0,vy:0,arrivalMs:arrival,errX:reacquired?old.x-m.x:0,errY:reacquired?old.y-m.y:0,errYaw:reacquired?P.angleDeltaDeg(m.yaw,old.yaw):0,errVx:0,errVy:0,yaw:m.yaw,ageAtArrivalMs:m.ageAtArrivalMs||0};
   this.metrics[reacquired?'reacquired':old?'resets':'created']++;return reacquired?'REACQUIRED':old?'RESET':'CREATED';}
  const old=this.sample(key,arrival),px=s.x+s.vx*dt,py=s.y+s.vy*dt,rx=m.x-px,ry=m.y-py,innovation=Math.hypot(rx,ry);
  if(innovation>5){delete this.states[key];this.observe(key,m,arrival);this.metrics.resets++;return 'RESET';}
  // Alpha-beta at capture time, then predict to the SAME display clock as ego.
  // A stop is detected from measured displacement, not filtered displacement.
  const alpha=this.ego?1:.4,beta=this.ego?.5:.1,rawSpeed=Math.hypot(m.x-s.rawX,m.y-s.rawY)/dt;
  let vx=s.vx+beta*rx/dt,vy=s.vy+beta*ry/dt;
  if(rawSpeed<.25){vx=vy=0;}
  [vx,vy]=P.clampVelocity(vx,vy,P.maxDisplaySpeed(m.className));
  const x=rawSpeed<.25?m.x:px+alpha*rx,y=rawSpeed<.25?m.y:py+alpha*ry;
  // Heading is a noisy measurement too. Hold through missing upstream joins;
  // limit rotation without resetting positions. No road snapping.
  const yaw=s.yaw+clamp(P.angleDeltaDeg(s.yaw,m.yaw),-90*dt,90*dt);
  const age=Math.min(.5,(m.ageAtArrivalMs||0)/1000),ex=old.x-(x+vx*age),ey=old.y-(y+vy*age);
  // Continuity bridge is completed in 100 ms, less than the recorded cadence.
  // It targets age-compensated state and never extends prediction/expiry.
  this.states[key]={...m,x,y,yaw,rawX:m.x,rawY:m.y,vx,vy,arrivalMs:arrival,ageAtArrivalMs:m.ageAtArrivalMs||0,
   errX:ex,errY:ey,errVx:old.displayVx-vx,errVy:old.displayVy-vy,errYaw:P.angleDeltaDeg(yaw,old.yaw)};
  this.metrics.updates++;for(const [k,v] of [['corrections',Math.hypot(ex,ey)],['innovations',innovation],['filterOffsets',Math.hypot(x-m.x,y-m.y)]]){this.metrics[k].push(v);if(this.metrics[k].length>8192)this.metrics[k].shift();}
  return 'ACCEPTED';
 }
 sample(key,now){const s=this.states[key];if(!s)return null;const age=Math.max(0,now-s.arrivalMs+s.ageAtArrivalMs),t=Math.min(500,age)/1000;
  const u=clamp((now-s.arrivalMs)/100,0,1),decay=1-u*u*(3-2*u),tangent=.1*(u-2*u*u+u*u*u),dDecay=(6*u*u-6*u)/.1,dTangent=1-4*u+3*u*u;
  return {x:s.x+s.vx*t+s.errX*decay+(s.errVx||0)*tangent,y:s.y+s.vy*t+s.errY*decay+(s.errVy||0)*tangent,yaw:s.yaw+s.errYaw*decay,
   vx:s.vx,vy:s.vy,displayVx:(age>=500?0:s.vx)+s.errX*dDecay+(s.errVx||0)*dTangent,displayVy:(age>=500?0:s.vy)+s.errY*dDecay+(s.errVy||0)*dTangent,held:age>=500,ageMs:age};}
 atCapture(key,ms){const s=this.states[key];if(!s)return null;const dt=clamp((ms-s.measuredMs)/1000,0,.5);return {x:s.x+s.vx*dt,y:s.y+s.vy*dt};}
 retain(keys){const keep=new Set(keys);for(const key of Object.keys(this.states))if(!keep.has(key))delete this.states[key];}
}
// Rectangular Hungarian assignment with unmatched columns. Deterministic global
// one-to-one matching avoids greedy order swaps at adjacent/crossing tracks.
function assignment(cost){const n=cost.length;if(!n)return [];const m=cost[0].length,u=Array(n+1).fill(0),v=Array(m+1).fill(0),p=Array(m+1).fill(0),way=Array(m+1).fill(0);
 for(let i=1;i<=n;i++){p[0]=i;let j0=0;const min=Array(m+1).fill(Infinity),used=Array(m+1).fill(false);
  do{used[j0]=true;const i0=p[j0];let delta=Infinity,j1=0;for(let j=1;j<=m;j++)if(!used[j]){const cur=cost[i0-1][j-1]-u[i0]-v[j];if(cur<min[j]){min[j]=cur;way[j]=j0;}if(min[j]<delta){delta=min[j];j1=j;}}
   for(let j=0;j<=m;j++)if(used[j]){u[p[j]]+=delta;v[j]-=delta;}else min[j]-=delta;j0=j1;
  }while(p[j0]!==0);do{const j1=way[j0];p[j0]=p[j1];j0=j1;}while(j0!==0);
 }const result=Array(n).fill(-1);for(let j=1;j<=m;j++)if(p[j])result[p[j]-1]=j-1;return result;
}
function match(engine,entries,event){const tracks=[...engine.tracks.values()].filter(t=>t.entry.source===event.source&&event.capture_ms-t.event.capture_ms<750).sort((a,b)=>a.key.localeCompare(b.key));
 const costs=entries.map(e=>[...tracks.map(t=>{if(e.raw.type!==t.entry.raw.type)return 1e4;
  const p=engine.predictor.atCapture(t.key,event.capture_ms),d=Math.hypot(p.x-e.raw.location.x,p.y-e.raw.location.y);
  const lastD=distance(e,t.entry);if(Math.min(d,lastD)>3||['length','width'].some(k=>Math.min(e.raw.dimensions[k],t.entry.raw.dimensions[k])/Math.max(e.raw.dimensions[k],t.entry.raw.dimensions[k])<.5))return 1e4;
  const overlapCost=box(e)&&box(t.entry)?1-overlap(box(e),box(t.entry)).iou:0.5;
  const headingCost=e.heading_valid&&t.entry.heading_valid?Math.abs(P.angleDeltaDeg(e.heading_yaw_deg,t.entry.heading_yaw_deg))/180:0;
  return .65*d+.35*lastD+overlapCost*.4+headingCost*.3;
 }),...entries.map(()=>5)]);
 const pairs=assignment(costs);return entries.map((e,i)=>{const j=pairs[i],previous=j<tracks.length&&costs[i][j]<5?tracks[j]:null;
  if(!previous)engine.churn.created++;
  else if(previous.entry.backend_track_id!==e.track_id)engine.churn.backend_id_changes++;
  return {...e,backend_track_id:e.track_id,track_id:previous?previous.entry.track_id:'display_'+(++engine.nextID)};});
}
const api={Filter,box,overlap,duplicate,consolidate,match,assignment};root.ScenesenseReplayMotion=api;if(typeof module!=='undefined'&&module.exports)module.exports=api;
})(typeof globalThis!=='undefined'?globalThis:this);
