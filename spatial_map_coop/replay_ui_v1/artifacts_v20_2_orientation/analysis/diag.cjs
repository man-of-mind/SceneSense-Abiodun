/* DIAGNOSIS ONLY. GT used to classify display heading errors; never fed to the engine. */
const fs=require('fs'),readline=require('readline'),path=require('path');
const [dir,gtPath,out,which]=process.argv.slice(2);
const V='/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/spatial_map_coop/';
const T=require(which==='before'?V+'replay_ui_v1/baseline_v3_1/replay_tracks_v3.js':which==='mid'?V+'replay_ui_v1/baseline_v20_1/replay_tracks_v3.js':V+'viewer_assets/replay_tracks_v3.js');
const geo=JSON.parse(fs.readFileSync(path.join(dir,'geometry.json'))),gt=JSON.parse(fs.readFileSync(gtPath));
const ax=(a,b)=>Math.abs(T.axisDelta(a,b));
// road segment grid
const segs=[];for(const r of geo.roads)for(let i=1;i<r.length;i++)segs.push([r[i-1],r[i]]);
const grid=new Map(),cell=10;segs.forEach((s,i)=>{const k=Math.floor((s[0][0]+s[1][0])/2/cell)+','+Math.floor((s[0][1]+s[1][1])/2/cell);if(!grid.has(k))grid.set(k,[]);grid.get(k).push(i);});
function road(x,y){let best=null;const cx=Math.floor(x/cell),cy=Math.floor(y/cell);for(let dx=-1;dx<=1;dx++)for(let dy=-1;dy<=1;dy++)for(const i of grid.get((cx+dx)+','+(cy+dy))||[]){
 const [a,b]=segs[i],vx=b[0]-a[0],vy=b[1]-a[1],t=Math.max(0,Math.min(1,((x-a[0])*vx+(y-a[1])*vy)/(vx*vx+vy*vy))),d=Math.hypot(a[0]+t*vx-x,a[1]+t*vy-y);
 if(!best||d<best.d)best={d,yaw:Math.atan2(vy,vx)*180/Math.PI};}return best;}
const frames=new Map(Object.entries(gt.frames).map(([k,v])=>[+k,v.filter(b=>b.label==='vehicle')]));
function gtNear(fid,x,y){let best=null;const cand=[...(frames.get(fid)||[]).map(b=>({...b,kind:b.src})),...(frames.has(fid)?[]:gt.static.map(s=>({...s,id:'static',kind:'catalogue'})))];
 for(const b of cand){const d=Math.hypot(b.x-x,b.y-y);if(d<3&&(!best||d<best.d))best={...b,d};}return best;}
const engine=new T.TrackEngine({policy:JSON.parse(process.env.POLICY||'{}'),roads:geo}),rows=[];
(async()=>{
 for await(const line of readline.createInterface({input:fs.createReadStream(path.join(dir,'events.jsonl')),crlfDelay:Infinity})){
  const e=JSON.parse(line);engine.observe(e);const v=engine.view(e.arrival_ms);
  for(const r of v.visible){if(r.cls!=='Vehicle')continue;const t=r.track,g=gtNear(e.frame_id,r.pose.x,r.pose.y),rd=road(r.pose.x,r.pose.y);
   const qual=t.hist.filter(h=>h.strong&&Number.isFinite(h.yaw)&&e.capture_ms-h.t<=1500);
   rows.push({f:e.frame_id,ms:e.capture_ms,key:r.key,tier:r.tier,conflict:!!r.conflict,motion:r.motion,speed:r.speed,yaw:r.displayYaw!==undefined?r.displayYaw:r.yaw,assocYaw:r.yaw,dsrc:t.display?.source??null,src:t.heading?.source??null,
    known:r.headingKnown,dist:Math.hypot(r.pose.x-e.pose.x,r.pose.y-e.pose.y),x:r.pose.x,y:r.pose.y,
    gt:g?{id:g.id,yaw:g.yaw,speed:g.speed??0,kind:g.kind,d:g.d,parked:g.parked}:null,road:rd&&rd.d<6?rd:null,
    raw:qual.map(h=>h.yaw),rawLast:t.hist.at(-1).yaw,updated:t.lastT===e.capture_ms,firstYaw:t.firstYaw});
  }
  // remember first accepted axis per track
  for(const t of engine.tracks.values())if(t.firstYaw===undefined&&t.heading&&Number.isFinite(t.heading.yaw))t.firstYaw=t.heading.yaw;
 }
 fs.writeFileSync(out,rows.map(r=>JSON.stringify(r)).join('\n')+'\n');console.log(rows.length);
})();
