/* Offline display diagnostics only: no GT, no scientific association or scoring. */
const fs=require('fs'),readline=require('readline'),path=require('path');
const dir=process.argv[2],out=process.argv[3],which=process.argv[4]||'after';
const T=require(which==='before'?'./baseline_v3_1/replay_tracks_v3.js':'../viewer_assets/replay_tracks_v3.js');
const G=require('../viewer_assets/replay_geometry_v2.js');
const geometry=JSON.parse(fs.readFileSync(path.join(dir,'geometry.json'))),buildings=new G.Buildings(geometry);
const engine=new T.TrackEngine(),history=new Map(),counts={},dispositions={},rows=fs.createWriteStream(out+'.jsonl');
const d=(a,b)=>Math.abs(T.axisDelta(a,b)),inc=(o,k)=>o[k]=(o[k]||0)+1;
(async()=>{
 for await(const line of readline.createInterface({input:fs.createReadStream(path.join(dir,'events.jsonl')),crlfDelay:Infinity})){
  const e=JSON.parse(line);engine.observe(e);const v=engine.view(e.arrival_ms);
  for(const a of v.dispositions)inc(dispositions,a.disposition);
  for(const r of v.tracks){const t=r.track,entry=t.entry,geo=r.shown?buildings.classify(v.sensor,r.pose,{length:r.dims.l,width:r.dims.w},r.yaw):null;
   const raw=entry.raw;const inside=geometry.buildings.map((poly,i)=>G.inside([r.pose.x,r.pose.y],poly)?i:null).filter(i=>i!==null);
   const row={frame:e.frame_id,ms:e.capture_ms,arrival:e.arrival_ms,id:r.id,key:r.key,cls:r.cls,tier:r.tier,
    rawId:raw.id,rawXY:[raw.location.x,raw.location.y],rawYaw:entry.heading_yaw_deg,qualified:entry.heading_valid,score:raw.score,
    yaw:r.yaw,heading:t.heading,motion:r.motion,speed:r.speed,z:t.tstat,vx:t.est.vx,vy:t.est.vy,xy:[r.pose.x,r.pose.y],
    dims:r.dims,shown:r.shown,reason:r.reason,geo,inside,conflict:r.conflict,updated:t.lastT===e.capture_ms};
   rows.write(JSON.stringify(row)+'\n');
   if(!history.has(r.key))history.set(r.key,[]);history.get(r.key).push(row);
   inc(counts,r.cls+'/'+r.motion);if(r.shown)inc(counts,'shown/'+r.cls);if(geo)inc(counts,geo.state+'/'+geo.reason);
  }
 }
 rows.end();
 const summaries=[...history].map(([key,a])=>{const u=a.filter(r=>r.updated),stat=u.filter(r=>r.motion==='stationary');
  let stationaryTV=0,totalTV=0,maxStep=0,switches=0;
  for(let i=1;i<u.length;i++){if(u[i].yaw!==null&&u[i-1].yaw!==null){const q=d(u[i-1].yaw,u[i].yaw);totalTV+=q;maxStep=Math.max(q,maxStep);if(u[i].motion==='stationary'&&u[i-1].motion==='stationary')stationaryTV+=q;}
   if(u[i].heading?.source!==u[i-1].heading?.source)switches++;}
  return {key,id:a[a.length-1].id,cls:a[0].cls,start:a[0].frame,end:a[a.length-1].frame,startMs:a[0].ms,endMs:a[a.length-1].ms,
   n:u.length,stationary:stat.length,moving:u.filter(r=>r.motion==='moving').length,stationaryTV,totalTV,maxStep,switches,
   displacement:Math.hypot(u.at(-1).xy[0]-u[0].xy[0],u.at(-1).xy[1]-u[0].xy[1]),inside:a.filter(r=>r.shown&&r.inside.length).length};});
 fs.writeFileSync(out+'.summary.json',JSON.stringify({counts,dispositions,stats:{...engine.stats,bridges:undefined},
  stationary:summaries.filter(s=>s.cls==='Vehicle'&&s.stationary>=10).sort((a,b)=>b.stationaryTV-a.stationaryTV).slice(0,20),
  moving:summaries.filter(s=>s.cls==='Vehicle'&&s.moving>=10).sort((a,b)=>b.totalTV-a.totalTV).slice(0,20),
  overlaps:summaries.filter(s=>s.inside).sort((a,b)=>b.inside-a.inside).slice(0,20)},null,2)+'\n');
 console.log({events:engine.stats.events,tracks:history.size,counts});
})();
