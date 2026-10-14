/* Deterministic normal-speed (60 Hz) full-route comparison. Presentation metrics only. */
'use strict';
const fs=require('fs'),readline=require('readline'),path=require('path'),assert=require('assert/strict');
const B=require('./baseline_v3_1/replay_tracks_v3.js'),A=require('../viewer_assets/replay_tracks_v3.js'),G=require('../viewer_assets/replay_geometry_v2.js');
const input=process.argv[2],out=process.argv[3];
const buildings=new G.Buildings(JSON.parse(fs.readFileSync(path.join(input,'geometry.json'))));
const sum=a=>{a.sort((x,y)=>x-y);return {n:a.length,mean:a.reduce((x,y)=>x+y,0)/(a.length||1),p95:a[Math.floor(a.length*.95)]||0,max:a.at(-1)||0};};
const modes=['before','after'],engines=[new B.TrackEngine(),new A.TrackEngine()];
const all=engines.map(()=>({prev:new Map(),axis:[],stationaryAxis:[],pos:[],appear:0,disappear:0,hidden:{},grouped:0,visibleSamples:0,jumpsOver20:0,stationaryTV:0,stationarySeconds:0}));
let now=0,frames=0,pedDifferences=0,matchedPositionDifferences=0,matched=0,previousArrival=null;
const turns=engines.map(()=>({target:null,display:null}));
function sample(now){
 const views=engines.map(e=>e.view(now));frames++;
 const ped=views.map(v=>v.tracks.filter(r=>r.cls==='Pedestrian').map(r=>({raw:r.entry.raw.id,xy:r.pose,yaw:r.yaw,motion:r.motion,tier:r.tier,shown:r.shown,conflict:!!r.conflict})).sort((a,b)=>a.raw.localeCompare(b.raw)));
 if(JSON.stringify(ped[0])!==JSON.stringify(ped[1]))pedDifferences++;
 const ra=new Map(views[0].tracks.map(r=>[r.entry.raw.id,r]));
 for(const r of views[1].tracks){const b=ra.get(r.entry.raw.id);if(b){matched++;if(Math.hypot(b.pose.x-r.pose.x,b.pose.y-r.pose.y)>1e-8)matchedPositionDifferences++;}}
 for(let i=0;i<2;i++){
  const v=views[i],m=all[i],next=new Map();
  for(const r of v.tracks){let reason=r.reason;
   if(!reason&&buildings.classify(v.sensor,r.pose,{length:r.dims.l,width:r.dims.w},r.yaw).state==='STATIC_MAP_OCCLUDED')reason='STATIC_MAP_OCCLUDED';
   if(reason){m.hidden[reason]=(m.hidden[reason]||0)+1;continue;}
   if(r.conflict)m.grouped++;m.visibleSamples++;
   next.set(r.key,r);const prev=m.prev.get(r.key);
   if(!prev)m.appear++;
   else if(r.cls==='Vehicle'){
    m.pos.push(Math.hypot(prev.pose.x-r.pose.x,prev.pose.y-r.pose.y));
    if(Number.isFinite(r.yaw)&&Number.isFinite(prev.yaw)){
     const d=Math.abs(A.axisDelta(prev.yaw,r.yaw));m.axis.push(d);if(d>20)m.jumpsOver20++;
     if(r.motion==='stationary'&&prev.motion==='stationary'){m.stationaryAxis.push(d);m.stationaryTV+=d;m.stationarySeconds+=1/60;}
    }
   }
   // Recorded trajectory turns from south to west. No GT angle is assumed.
   if(r.cls==='Vehicle'&&r.track.bornT===221772.766208&&now>240000&&now<254000){
    const th=turns[i],a=Math.atan2(r.track.est.vy,r.track.est.vx)*180/Math.PI;
    const display=-90+A.axisDelta(-90,r.yaw);
    if(th.target===null&&r.motion==='moving'&&a<=-135)th.target=now;
    if(th.display===null&&display<=-135)th.display=now;
   }
  }
  for(const k of m.prev.keys())if(!next.has(k))m.disappear++;m.prev=next;
 }
}
(async()=>{
 for await(const line of readline.createInterface({input:fs.createReadStream(path.join(input,'events.jsonl')),crlfDelay:Infinity})){
  const e=JSON.parse(line);
  while(now<e.arrival_ms){sample(now);now+=1000/60;}
  engines.forEach(t=>t.observe(e));previousArrival=e.arrival_ms;
 }
 while(now<previousArrival+800){sample(now);now+=1000/60;}
 const results=Object.fromEntries(all.map((m,i)=>[modes[i],{angular_step_deg_60hz:sum(m.axis),stationary_step_deg_60hz:sum(m.stationaryAxis),
   position_step_m_60hz:sum(m.pos),stationary_total_variation_deg:m.stationaryTV,stationary_deg_per_track_second:m.stationaryTV/m.stationarySeconds,
   axis_steps_over_20_deg:m.jumpsOver20,appearances:m.appear,disappearances:m.disappear,visible_track_samples:m.visibleSamples,
   conflict_track_samples:m.grouped,hidden_track_samples_by_reason:m.hidden,
   created:engines[i].stats.tentativeCreated,established:engines[i].stats.established,absorbed_raw:engines[i].stats.absorbed,
   turn_crossing_ms:turns[i],turn_lag_from_causal_fit_ms:turns[i].display-turns[i].target}]));
 assert.equal(pedDifferences,0,'pedestrian positions / headings / motion / tiers / visibility unchanged at every sample');
 const report={conditions:'3576 identical V20 events at recorded installation times; 60 Hz 1x clock; 45 m forward view; existing building withholding on. Full-route IDs; no GT. Steps refer to the same display key, not scientific identity.',
  frames,pedestrian_sample_differences:pedDifferences,matched_raw_track_samples:matched,matched_position_sample_differences:matchedPositionDifferences,...results};
 fs.writeFileSync(out,JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
})().catch(e=>{console.error(e);process.exitCode=1;});
