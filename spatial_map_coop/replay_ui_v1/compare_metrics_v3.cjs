// Deterministic 60 Hz display comparison: v2 (current default UI engine) vs v3
// display tracks, identical causal inputs, current-forward view, 45 m radius.
// GT is read ONLY here, to score what each display shows; it never enters a UI.
// Usage: node compare_metrics_v3.cjs <clipsDir> <gt_audit.json> [outJson]
'use strict';
const fs=require('fs'),path=require('path');
const V2=require('../viewer_assets/replay_policy_v1.js'),V3=require('../viewer_assets/replay_tracks_v3.js');
const [clipsDir,gtPath,outPath]=process.argv.slice(2);
const gt=JSON.parse(fs.readFileSync(gtPath)),statics=gt.static_vehicles;
const q=(a,p)=>{if(!a.length)return null;const s=a.slice().sort((x,y)=>x-y);return s[Math.min(s.length-1,Math.floor(s.length*p))];};
const summary=a=>({n:a.length,mean:a.length?a.reduce((s,x)=>s+x,0)/a.length:null,p50:q(a,.5),p95:q(a,.95),max:a.length?Math.max(...a):null});
const dYaw=(a,b)=>((b-a+540)%360)-180;
const dAxis=(a,b)=>{const d=dYaw(a,b);return d>90?d-180:d<-90?d+180:d;};
const cls=t=>t==='Pedestrian'?'person':'vehicle';
function glyphs(kind,view){
  if(kind==='v2')return view.visible.map(r=>({key:r.key,x:r.pose.x,y:r.pose.y,yaw:r.pose.yaw,cls:cls(r.entry.raw.type),tier:'glyph',
    speed:Math.hypot(r.pose.vx||0,r.pose.vy||0),nose:r.heading!=='unknown'}));
  return view.visible.map(r=>({key:r.key,x:r.pose.x,y:r.pose.y,yaw:r.yaw,cls:cls(r.cls),tier:r.tier,missed:r.missed,speed:r.speed,nose:r.headingKnown&&r.cls==='Vehicle'}));
}
function run(kind,clip,frames){
  const engine=kind==='v2'?new V2.Engine({refined:true,consolidation:true}):new V3.TrackEngine();
  const events=clip.events,gtFrames=Object.values(frames).sort((a,b)=>a.capture_ms-b.capture_ms);
  let noseFlips=0,cursor=0,g=0,prev=new Map(),prevStep=new Map(),appear=0,disappear=0,samples=0,sumShown=0,sumEst=0,sumTent=0,sumField=0,sumRaw=0;
  const steps=[],second=[],estSteps=[],yawRate=[],stillYawRate=[],lifetimes=new Map(),cov={gt:0,any:0,est:0,estTent:0},unsupported={glyphs:0,unsupported:0,est:0,estUnsupported:0};
  const lag=[],lateral=[],perActor=new Map(),byRange={};
  const dt=1000/60;
  for(let now=0;now<=clip.end_ms;now+=dt){
    while(cursor<events.length&&events[cursor].arrival_ms<=now)engine.observe(events[cursor++]);
    const view=engine.view(now,{mode:'current',radius:45}),gs=glyphs(kind,view),next=new Map();samples++;
    sumShown+=gs.length;sumRaw+=engine.latest?.objects?.length||engine.latest?.original_objects?.length||0;
    if(kind==='v3'){sumEst+=gs.filter(x=>x.tier==='established').length;sumTent+=gs.filter(x=>x.tier==='tentative').length;sumField+=view.field.filter(c=>c.shown).length;}
    for(const o of gs){next.set(o.key,o);lifetimes.set(o.key,(lifetimes.get(o.key)||0)+dt);const p=prev.get(o.key);
      if(!p){appear++;continue;}const dx=o.x-p.x,dy=o.y-p.y,d=Math.hypot(dx,dy);steps.push(d);if(o.tier!=='tentative')estSteps.push(d);
      const ps=prevStep.get(o.key);if(ps)second.push(Math.hypot(dx-ps[0],dy-ps[1]));prevStep.set(o.key,[dx,dy]);
      if(Number.isFinite(o.yaw)&&Number.isFinite(p.yaw)){const r=Math.abs(dAxis(p.yaw,o.yaw))*60;yawRate.push(r);if((o.speed||0)<.3)stillYawRate.push(r);if(o.nose&&p.nose&&Math.abs(dYaw(p.yaw,o.yaw))>120)noseFlips++;}}
    for(const k of prev.keys())if(!next.has(k))disappear++;prev=next;
    // GT scoring at the instant the GT snapshot was captured (display shows older data + prediction).
    while(g<gtFrames.length&&gtFrames[g].capture_ms<=now){const f=gtFrames[g++];if(!view.ego||f.capture_ms<now-dt)continue;
      const ego=view.ego,real=[...f.actors,...f.positives.map(p=>[p.x,p.y]),...statics];
      for(const p of f.positives){if(Math.hypot(p.x-ego.x,p.y-ego.y)>45)continue;cov.gt++;
        const near=gs.filter(o=>o.cls===p.label&&Math.hypot(o.x-p.x,o.y-p.y)<=3);
        if(near.length)cov.any++;if(near.some(o=>o.tier!=='tentative'))cov.est++;if(near.length&&kind==='v3')cov.estTent++;
        const best=near.sort((a,b)=>Math.hypot(a.x-p.x,a.y-p.y)-Math.hypot(b.x-p.x,b.y-p.y))[0];
        const a=perActor.get(p.id)||{still:p.label==='vehicle'&&p.speed<.1,obs:[]};/* pedestrian speed field is 0 in the archive: vehicles only */a.obs.push({t:f.capture_ms,key:best?.key??null,x:best?.x,y:best?.y,yaw:best?.yaw,count:near.length});perActor.set(p.id,a);
        if(best&&p.speed>2){const h=p.yaw*Math.PI/180,ex=best.x-p.x,ey=best.y-p.y;lag.push(ex*Math.cos(h)+ey*Math.sin(h));lateral.push(-ex*Math.sin(h)+ey*Math.cos(h));}}
      for(const o of gs){if(Math.hypot(o.x-ego.x,o.y-ego.y)>45)continue;const ok=real.some(r=>Math.hypot(r[0]-o.x,r[1]-o.y)<=3);
        unsupported.glyphs++;if(!ok)unsupported.unsupported++;if(o.tier!=='tentative'){unsupported.est++;if(!ok)unsupported.estUnsupported++;}
        const band=Math.hypot(o.x-ego.x,o.y-ego.y)<20?'lt20':Math.hypot(o.x-ego.x,o.y-ego.y)<35?'20to35':'35to45';byRange[band]=byRange[band]||[0,0];byRange[band][0]++;if(!ok)byRange[band][1]++;}}
  }
  // Per GT actor: how often the glyph representing it changes identity, how far a
  // parked actor's glyph wanders, and how much it rotates (evaluation only).
  let switches=0,pairs=0,multi=0,obsN=0;const wobble=[],stillRot=[],parkedSpeed=[];
  for(const a of perActor.values()){let last=null;const xs=[],ys=[];
    for(const o of a.obs){obsN++;if(o.count>1)multi++;
      if(last&&o.t-last.t<=300&&last.key&&o.key){pairs++;if(o.key!==last.key)switches++;
        else if(a.still&&Number.isFinite(o.yaw)&&Number.isFinite(last.yaw))stillRot.push(Math.abs(dAxis(last.yaw,o.yaw)));}
      if(a.still&&o.key){xs.push(o.x);ys.push(o.y);}if(a.still&&last&&last.key&&o.key===last.key&&o.t-last.t<=300)parkedSpeed.push(Math.hypot(o.x-last.x,o.y-last.y)/((o.t-last.t)/1000));last=o;}
    if(xs.length>=5){const mx=xs.reduce((s,x)=>s+x,0)/xs.length,my=ys.reduce((s,x)=>s+x,0)/ys.length;
      wobble.push(Math.sqrt(xs.reduce((s,x,i)=>s+(x-mx)**2+(ys[i]-my)**2,0)/xs.length));}}
  const life=[...lifetimes.values()],seconds=clip.end_ms/1000;
  const st=kind==='v3'?engine.stats:null;
  return {mean_glyphs:sumShown/samples,mean_established:kind==='v3'?sumEst/samples:null,mean_tentative:kind==='v3'?sumTent/samples:null,
    mean_hypothesis_cells:kind==='v3'?sumField/samples:null,mean_raw_candidates:sumRaw/samples,
    glyph_appearances_per_s:appear/seconds,glyph_disappearances_per_s:disappear/seconds,glyph_lifetime_ms:summary(life),
    step_m:summary(steps),established_step_m:summary(estSteps),second_difference_m:summary(second),axis_rate_dps:summary(yawRate),nose_flips_per_min:noseFlips/seconds*60,stationary_glyph_yaw_rate_dps:summary(stillYawRate),
    display_ids_created:kind==='v2'?engine.churn.created:st.tentativeCreated,established_created:kind==='v3'?st.established:null,
    gt_coverage:{gt_positive_within_45m:cov.gt,any_glyph_3m:cov.any/cov.gt,established_3m:kind==='v3'?cov.est/cov.gt:null,established_or_tentative_3m:kind==='v3'?cov.estTent/cov.gt:null},
    glyphs_without_any_real_object_3m:{by_range_m:Object.fromEntries(Object.entries(byRange).map(([k,[n,u]])=>[k,{n,fraction:u/n}])),fraction:unsupported.unsupported/unsupported.glyphs,established_fraction:kind==='v3'?unsupported.estUnsupported/unsupported.est:null,n:unsupported.glyphs},
    gt_actor_glyph_identity_switch_rate:pairs?switches/pairs:null,gt_actor_pairs:pairs,gt_actor_multiple_glyphs_fraction:obsN?multi/obsN:null,
    stationary_gt_actor_glyph_wobble_rms_m:summary(wobble),stationary_gt_actor_same_glyph_apparent_speed_mps:summary(parkedSpeed),stationary_gt_actor_glyph_axis_rotation_deg_per_gt_frame:summary(stillRot),
    moving_actor_along_track_error_m:summary(lag),moving_actor_lateral_error_m:summary(lateral.map(Math.abs)),
    confirmation:st?{capture_delay_ms:summary(st.confirmCaptureMs),arrival_delay_ms:summary(st.confirmArrivalMs),tentative_dropped:st.tentativeDropped,
      ended_expired:st.endedExpired,ended_weak_only:st.endedWeakOnly,established_lifetime_ms:summary(st.establishedLifetimeMs),
      bridge_m:summary(st.bridges),snaps:st.snaps,absorbed_duplicates:st.absorbed,hypotheses:st.hypotheses}:null,
    added_input_buffer_ms:0,continuity_bridge_ms:kind==='v2'?100:V3.POLICY.bridgeMs};
}
const result={schema:'replay_display_comparison.v3',conditions:'60 Hz deterministic; current-forward view; 45 m radius; building withholding off in both; v2 consolidation on (its default). GT scored at the GT capture instant using the display as shown then; GT never enters either engine.',sequences:{}};
for(const name of ['departure','turn','stop','avenue']){const clip=JSON.parse(fs.readFileSync(path.join(clipsDir,name+'.json')));
  result.sequences[name]={v2:run('v2',clip,gt.clips[name]),v3:run('v3',clip,gt.clips[name])};}
if(outPath)fs.writeFileSync(outPath,JSON.stringify(result,null,2)+'\n');
const R=(x,d=2)=>x==null?null:+x.toFixed(d);
const pick=m=>({glyphs:+m.mean_glyphs.toFixed(1),est:m.mean_established&&+m.mean_established.toFixed(1),tent:m.mean_tentative&&+m.mean_tentative.toFixed(1),
  appear_s:+m.glyph_appearances_per_s.toFixed(1),life50:m.glyph_lifetime_ms.p50&&Math.round(m.glyph_lifetime_ms.p50),ids:m.display_ids_created,estIds:m.established_created,
  cov:+m.gt_coverage.any_glyph_3m.toFixed(3),covEst:m.gt_coverage.established_3m&&+m.gt_coverage.established_3m.toFixed(3),unsup:+m.glyphs_without_any_real_object_3m.fraction.toFixed(3),
  unsupEst:m.glyphs_without_any_real_object_3m.established_fraction&&+m.glyphs_without_any_real_object_3m.established_fraction.toFixed(3),
  switch:m.gt_actor_glyph_identity_switch_rate&&+m.gt_actor_glyph_identity_switch_rate.toFixed(3),multi:+m.gt_actor_multiple_glyphs_fraction.toFixed(2),
  pspd50:R(m.stationary_gt_actor_same_glyph_apparent_speed_mps.p50),pspd95:R(m.stationary_gt_actor_same_glyph_apparent_speed_mps.p95),rot95:m.stationary_gt_actor_glyph_axis_rotation_deg_per_gt_frame.p95!=null&&+m.stationary_gt_actor_glyph_axis_rotation_deg_per_gt_frame.p95.toFixed(1),flips:+m.nose_flips_per_min.toFixed(1),
  step95:+m.step_m.p95.toFixed(3),lag50:m.moving_actor_along_track_error_m.p50&&+m.moving_actor_along_track_error_m.p50.toFixed(2),
  confirm50:m.confirmation&&m.confirmation.arrival_delay_ms.p50});
for(const [k,v] of Object.entries(result.sequences))console.log(k,'\n  v2',JSON.stringify(pick(v.v2)),'\n  v3',JSON.stringify(pick(v.v3)));
