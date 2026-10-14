// Display motion: CURRENT (frozen v20.2) vs ONE candidate engine, identical causal inputs, 60 Hz, 45 m, current view.
// GT is read ONLY here, interpolated to each display instant, to score what each display draws. It never enters an engine.
// Usage: node motion_smoothing_eval_v1.cjs <out.json> <dir:clip> [<dir:clip> ...]
//   env BEFORE / AFTER = engine modules (default: baseline_v20_2/replay_tracks_v3.js vs motion_ema_candidate_v1.js)
//   each <dir> holds <clip>.json (recorded clip), geometry.json and gt_audit.json (gt_audit_v3 format).
'use strict';
const fs=require('fs'),path=require('path');
const A=require(process.env.BEFORE||path.join(__dirname,'baseline_v20_2/replay_tracks_v3.js'));
const B=require(process.env.AFTER||path.join(__dirname,'motion_ema_candidate_v1.js'));
const [outPath,...specs]=process.argv.slice(2);
const HZ=60,DT=1000/HZ;
const q=(a,p)=>{if(!a.length)return null;const s=a.slice().sort((x,y)=>x-y);return s[Math.min(s.length-1,Math.floor(s.length*p))];};
const S=a=>({n:a.length,p50:q(a,.5),p95:q(a,.95),p99:q(a,.99),max:a.length?Math.max(...a):null});
const mean=a=>a.length?a.reduce((s,x)=>s+x,0)/a.length:null;
const gtLabel=c=>c==='Pedestrian'?'person':'vehicle';

function gtActors(frames){ // per actor: time-sorted GT samples with finite-difference velocity
  const per={};for(const f of Object.values(frames))for(const p of f.positives)(per[p.id]=per[p.id]||[]).push({t:f.capture_ms,x:p.x,y:p.y,label:p.label});
  for(const v of Object.values(per)){v.sort((a,b)=>a.t-b.t);v.forEach((o,i)=>{const a=v[Math.max(0,i-1)],b=v[Math.min(v.length-1,i+1)],dt=(b.t-a.t)/1000;
    o.vx=dt>0?(b.x-a.x)/dt:0;o.vy=dt>0?(b.y-a.y)/dt:0;});}
  return per;}
function gtAt(v,t){ // linear interpolation between GT samples <= 400 ms apart; null outside
  let i=0;while(i<v.length&&v[i].t<t)i++;if(i===0||i>=v.length){return v[i]&&Math.abs(v[i].t-t)<1?v[i]:null;}
  const a=v[i-1],b=v[i];if(b.t-a.t>400)return null;const k=(t-a.t)/(b.t-a.t);
  return {x:a.x+(b.x-a.x)*k,y:a.y+(b.y-a.y)*k,vx:a.vx+(b.vx-a.vx)*k,vy:a.vy+(b.vy-a.vy)*k,label:a.label};}

function rows(view){return view.visible.map(r=>({key:r.key,id:r.id,cls:r.cls,tier:r.tier,motion:r.motion,missed:r.missed,alpha:r.alpha,conflict:r.conflict,
  displayYaw:r.displayYaw,orientation:r.orientation,x:r.pose.x,y:r.pose.y}));}

function runClip(dir,name){
  const clip=JSON.parse(fs.readFileSync(path.join(dir,name+'.json'))),geo=JSON.parse(fs.readFileSync(path.join(dir,'geometry.json')));
  const gt=JSON.parse(fs.readFileSync(path.join(dir,'gt_audit.json'))),actors=gtActors(gt.clips[name]||{});
  const eng={before:new A.TrackEngine({roads:geo}),after:new B.TrackEngine({roads:geo})};
  const eq={ticks:0,shownSetDiffers:0,attrDiffers:0,rowsCompared:0,posDiff:[],shownOnlyBefore:0,shownOnlyAfter:0};
  const acc={before:mk(),after:mk()};
  function mk(){return {prevRows:new Map(),prevStep:new Map(),steps:{Vehicle:[],Pedestrian:[]},second:{Vehicle:[],Pedestrian:[]},
    groups:{},hist:new Map(),pedGt:0,pedShown:0,vehGt:0,vehShown:0,series:new Map()};}
  const group=(a,k)=>a.groups[k]=a.groups[k]||{err:[],along:[],lagMs:[],lateral:[],jit:[],jit250:[]};
  let cursor=0;
  for(let now=0;now<=clip.end_ms;now+=DT){
    while(cursor<clip.events.length&&clip.events[cursor].arrival_ms<=now){const e=clip.events[cursor++];eng.before.observe(e);eng.after.observe(e);}
    const views={before:eng.before.view(now,{mode:'current',radius:45}),after:eng.after.view(now,{mode:'current',radius:45})};
    // ---- equivalence: only drawn position may differ -------------------------------------------------------------
    const ra=new Map(rows(views.before).map(r=>[r.key,r])),rb=new Map(rows(views.after).map(r=>[r.key,r]));eq.ticks++;
    let differs=false;for(const k of ra.keys())if(!rb.has(k)){differs=true;eq.shownOnlyBefore++;}
    for(const k of rb.keys())if(!ra.has(k)){differs=true;eq.shownOnlyAfter++;}if(differs)eq.shownSetDiffers++;
    for(const [k,x] of ra){const y=rb.get(k);if(!y)continue;eq.rowsCompared++;
      if(['id','cls','tier','motion','missed','alpha','conflict','orientation'].some(f=>x[f]!==y[f])||(x.displayYaw??null)!==(y.displayYaw??null))eq.attrDiffers++;
      eq.posDiff.push(Math.hypot(x.x-y.x,x.y-y.y));}
    // ---- per engine: GT-free steps, GT-scored error/jitter/lag --------------------------------------------------
    for(const side of ['before','after']){const a=acc[side],view=views[side],rs=rows(view),next=new Map();
      for(const r of rs){next.set(r.key,r);const p=a.prevRows.get(r.key);if(!p)continue;const dx=r.x-p.x,dy=r.y-p.y;a.steps[r.cls].push(Math.hypot(dx,dy));
        const ps=a.prevStep.get(r.key);if(ps)a.second[r.cls].push(Math.hypot(dx-ps[0],dy-ps[1]));a.prevStep.set(r.key,[dx,dy]);}
      a.prevRows=next;if(!view.ego)continue;
      for(const [id,v] of Object.entries(actors)){const g=gtAt(v,now);if(!g||Math.hypot(g.x-view.ego.x,g.y-view.ego.y)>45)continue;
        const cls=g.label==='person'?'Pedestrian':'Vehicle',sp=Math.hypot(g.vx,g.vy);
        const best=rs.filter(r=>r.cls===cls&&Math.hypot(r.x-g.x,r.y-g.y)<=3).sort((p,o)=>Math.hypot(p.x-g.x,p.y-g.y)-Math.hypot(o.x-g.x,o.y-g.y))[0];
        if(cls==='Pedestrian'){a.pedGt++;if(best)a.pedShown++;}else{a.vehGt++;if(best)a.vehShown++;}
        const ser=a.series.get(id)||[];ser.push({t:now,gx:g.x,gy:g.y,vx:g.vx,vy:g.vy,sp,cls,key:best?.key??null,x:best?.x,y:best?.y});a.series.set(id,ser);
        if(!best)continue;
        const state=cls==='Vehicle'?(sp>=2?'moving':sp<.3?'stopped':'slow'):(sp>=.3?'moving':'stopped');
        const G=group(a,cls+'/'+state),ex=best.x-g.x,ey=best.y-g.y;G.err.push(Math.hypot(ex,ey));
        if(sp>=.3){const ux=g.vx/sp,uy=g.vy/sp,al=ex*ux+ey*uy;G.along.push(al);G.lagMs.push(al/sp*1000);G.lateral.push(Math.abs(-ex*uy+ey*ux));}
        const h=a.hist.get(id)||[];const last=h[h.length-1];
        if(last&&last.key===best.key&&now-last.t<DT*1.5)G.jit.push(Math.hypot(ex-last.ex,ey-last.ey));
        const old=h.find(o=>Math.abs(now-250-o.t)<DT/2);if(old&&old.key===best.key&&h.slice(h.indexOf(old)).every(o=>o.key===best.key))G.jit250.push(Math.hypot(ex-old.ex,ey-old.ey));
        h.push({t:now,key:best.key,ex,ey});while(h.length&&now-h[0].t>400)h.shift();a.hist.set(id,h);}}
  }
  const summary={};
  for(const side of ['before','after']){const a=acc[side];
    const groups={};for(const [k,G] of Object.entries(a.groups))groups[k]={error_m:S(G.err),along_track_m_mean:mean(G.along),along_track_m:S(G.along),lag_ms_p50:q(G.lagMs,.5),
      lateral_m:S(G.lateral),jitter_tick_m:S(G.jit),error_change_250ms_m:S(G.jit250)};
    summary[side]={groups,visibility:{pedestrian_gt_ticks:a.pedGt,pedestrian_shown_3m:a.pedGt?a.pedShown/a.pedGt:null,vehicle_gt_ticks:a.vehGt,vehicle_shown_3m:a.vehGt?a.vehShown/a.vehGt:null},
      display_step_m:{Vehicle:S(a.steps.Vehicle),Pedestrian:S(a.steps.Pedestrian)},second_difference_m:{Vehicle:S(a.second.Vehicle),Pedestrian:S(a.second.Pedestrian)},
      events:events(a.series),engine:side==='after'&&eng.after.emaStats?{...eng.after.emaStats}:null};}
  return {clip:name,dir:path.basename(dir),ticks:eq.ticks,equivalence:{...eq,posDiff:S(eq.posDiff)},...summary};
}
// Stop / start / turn response of the drawn position, per GT actor. An event is a crossing of 0.3 m/s with the
// previous state held for 700 ms before it and the new state held for 700 ms after a 300-ms transition.
function events(series){const start=[],stop=[],turn=[];
  for(const [id,s] of series){const cls=s[0].cls,go=cls==='Vehicle'?1:.5;
    const held=(i,from,to,pred)=>{const t0=s[i].t,w=s.filter(x=>x.t-t0>=from&&x.t-t0<=to);return w.length>=(to-from)/DT*.8&&w.every(x=>pred(x.sp));};
    for(let i=1;i<s.length;i++){
      if(s[i-1].sp<.3&&s[i].sp>=.3&&held(i,-700,-DT,v=>v<.3)&&held(i,300,1000,v=>v>=go)){ // START: display reaches 50% of true displacement
        const o=s.slice(i).find(x=>x.key);if(!o)continue;let d=null;
        for(const x of s.slice(i)){if(x.t-s[i].t>4000)break;if(!x.key)continue;const gd=Math.hypot(x.gx-s[i].gx,x.gy-s[i].gy);if(gd<.5)continue;
          if(((x.x-o.x)*(x.gx-s[i].gx)+(x.y-o.y)*(x.gy-s[i].gy))/gd>=.5*gd){d=x.t-s[i].t;break;}}
        start.push({id,cls,t:s[i].t,delay_ms:d});}
      if(s[i-1].sp>=.3&&s[i].sp<.3&&held(i,-1000,-300,v=>v>=go)&&held(i,DT,700,v=>v<.3)){ // STOP: overshoot past the stop point, settle
        const ux=s[i-1].vx/s[i-1].sp,uy=s[i-1].vy/s[i-1].sp;let over=0,settle=null,run=0,prev=null;
        for(const x of s.slice(i)){if(x.t-s[i].t>3000)break;if(!x.key){prev=null;continue;}over=Math.max(over,(x.x-s[i].gx)*ux+(x.y-s[i].gy)*uy);
          if(prev&&prev.key===x.key){const v=Math.hypot(x.x-prev.x,x.y-prev.y)/((x.t-prev.t)/1000);run=v<.3?run+(x.t-prev.t):0;if(run>=300&&settle===null)settle=x.t-300-s[i].t;}prev=x;}
        stop.push({id,cls,t:s[i].t,overshoot_m:over,settle_ms:settle});}}
    // TURN: vehicle moving >= 2 m/s with GT heading rate >= 10 deg/s; error of the drawn position there
    if(cls==='Vehicle')for(let i=1;i<s.length;i++){const a=s[i-1],b=s[i];if(a.sp<2||b.sp<2||!b.key)continue;
      const da=Math.abs(((Math.atan2(b.vy,b.vx)-Math.atan2(a.vy,a.vx))*180/Math.PI+540)%360-180)/((b.t-a.t)/1000);
      if(da>=10){const ux=b.vx/b.sp,uy=b.vy/b.sp,ex=b.x-b.gx,ey=b.y-b.gy;turn.push({err:Math.hypot(ex,ey),lateral:Math.abs(-ex*uy+ey*ux),along:ex*ux+ey*uy});}}}
  return {start:{n:start.length,never:start.filter(x=>x.delay_ms===null).length,delay_ms:S(start.filter(x=>x.delay_ms!==null).map(x=>x.delay_ms)),items:start},
    stop:{n:stop.length,overshoot_m:S(stop.map(x=>x.overshoot_m)),settle_ms:S(stop.filter(x=>x.settle_ms!==null).map(x=>x.settle_ms)),never_settled:stop.filter(x=>x.settle_ms===null).length,items:stop},
    turn:{n:turn.length,error_m:S(turn.map(x=>x.err)),lateral_m:S(turn.map(x=>x.lateral)),along_m_mean:mean(turn.map(x=>x.along))}};}

const result={schema:'display_motion_smoothing_eval.v1',conditions:'60 Hz lockstep, identical recorded inputs, current view, 45 m, building withholding off (app-level), roads set as in the app. GT interpolated to the display instant; nearest same-class glyph within 3 m. GT never enters either engine.',
  before:process.env.BEFORE||'baseline_v20_2/replay_tracks_v3.js',after:process.env.AFTER||'motion_ema_candidate_v1.js',candidate:B.EMA_CANDIDATE||null,clips:[]};
for(const spec of specs){const i=spec.lastIndexOf(':');result.clips.push(runClip(spec.slice(0,i),spec.slice(i+1)));}
if(outPath&&outPath!=='-')fs.writeFileSync(outPath,JSON.stringify(result,null,1)+'\n');
const f=(x,d=2)=>x===null||x===undefined?'  -  ':(+x).toFixed(d);
for(const c of result.clips){const e=c.equivalence;
  console.log(`\n== ${c.dir}/${c.clip}  ticks ${c.ticks} | equivalence: shown-set differs ${e.shownSetDiffers} ticks (only-before ${e.shownOnlyBefore}, only-after ${e.shownOnlyAfter}), attr differs ${e.attrDiffers}/${e.rowsCompared}, |dpos| p50 ${f(e.posDiff.p50,3)} p95 ${f(e.posDiff.p95,3)} max ${f(e.posDiff.max,3)}`);
  const keys=[...new Set([...Object.keys(c.before.groups),...Object.keys(c.after.groups)])].sort();
  for(const k of keys){const a=c.before.groups[k],b=c.after.groups[k];if(!a||!b)continue;
    console.log(`  ${k.padEnd(20)} n ${String(a.error_m.n).padStart(5)} | err50 ${f(a.error_m.p50)}->${f(b.error_m.p50)} err95 ${f(a.error_m.p95)}->${f(b.error_m.p95)} | jit95 ${f(a.jitter_tick_m.p95,3)}->${f(b.jitter_tick_m.p95,3)} jit99 ${f(a.jitter_tick_m.p99,3)}->${f(b.jitter_tick_m.p99,3)} | d250 p50 ${f(a.error_change_250ms_m.p50,3)}->${f(b.error_change_250ms_m.p50,3)} p95 ${f(a.error_change_250ms_m.p95,3)}->${f(b.error_change_250ms_m.p95,3)} | along ${f(a.along_track_m_mean)}->${f(b.along_track_m_mean)} lag50 ${f(a.lag_ms_p50,0)}->${f(b.lag_ms_p50,0)}ms`);}
  for(const cl of ['Vehicle','Pedestrian'])console.log(`  step ${cl.padEnd(10)} p99 ${f(c.before.display_step_m[cl].p99,3)}->${f(c.after.display_step_m[cl].p99,3)} max ${f(c.before.display_step_m[cl].max,3)}->${f(c.after.display_step_m[cl].max,3)} | 2nd-diff p95 ${f(c.before.second_difference_m[cl].p95,4)}->${f(c.after.second_difference_m[cl].p95,4)} p99 ${f(c.before.second_difference_m[cl].p99,4)}->${f(c.after.second_difference_m[cl].p99,4)}`);
  const v=c.before.visibility,w=c.after.visibility,E=c.before.events,F=c.after.events;
  console.log(`  visibility ped ${f(v.pedestrian_shown_3m,3)}->${f(w.pedestrian_shown_3m,3)} (n ${v.pedestrian_gt_ticks}) veh ${f(v.vehicle_shown_3m,3)}->${f(w.vehicle_shown_3m,3)}`);
  console.log(`  start n${E.start.n} delay p50 ${f(E.start.delay_ms.p50,0)}->${f(F.start.delay_ms.p50,0)} max ${f(E.start.delay_ms.max,0)}->${f(F.start.delay_ms.max,0)} never ${E.start.never}->${F.start.never} | stop n${E.stop.n} overshoot p50 ${f(E.stop.overshoot_m.p50)}->${f(F.stop.overshoot_m.p50)} max ${f(E.stop.overshoot_m.max)}->${f(F.stop.overshoot_m.max)} settle p50 ${f(E.stop.settle_ms.p50,0)}->${f(F.stop.settle_ms.p50,0)} | turn n${E.turn.n} err50 ${f(E.turn.error_m.p50)}->${f(F.turn.error_m.p50)} lat95 ${f(E.turn.lateral_m.p95)}->${f(F.turn.lateral_m.p95)} | ema ${JSON.stringify(c.after.engine)}`);}
