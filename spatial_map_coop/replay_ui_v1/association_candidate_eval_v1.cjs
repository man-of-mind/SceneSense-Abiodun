// ACCEPTED engine (frozen v20.2/v20.3) vs ONE association candidate, identical causal inputs, 60 Hz, 45 m, current view.
// GT is read ONLY here (interpolated to the display instant) to score what each display draws; it never enters an engine.
// Adds identity metrics to motion_smoothing_eval_v1: switches, display IDs per GT actor, duplicate glyphs, track purity.
// Usage: node association_candidate_eval_v1.cjs <out.json> <dir:clip> ...   (env BEFORE / AFTER = engine modules)
'use strict';
const fs=require('fs'),path=require('path');
const A=require(process.env.BEFORE||path.join(__dirname,'baseline_v20_2/replay_tracks_v3.js'));
const B=require(process.env.AFTER||path.join(__dirname,'ped_ray_gate_candidate_v1.js'));
const [outPath,...specs]=process.argv.slice(2);
const HZ=60,DT=1000/HZ;
const q=(a,p)=>{if(!a.length)return null;const s=a.slice().sort((x,y)=>x-y);return s[Math.min(s.length-1,Math.floor(s.length*p))];};
const S=a=>({n:a.length,p50:q(a,.5),p95:q(a,.95),p99:q(a,.99),max:a.length?Math.max(...a):null});
const mean=a=>a.length?a.reduce((s,x)=>s+x,0)/a.length:null;
function gtActors(frames){const per={};for(const f of Object.values(frames))for(const p of f.positives)(per[p.id]=per[p.id]||[]).push({t:f.capture_ms,x:p.x,y:p.y,label:p.label});
  for(const v of Object.values(per)){v.sort((a,b)=>a.t-b.t);v.forEach((o,i)=>{const a=v[Math.max(0,i-1)],b=v[Math.min(v.length-1,i+1)],dt=(b.t-a.t)/1000;o.vx=dt>0?(b.x-a.x)/dt:0;o.vy=dt>0?(b.y-a.y)/dt:0;});}
  return per;}
function gtAt(v,t){let i=0;while(i<v.length&&v[i].t<t)i++;if(i===0||i>=v.length)return v[i]&&Math.abs(v[i].t-t)<1?v[i]:null;
  const a=v[i-1],b=v[i];if(b.t-a.t>400)return null;const k=(t-a.t)/(b.t-a.t);
  return {x:a.x+(b.x-a.x)*k,y:a.y+(b.y-a.y)*k,vx:a.vx+(b.vx-a.vx)*k,vy:a.vy+(b.vy-a.vy)*k,label:a.label};}
const rows=view=>view.visible.map(r=>({key:r.key,cls:r.cls,tier:r.tier,motion:r.motion,missed:r.missed,alpha:r.alpha,conflict:!!r.conflict,displayYaw:r.displayYaw??null,x:r.pose.x,y:r.pose.y}));
const vehSig=rs=>rs.filter(r=>r.cls==='Vehicle').map(r=>[r.tier,r.motion,r.missed,r.alpha.toFixed(9),r.conflict,r.displayYaw===null?'n':r.displayYaw.toFixed(6),r.x.toFixed(9),r.y.toFixed(9)].join()).sort().join('|');

function runClip(dir,name){
  const clip=JSON.parse(fs.readFileSync(path.join(dir,name+'.json'))),geo=JSON.parse(fs.readFileSync(path.join(dir,'geometry.json')));
  const actors=gtActors(JSON.parse(fs.readFileSync(path.join(dir,'gt_audit.json'))).clips[name]||{});
  const eng={before:new A.TrackEngine({roads:geo}),after:new B.TrackEngine({roads:geo})};
  const eq={ticks:0,vehicleRowsDiffer:0,pedShownBefore:0,pedShownAfter:0,pedEstablishedBefore:0,pedEstablishedAfter:0};
  const mk=()=>({prev:new Map(),steps:{Vehicle:[],Pedestrian:[]},groups:{},series:new Map(),trackGt:new Map(),gt:{Vehicle:0,Pedestrian:0},shown:{Vehicle:0,Pedestrian:0},dup:{Vehicle:0,Pedestrian:0},
    stillSpeed:{Vehicle:[],Pedestrian:[]},movingAsMoving:0,movingPed:0,hist:new Map()});
  const acc={before:mk(),after:mk()},group=(a,k)=>a.groups[k]=a.groups[k]||{err:[],lagMs:[],jit:[]};
  let cursor=0;
  for(let now=0;now<=clip.end_ms;now+=DT){
    while(cursor<clip.events.length&&clip.events[cursor].arrival_ms<=now){const e=clip.events[cursor++];eng.before.observe(e);eng.after.observe(e);}
    const views={before:eng.before.view(now,{mode:'current',radius:45}),after:eng.after.view(now,{mode:'current',radius:45})};
    const R={before:rows(views.before),after:rows(views.after)};eq.ticks++;
    if(vehSig(R.before)!==vehSig(R.after))eq.vehicleRowsDiffer++;
    eq.pedShownBefore+=R.before.filter(r=>r.cls==='Pedestrian').length;eq.pedShownAfter+=R.after.filter(r=>r.cls==='Pedestrian').length;
    eq.pedEstablishedBefore+=R.before.filter(r=>r.cls==='Pedestrian'&&r.tier==='established').length;eq.pedEstablishedAfter+=R.after.filter(r=>r.cls==='Pedestrian'&&r.tier==='established').length;
    for(const side of ['before','after']){const a=acc[side],view=views[side],rs=R[side],next=new Map();
      for(const r of rs){next.set(r.key,r);const p=a.prev.get(r.key);if(p)a.steps[r.cls].push(Math.hypot(r.x-p.x,r.y-p.y));}a.prev=next;
      if(!view.ego)continue;
      for(const [id,v] of Object.entries(actors)){const g=gtAt(v,now);if(!g||Math.hypot(g.x-view.ego.x,g.y-view.ego.y)>45)continue;
        const cls=g.label==='person'?'Pedestrian':'Vehicle',sp=Math.hypot(g.vx,g.vy);
        const near=rs.filter(r=>r.cls===cls&&Math.hypot(r.x-g.x,r.y-g.y)<=3).sort((p,o)=>Math.hypot(p.x-g.x,p.y-g.y)-Math.hypot(o.x-g.x,o.y-g.y)),best=near[0];
        a.gt[cls]++;if(best)a.shown[cls]++;if(near.length>1)a.dup[cls]++;
        const ser=a.series.get(id)||[];ser.push({t:now,gx:g.x,gy:g.y,vx:g.vx,vy:g.vy,sp,cls,key:best?.key??null,x:best?.x,y:best?.y,motion:best?.motion});a.series.set(id,ser);
        if(!best)continue;
        const tg=a.trackGt.get(best.key)||new Map();tg.set(id,(tg.get(id)||0)+1);a.trackGt.set(best.key,tg);
        const state=cls==='Vehicle'?(sp>=2?'moving':sp<.3?'stopped':'slow'):(sp>=.3?'moving':'stopped');
        const G=group(a,cls+'/'+state),ex=best.x-g.x,ey=best.y-g.y;G.err.push(Math.hypot(ex,ey));
        if(sp>=.3)G.lagMs.push((ex*g.vx+ey*g.vy)/sp/sp*1000);
        if(cls==='Pedestrian'&&sp>=.5){a.movingPed++;if(best.motion==='moving')a.movingAsMoving++;}
        const last=a.hist.get(id);if(last&&last.key===best.key&&now-last.t<DT*1.5){G.jit.push(Math.hypot(ex-last.ex,ey-last.ey));
          if(sp<.3)a.stillSpeed[cls].push(Math.hypot(best.x-last.x,best.y-last.y)/((now-last.t)/1000));}
        a.hist.set(id,{t:now,key:best.key,ex,ey,x:best.x,y:best.y});}}
  }
  const out={clip:name,dir:path.basename(dir),ticks:eq.ticks,equivalence:eq};
  for(const side of ['before','after']){const a=acc[side],id={};
    for(const cls of ['Vehicle','Pedestrian']){let pairs=0,sw=0;const keysPer=[];
      for(const s of a.series.values()){if(s[0].cls!==cls)continue;const keys=new Set();
        for(let i=1;i<s.length;i++){if(s[i].key)keys.add(s[i].key);if(!s[i].key||!s[i-1].key||s[i].t-s[i-1].t>DT*1.5)continue;pairs++;if(s[i].key!==s[i-1].key)sw++;}
        if(keys.size)keysPer.push(keys.size);}
      // purity: share of a display track's matched ticks spent on its majority GT actor; tracks serving >=2 actors for >=300 ms each
      let tot=0,maj=0,multi=0;for(const [k,m] of a.trackGt){const c=[...m.values()];
        const any=[...m.keys()][0],clsOf=a.series.get(any)?.[0]?.cls;if(clsOf!==cls)continue;tot+=c.reduce((x,y)=>x+y,0);maj+=Math.max(...c);if(c.filter(n=>n>=18).length>=2)multi++;}
      id[cls]={gt_ticks:a.gt[cls],visible_3m:a.gt[cls]?a.shown[cls]/a.gt[cls]:null,duplicate_glyph_fraction:a.gt[cls]?a.dup[cls]/a.gt[cls]:null,
        switch_rate:pairs?sw/pairs:null,switches:sw,display_ids_per_gt_actor_mean:mean(keysPer),display_ids_total:keysPer.reduce((x,y)=>x+y,0),
        track_purity:tot?maj/tot:null,tracks_serving_two_actors:multi,stationary_glyph_speed_mps:S(a.stillSpeed[cls]),step_m:S(a.steps[cls])};}
    const groups={};for(const [k,G] of Object.entries(a.groups))groups[k]={error_m:S(G.err),lag_ms_p50:q(G.lagMs,.5),jitter_tick_m:S(G.jit)};
    out[side]={identity:id,groups,pedestrian_moving_shown_as_moving:a.movingPed?a.movingAsMoving/a.movingPed:null,events:events(a.series)};}
  out.after.engine=eng.after.rayStats||null;return out;
}
function events(series){const start=[],stop=[],turn=[];
  for(const [id,s] of series){const cls=s[0].cls,go=cls==='Vehicle'?1:.5;
    const held=(i,from,to,pred)=>{const t0=s[i].t,w=s.filter(x=>x.t-t0>=from&&x.t-t0<=to);return w.length>=(to-from)/DT*.8&&w.every(x=>pred(x.sp));};
    for(let i=1;i<s.length;i++){
      if(s[i-1].sp<.3&&s[i].sp>=.3&&held(i,-700,-DT,v=>v<.3)&&held(i,300,1000,v=>v>=go)){
        const o=s.slice(i).find(x=>x.key);if(!o)continue;let d=null,m=null;
        for(const x of s.slice(i)){if(x.t-s[i].t>4000)break;if(!x.key)continue;if(m===null&&x.motion==='moving')m=x.t-s[i].t;const gd=Math.hypot(x.gx-s[i].gx,x.gy-s[i].gy);if(gd<.5||d!==null)continue;
          if(((x.x-o.x)*(x.gx-s[i].gx)+(x.y-o.y)*(x.gy-s[i].gy))/gd>=.5*gd)d=x.t-s[i].t;}
        start.push({id,cls,t:s[i].t,displacement_delay_ms:d,moving_state_delay_ms:m});}
      if(s[i-1].sp>=.3&&s[i].sp<.3&&held(i,-1000,-300,v=>v>=go)&&held(i,DT,700,v=>v<.3)){
        const ux=s[i-1].vx/s[i-1].sp,uy=s[i-1].vy/s[i-1].sp;let over=0;
        for(const x of s.slice(i)){if(x.t-s[i].t>3000)break;if(x.key)over=Math.max(over,(x.x-s[i].gx)*ux+(x.y-s[i].gy)*uy);}
        stop.push({id,cls,t:s[i].t,overshoot_m:over});}}
    if(cls==='Vehicle')for(let i=1;i<s.length;i++){const a=s[i-1],b=s[i];if(a.sp<2||b.sp<2||!b.key)continue;
      const da=Math.abs(((Math.atan2(b.vy,b.vx)-Math.atan2(a.vy,a.vx))*180/Math.PI+540)%360-180)/((b.t-a.t)/1000);if(da>=10)turn.push(Math.hypot(b.x-b.gx,b.y-b.gy));}}
  return {start,stop,turn_error_m:S(turn)};}

const result={schema:'display_association_candidate_eval.v1',conditions:'60 Hz lockstep, identical recorded inputs, current view, 45 m, building withholding off (app-level), roads as in the app. GT interpolated to the display instant; nearest same-class glyph within 3 m; duplicates = >1 same-class glyph within 3 m. GT never enters either engine.',
  before:process.env.BEFORE||'baseline_v20_2/replay_tracks_v3.js',after:process.env.AFTER||'ped_ray_gate_candidate_v1.js',clips:[]};
for(const spec of specs){const i=spec.lastIndexOf(':');result.clips.push(runClip(spec.slice(0,i),spec.slice(i+1)));}
if(outPath&&outPath!=='-')fs.writeFileSync(outPath,JSON.stringify(result,null,1)+'\n');
const f=(x,d=2)=>x===null||x===undefined?' - ':(+x).toFixed(d);
for(const c of result.clips){const e=c.equivalence,a=c.before,b=c.after;
  console.log(`\n== ${c.dir}/${c.clip} | vehicle rows differ ${e.vehicleRowsDiffer}/${e.ticks} ticks | ped glyph-ticks ${e.pedShownBefore}->${e.pedShownAfter} (established ${e.pedEstablishedBefore}->${e.pedEstablishedAfter}) | ray-only gated pairs ${b.engine?.pedestrianStrongPairsGatedOnlyByRay}`);
  for(const cls of ['Pedestrian','Vehicle']){const x=a.identity[cls],y=b.identity[cls];
    console.log(`  ${cls.padEnd(10)} vis ${f(x.visible_3m,3)}->${f(y.visible_3m,3)} dup ${f(x.duplicate_glyph_fraction,3)}->${f(y.duplicate_glyph_fraction,3)} switch ${f(x.switch_rate,4)}->${f(y.switch_rate,4)} (${x.switches}->${y.switches}) ids/actor ${f(x.display_ids_per_gt_actor_mean)}->${f(y.display_ids_per_gt_actor_mean)} purity ${f(x.track_purity,3)}->${f(y.track_purity,3)} 2-actor tracks ${x.tracks_serving_two_actors}->${y.tracks_serving_two_actors} | step p95 ${f(x.step_m.p95,3)}->${f(y.step_m.p95,3)} p99 ${f(x.step_m.p99,3)}->${f(y.step_m.p99,3)} max ${f(x.step_m.max)}->${f(y.step_m.max)} | still spd p50 ${f(x.stationary_glyph_speed_mps.p50)}->${f(y.stationary_glyph_speed_mps.p50)} p95 ${f(x.stationary_glyph_speed_mps.p95)}->${f(y.stationary_glyph_speed_mps.p95)}`);}
  for(const k of Object.keys(a.groups).sort()){const x=a.groups[k],y=b.groups[k];if(!y)continue;
    console.log(`    ${k.padEnd(20)} n ${x.error_m.n}->${y.error_m.n} err50 ${f(x.error_m.p50)}->${f(y.error_m.p50)} err95 ${f(x.error_m.p95)}->${f(y.error_m.p95)} jit95 ${f(x.jitter_tick_m.p95,3)}->${f(y.jitter_tick_m.p95,3)} jit99 ${f(x.jitter_tick_m.p99,3)}->${f(y.jitter_tick_m.p99,3)} lag50 ${f(x.lag_ms_p50,0)}->${f(y.lag_ms_p50,0)}`);}
  console.log(`  ped moving shown as moving ${f(a.pedestrian_moving_shown_as_moving,3)}->${f(b.pedestrian_moving_shown_as_moving,3)} | starts ${JSON.stringify(a.events.start.map(s=>[s.cls[0],s.displacement_delay_ms&&Math.round(s.displacement_delay_ms),s.moving_state_delay_ms&&Math.round(s.moving_state_delay_ms)]))}->${JSON.stringify(b.events.start.map(s=>[s.cls[0],s.displacement_delay_ms&&Math.round(s.displacement_delay_ms),s.moving_state_delay_ms&&Math.round(s.moving_state_delay_ms)]))} | stops overshoot ${JSON.stringify(a.events.stop.map(s=>+s.overshoot_m.toFixed(2)))}->${JSON.stringify(b.events.stop.map(s=>+s.overshoot_m.toFixed(2)))} | turn err50 ${f(a.events.turn_error_m.p50)}->${f(b.events.turn_error_m.p50)}`);}
