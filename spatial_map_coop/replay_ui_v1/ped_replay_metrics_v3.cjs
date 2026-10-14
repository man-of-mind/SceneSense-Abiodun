// Pedestrian display before/after (v3.0 vs v3.1) on recorded replay clips, 60 Hz.
// GT scores the display only. GT pedestrian speed comes from archived GT positions
// (the archived object_speed_mps field is 0 for every pedestrian).
// Usage: node ped_replay_metrics_v3.cjs <clipsDir> <gt_audit.json> [out.json]
'use strict';
const fs=require('fs'),path=require('path'),T=require('../viewer_assets/replay_tracks_v3.js');
const [clipsDir,gtPath,outPath]=process.argv.slice(2),gt=JSON.parse(fs.readFileSync(gtPath));
const CLIPS=['departure','turn','stop','avenue','ped_pause','ped_standing'],HZ=60;
const q=(a,p)=>{if(!a.length)return null;const s=a.slice().sort((x,y)=>x-y);return s[Math.min(s.length-1,Math.floor(s.length*p))];};
const S=a=>({n:a.length,p50:q(a,.5),p95:q(a,.95),max:a.length?Math.max(...a):null});
function gtSpeeds(frames){const per={};for(const f of Object.values(frames))for(const p of f.positives)if(p.label==='person')(per[p.id]=per[p.id]||[]).push({t:f.capture_ms,x:p.x,y:p.y});
  const sp={};for(const [id,v] of Object.entries(per)){v.sort((a,b)=>a.t-b.t);v.forEach((o,i)=>{const a=v[Math.max(0,i-1)],b=v[Math.min(v.length-1,i+1)];
    o.speed=b.t>a.t?Math.hypot(b.x-a.x,b.y-a.y)/((b.t-a.t)/1000):null;o.vx=b.t>a.t?(b.x-a.x)/((b.t-a.t)/1000):0;o.vy=b.t>a.t?(b.y-a.y)/((b.t-a.t)/1000):0;});sp[id]=v;}return sp;}
function run(version,clipName){const clip=JSON.parse(fs.readFileSync(path.join(clipsDir,clipName+'.json'))),frames=gt.clips[clipName],sp=gtSpeeds(frames);
  const e=new T.TrackEngine({policy:{pedestrianMotion:version}}),gf=Object.values(frames).sort((a,b)=>a.capture_ms-b.capture_ms);
  let c=0,g=0,prev=new Map();const out={gt:0,shown:0,shownEst:0,steps:[],movingErr:[],alongErr:[],movingShownMoving:0,movingN:0,slowSpeed:[],slowFalseMoving:0,slowN:0,obs:{}};
  for(let now=0;now<=clip.end_ms;now+=1000/HZ){while(c<clip.events.length&&clip.events[c].arrival_ms<=now)e.observe(clip.events[c++]);
    const v=e.view(now,{radius:45}),peds=v.visible.filter(r=>r.cls==='Pedestrian'),next=new Map();
    for(const r of peds){const p=prev.get(r.key);if(p)out.steps.push(Math.hypot(r.pose.x-p.x,r.pose.y-p.y));next.set(r.key,r.pose);}prev=next;
    while(g<gf.length&&gf[g].capture_ms<=now){const f=gf[g++];if(!v.ego||f.capture_ms<now-1000/HZ)continue;
      for(const p of f.positives){if(p.label!=='person'||Math.hypot(p.x-v.ego.x,p.y-v.ego.y)>45)continue;
        const s=(sp[p.id]||[]).find(o=>o.t===f.capture_ms);out.gt++;
        const near=peds.filter(r=>Math.hypot(r.pose.x-p.x,r.pose.y-p.y)<=3).sort((a,b)=>Math.hypot(a.pose.x-p.x,a.pose.y-p.y)-Math.hypot(b.pose.x-p.x,b.pose.y-p.y));
        const best=near[0];if(best)out.shown++;if(near.some(r=>r.tier==='established'))out.shownEst++;
        (out.obs[p.id]=out.obs[p.id]||[]).push({t:f.capture_ms,key:best?.key??null,x:best?.pose.x,y:best?.pose.y,motion:best?.motion,speed:s?.speed});
        if(!best||!s||s.speed===null)continue;
        if(s.speed>=.3){out.movingN++;out.movingErr.push(Math.hypot(best.pose.x-p.x,best.pose.y-p.y));if(best.motion==='moving')out.movingShownMoving++;
          const h=Math.hypot(s.vx,s.vy);out.alongErr.push(((best.pose.x-p.x)*s.vx+(best.pose.y-p.y)*s.vy)/h);}
        else{out.slowN++;if(best.motion==='moving')out.slowFalseMoving++;}}}}
  // per-actor identity continuity, stopped-actor glyph speed, and stop/start representation delays
  let pairs=0,switches=0;const stopDelay=[],startDelay=[];
  for(const o of Object.values(out.obs)){for(let i=1;i<o.length;i++){const a=o[i-1],b=o[i];if(b.t-a.t>300||!a.key||!b.key)continue;pairs++;if(a.key!==b.key)switches++;
      else if(b.speed!==undefined&&b.speed<.3&&a.speed<.3)out.slowSpeed.push(Math.hypot(b.x-a.x,b.y-a.y)/((b.t-a.t)/1000));}
    // events: GT state persisting >= 700 ms on both sides (brief dips are not stops)
    const hold=(i,pred,dir)=>{const t0=o[i].t;let ok=true,seen=false;for(let k=i;k>=0&&k<o.length;k+=dir){const dt=Math.abs(o[k].t-t0);if(dt>700)break;seen=dt>=450;if(o[k].speed===undefined||o[k].speed===null||!pred(o[k].speed))ok=false;}return ok&&seen;};
    for(let i=1;i<o.length;i++){
      if(o[i-1].speed>=.3&&o[i].speed<.3&&hold(i-1,v=>v>=.5,-1)&&hold(i,v=>v<.3,1)){const k=o.slice(i).find(x=>x.key&&x.motion!=='moving');stopDelay.push(k?k.t-o[i].t:null);}
      if(o[i-1].speed<.3&&o[i].speed>=.3&&hold(i-1,v=>v<.3,-1)&&hold(i,v=>v>=.5,1)){const k=o.slice(i).find(x=>x.key&&x.motion==='moving');startDelay.push(k?k.t-o[i].t:null);}}}
  return {gt_person_frames:out.gt,visible:out.shown/out.gt,visible_established:out.shownEst/out.gt,identity_switch_rate:pairs?switches/pairs:null,identity_pairs:pairs,
    moving_person_frames:out.movingN,moving_shown_as_moving:out.movingN?out.movingShownMoving/out.movingN:null,moving_error_m:S(out.movingErr),along_track_error_m:S(out.alongErr),
    slow_or_stopped_person_frames:out.slowN,slow_false_moving:out.slowN?out.slowFalseMoving/out.slowN:null,slow_glyph_apparent_speed_mps:S(out.slowSpeed),
    step_m:S(out.steps),step_p99:q(out.steps,.99),stop_events:stopDelay.length,stop_represent_delay_ms:stopDelay,start_events:startDelay.length,start_represent_delay_ms:startDelay};}
const result={schema:'pedestrian_replay_comparison.v1',conditions:'60 Hz, current-forward view, 45 m, glyph within 3 m of GT person at the GT capture instant; moving = GT speed >= 0.3 m/s from GT positions',clips:{}};
for(const c of CLIPS)result.clips[c]={'v3.0':run('v3.0',c),'v3.1':run('v3.1',c)};
if(outPath)fs.writeFileSync(outPath,JSON.stringify(result,null,2)+'\n');
const f=(x,d=2)=>x===null||x===undefined?'-':x.toFixed(d);
for(const [c,v] of Object.entries(result.clips))for(const k of ['v3.0','v3.1']){const m=v[k];
  console.log(c.padEnd(13),k,`gt ${m.gt_person_frames} vis ${f(m.visible)} est ${f(m.visible_established)} idsw ${f(m.identity_switch_rate)} | moving n${m.moving_person_frames} asMoving ${f(m.moving_shown_as_moving)} err50 ${f(m.moving_error_m.p50)} err95 ${f(m.moving_error_m.p95)} along50 ${f(m.along_track_error_m.p50)} | slow n${m.slow_or_stopped_person_frames} falseMove ${f(m.slow_false_moving)} spd50 ${f(m.slow_glyph_apparent_speed_mps.p50)} spd95 ${f(m.slow_glyph_apparent_speed_mps.p95)} | step99 ${f(m.step_p99,3)} max ${f(m.step_m.max)} | stop ${JSON.stringify(m.stop_represent_delay_ms.map(x=>x===null?null:Math.round(x)))} start ${JSON.stringify(m.start_represent_delay_ms.map(x=>x===null?null:Math.round(x)))}`);}
