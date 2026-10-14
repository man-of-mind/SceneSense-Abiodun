// Deterministic pedestrian-motion fixtures for the v3 display tracker.
// Truth scores the display; it never enters the tracker. Noise model measured on
// the registered capture (confident pedestrian detections vs archived GT, route-wide):
//   radial sd 0.96 m (MAD 0.83), lag-240 ms radial correlation 0.42, tangential sd 0.23 (MAD 0.09),
//   confident-miss rate 0.42 with runs of 1-6 GT frames, 10% outliers > 1.5 m.
// Cadence: 120 ms capture interval, 186 ms capture-to-arrival (replay P50).
// Usage: node ped_motion_fixtures_v3.cjs [policyVersion v3.0|v3.1] [out.json]
'use strict';
const fs=require('fs'),T=require(process.env.TRACKS_MODULE||'../viewer_assets/replay_tracks_v3.js'); // TRACKS_MODULE: evaluation-only engine override
const version=process.argv[2]||null,outPath=process.argv[3];
const DT=120,AGE=186,HZ=60,SEEDS=30;
const NOISE={radialSd:.9,rho120:.65,tangSd:.15,outlierP:.08,outlierM:[1.5,3],missStart:.25,missStay:.55,weakWhenMissed:.5};
function rng(seed){let s=seed>>>0||1;return ()=>{s^=s<<13;s>>>=0;s^=s>>17;s^=s<<5;s>>>=0;return s/4294967296;};}
function gauss(r){let u=0,v=0;while(!u)u=r();v=r();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);}
// A path is a function t(ms) -> {x,y} plus an ego function t -> {x,y}. Ego faces +x.
const at=(x,y)=>({x,y});
function walk(segments){ // [{ms, vx, vy}] piecewise-constant velocity from start point
  return (start)=>t=>{let x=start.x,y=start.y,tt=0;for(const s of segments){const d=Math.min(Math.max(t-tt,0),s.ms);x+=s.vx*d/1000;y+=s.vy*d/1000;tt+=s.ms;}return at(x,y);};}
const still=ms=>({ms,vx:0,vy:0});
function fixtures(){const L=(v,ms)=>({ms,vx:0,vy:v}),Rd=(v,ms)=>({ms,vx:v,vy:0}),f=[];
  f.push({name:'stationary',truth:walk([still(8000)])(at(20,3)),moving:()=>false,dur:8000});
  for(const v of [.3,.6,1.0,1.5]){
    f.push({name:`lateral ${v} new`,truth:walk([L(v,8000)])(at(20,-2)),moving:()=>true,dur:8000,v,dir:'lateral',kind:'new'});
    f.push({name:`radial ${v} new`,truth:walk([Rd(-v,8000)])(at(24,2)),moving:()=>true,dur:8000,v,dir:'radial',kind:'new'});
    f.push({name:`lateral ${v} established`,truth:walk([still(3000),L(v,6000)])(at(20,-2)),moving:t=>t>=3000,onset:3000,dur:9000,v,dir:'lateral',kind:'established'});
    f.push({name:`radial ${v} established`,truth:walk([still(3000),Rd(-v,6000)])(at(24,2)),moving:t=>t>=3000,onset:3000,dur:9000,v,dir:'radial',kind:'established'});}
  f.push({name:'stop-pause-resume lateral 1.2',truth:walk([L(1.2,3000),still(3000),L(1.2,3000)])(at(20,-3)),moving:t=>t<3000||t>=6000,stopAt:3000,resumeAt:6000,dur:9000,v:1.2,dir:'lateral'});
  f.push({name:'stop-pause-resume radial 1.2',truth:walk([Rd(-1.2,3000),still(3000),Rd(-1.2,3000)])(at(24,2)),moving:t=>t<3000||t>=6000,stopAt:3000,resumeAt:6000,dur:9000,v:1.2,dir:'radial'});
  f.push({name:'stationary, ego approaching 5 m/s',truth:walk([still(6000)])(at(40,3)),ego:t=>at(5*t/1000,0),moving:()=>false,dur:6000,rangeBias:.05});
  return f;}
function event(frame,t,ego,objs){const ns=String(1000000000+Math.round(t*1e6));return {session:'r/c',source:'a',frame_id:frame,capture_ms:t,arrival_ms:t+AGE,capture_ns:ns,
  clock_domain:'unix_wall_ns',age_mapping:'RECORDED_SAME_HOST_WALL',
  pose:{x:ego.x,y:ego.y,yaw:0,source:'a',frame_id:frame,capture_ns:ns,binding:['r','c','a',frame,ns],sensor:{x:ego.x+1.8,y:ego.y,yaw:0},fov_deg:120,range_m:120,
    camera_matrix:[[1,0,0,ego.x+1.8],[0,1,0,ego.y],[0,0,1,1.55],[0,0,0,1]]},
  objects:objs.map((o,i)=>({source:'a',track_id:'u'+i,frame_id:frame,capture_ns:ns,heading_valid:false,heading_yaw_deg:null,
    raw:{id:`a:${frame}:${i}`,frame_id:frame,type:'Pedestrian',score:o.score,location:{x:o.x,y:o.y,z:.9},dimensions:{length:.6,width:.6,height:1.7}}}))};}
function measurements(fx,seed,clean){const r=rng(seed*7919+13),out=[];let ar=0,miss=false;
  for(let t=0,k=1;t<=fx.dur;t+=DT,k++){const g=fx.truth(t),ego=fx.ego?fx.ego(t):at(0,0),sx=ego.x+1.8,sy=ego.y;
    let x=g.x,y=g.y,score=.6;
    if(!clean){const dx=x-sx,dy=y-sy,rr=Math.hypot(dx,dy),ux=dx/rr,uy=dy/rr;
      ar=NOISE.rho120*ar+Math.sqrt(1-NOISE.rho120**2)*gauss(r)*NOISE.radialSd;let rad=ar+(fx.rangeBias?fx.rangeBias*rr:0);
      if(r()<NOISE.outlierP)rad+=(r()<.5?-1:1)*(NOISE.outlierM[0]+r()*(NOISE.outlierM[1]-NOISE.outlierM[0]));
      const tan=gauss(r)*NOISE.tangSd;x+=ux*rad-uy*tan;y+=uy*rad+ux*tan;
      miss=miss?r()<NOISE.missStay:r()<NOISE.missStart;if(miss){if(r()<NOISE.weakWhenMissed)score=.1;else score=null;}}
    out.push({t,frame:k,ego,obj:score===null?[]:[{x,y,score}]});}
  return out;}
const q=(a,p)=>{if(!a.length)return null;const s=a.slice().sort((x,y)=>x-y);return s[Math.min(s.length-1,Math.floor(s.length*p))];};
function runOne(fx,seed,clean){const policy={...(version?{pedestrianMotion:version}:{}),...JSON.parse(process.env.PED_POLICY||'{}')};const e=new T.TrackEngine({policy}),ms=measurements(fx,seed,clean);
  let i=0,prev=null;const rec=[];const ids=new Set();
  for(let now=0;now<=fx.dur+AGE;now+=1000/HZ){while(i<ms.length&&ms[i].t+AGE<=now){const m=ms[i++];e.observe(event(m.frame,m.t,m.ego,m.obj));}
    const v=e.view(now,{radius:80}),row=v.visible.filter(r=>r.cls==='Pedestrian').sort((a,b)=>Math.hypot(a.pose.x-fx.truth(now).x,a.pose.y-fx.truth(now).y)-Math.hypot(b.pose.x-fx.truth(now).x,b.pose.y-fx.truth(now).y))[0];
    const g=fx.truth(now);if(row)ids.add(row.key);
    rec.push({now,shown:!!row,x:row?.pose.x,y:row?.pose.y,motion:row?.motion,key:row?.key,tier:row?.tier,gx:g.x,gy:g.y,step:row&&prev&&prev.key===row.key?Math.hypot(row.pose.x-prev.x,row.pose.y-prev.y):null});
    prev=row?{key:row.key,x:row.pose.x,y:row.pose.y}:null;}
  return {rec,ids:ids.size};}
function score(fx,runs){
  const firstVisible=r=>r.rec.find(s=>s.shown)?.now??null;
  const after=(r,t0)=>r.rec.filter(s=>s.now>=t0);
  const err=[],jumps=[],still=[],shown=[],ids=[],moveDelay=[],dispDelay=[],settle=[],resumeDelay=[],stillSpeed=[],unresolvedFrac=[],movingFrac=[];
  for(const r of runs){ids.push(r.ids);const fv=firstVisible(r);
    const live=r.rec.filter(s=>fv!==null&&s.now>=fv+300&&s.now<=fx.dur);shown.push(live.filter(s=>s.shown).length/Math.max(1,live.length));
    for(const s of r.rec)if(s.step!==null)jumps.push(s.step);
    const mv=live.filter(s=>s.shown&&fx.moving(s.now-0));for(const s of mv)err.push(Math.hypot(s.x-s.gx,s.y-s.gy));
    const st=live.filter(s=>s.shown&&!fx.moving(s.now));for(let k=1;k<st.length;k++)if(st[k].key===st[k-1].key&&st[k].now-st[k-1].now<20)stillSpeed.push(Math.hypot(st[k].x-st[k-1].x,st[k].y-st[k-1].y)*HZ);
    unresolvedFrac.push(live.filter(s=>s.shown&&s.motion==='unresolved').length/Math.max(1,live.filter(s=>s.shown).length));
    movingFrac.push(live.filter(s=>s.shown&&s.motion==='moving').length/Math.max(1,live.filter(s=>s.shown).length));
    // delay until supported movement is represented (state 'moving' sustained 300 ms) after onset
    const onset=fx.onset??(fx.kind==='new'?fv:null);
    const until=(t0,pred)=>{const xs=after(r,t0);for(let k=0;k<xs.length;k++){const win=xs.filter(s=>s.now>=xs[k].now&&s.now<xs[k].now+300);if(win.length>15&&win.every(pred))return xs[k].now-t0;}return null;};
    if(onset!==null&&fx.v){moveDelay.push(until(onset,s=>s.shown&&s.motion==='moving'));
      // display displaced >= half of truth displacement since onset, in the right direction
      const o=r.rec.find(s=>s.now>=onset&&s.shown);if(o){const tr=fx.truth(onset);let d=null;for(const s of after(r,onset)){if(!s.shown)continue;const gd=Math.hypot(s.gx-tr.x,s.gy-tr.y);
        if(gd<.5)continue;const proj=((s.x-o.x)*(s.gx-tr.x)+(s.y-o.y)*(s.gy-tr.y))/gd;if(proj>=.5*gd){d=s.now-onset;break;}}dispDelay.push(d);}}
    if(fx.stopAt){settle.push(until(fx.stopAt,s=>s.shown&&s.motion!=='moving'));resumeDelay.push(until(fx.resumeAt,s=>s.shown&&s.motion==='moving'));}}
  const nn=a=>a.filter(x=>x!==null),fail=a=>a.filter(x=>x===null).length;
  return {visible_fraction_p10:q(shown,.1),visible_fraction_mean:shown.reduce((a,b)=>a+b,0)/shown.length,display_ids_median:q(ids,.5),display_ids_max:Math.max(...ids),
    stationary_apparent_speed_p50:q(stillSpeed,.5),stationary_apparent_speed_p95:q(stillSpeed,.95),moving_error_p50:q(err,.5),moving_error_p95:q(err,.95),
    jump_p99:q(jumps,.99),jump_max:jumps.length?Math.max(...jumps):null,unresolved_fraction:unresolvedFrac.reduce((a,b)=>a+b,0)/unresolvedFrac.length,moving_state_fraction:movingFrac.reduce((a,b)=>a+b,0)/movingFrac.length,
    move_state_delay_p50:q(nn(moveDelay),.5),move_state_never:fail(moveDelay),displacement_delay_p50:q(nn(dispDelay),.5),displacement_never:fail(dispDelay),
    settle_delay_p50:q(nn(settle),.5),settle_never:fail(settle),resume_delay_p50:q(nn(resumeDelay),.5),resume_never:fail(resumeDelay),runs:runs.length};}
const result={schema:'pedestrian_motion_fixtures.v1',policy:version||'default',noise:NOISE,cadence:{dt_ms:DT,arrival_ms:AGE,hz:HZ,seeds:SEEDS},fixtures:{}};
for(const fx of fixtures()){
  const clean=score(fx,[runOne(fx,0,true)]),noisy=score(fx,Array.from({length:SEEDS},(_,s)=>runOne(fx,s+1,false)));
  result.fixtures[fx.name]={clean,noisy};}
if(outPath)fs.writeFileSync(outPath,JSON.stringify(result,null,2)+'\n');
const f=(x,d=2)=>x===null||x===undefined?'-':typeof x==='number'?x.toFixed(d):x;
console.log('policy',result.policy);
console.log('fixture'.padEnd(34),'| clean: moveDelay disp settle err50 | noisy: vis ids moving% unres% moveDelay(never) disp(never) settle(never) resume(never) err50 err95 still50 still95 jump99 max');
for(const [k,v] of Object.entries(result.fixtures)){const c=v.clean,n=v.noisy;
  console.log(k.padEnd(34),'|',f(c.move_state_delay_p50,0),f(c.displacement_delay_p50,0),f(c.settle_delay_p50,0),f(c.moving_error_p50),'|',
    f(n.visible_fraction_mean),n.display_ids_median,f(n.moving_state_fraction),f(n.unresolved_fraction),`${f(n.move_state_delay_p50,0)}(${n.move_state_never})`,`${f(n.displacement_delay_p50,0)}(${n.displacement_never})`,
    `${f(n.settle_delay_p50,0)}(${n.settle_never})`,`${f(n.resume_delay_p50,0)}(${n.resume_never})`,f(n.moving_error_p50),f(n.moving_error_p95),f(n.stationary_apparent_speed_p50),f(n.stationary_apparent_speed_p95),f(n.jump_p99),f(n.jump_max));}
