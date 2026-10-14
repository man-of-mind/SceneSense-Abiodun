'use strict';
// Focused regression checks for the v3.1 pedestrian display-motion policy.
const assert=require('assert/strict'),T=require('../../viewer_assets/replay_tracks_v3.js');
let checks=0;const test=(n,f)=>{try{f();checks++;}catch(e){console.error('FAIL',n);throw e;}};
const DT=120,AGE=186;
function event(frame,t,objs){const ns=String(1000000000+t*1e6);return {session:'r/c',source:'a',frame_id:frame,capture_ms:t,arrival_ms:t+AGE,capture_ns:ns,
  clock_domain:'unix_wall_ns',age_mapping:'RECORDED_SAME_HOST_WALL',
  pose:{x:0,y:0,yaw:0,source:'a',frame_id:frame,capture_ns:ns,binding:['r','c','a',frame,ns],sensor:{x:1.8,y:0,yaw:0},fov_deg:120,range_m:120,camera_matrix:[[1,0,0,1.8],[0,1,0,0],[0,0,1,1.55],[0,0,0,1]]},
  objects:objs.map((o,i)=>({source:'a',track_id:'u'+i,frame_id:frame,capture_ns:ns,heading_valid:o.yaw!==undefined,heading_yaw_deg:o.yaw??null,
    raw:{id:`a:${frame}:${i}`,frame_id:frame,type:o.type||'Pedestrian',score:o.score??.6,location:{x:o.x,y:o.y,z:.9},dimensions:o.type==='Vehicle'?{length:4.8,width:1.9,height:1.5}:{length:.6,width:.6,height:1.7}}}))};}
const run=(path,n,policy={})=>{const e=new T.TrackEngine({policy});for(let k=0;k<n;k++){const o=path(k*DT,k);e.observe(event(k+1,k*DT,o?[o]:[]));}return e;};
const ped=e=>[...e.tracks.values()].find(t=>t.cls==='Pedestrian');
const wobble=k=>[0,.8,-.7,.9,-.6,.5,-.9,.7][k%8]; // radial (x) jitter, lateral small

test('clean slow lateral walker (0.6 m/s) is represented as moving; v3.0 never did',()=>{
  const path=t=>({x:20,y:-2+.6*t/1000});assert.equal(ped(run(path,13)).motion,'moving');assert.notEqual(ped(run(path,40,{pedestrianMotion:'v3.0'})).motion,'moving');});
test('jittering standing pedestrian is not forced to move and stays visible',()=>{
  const e=new T.TrackEngine();let moving=0,shown=0,n=0;
  for(let k=0;k<60;k++){e.observe(event(k+1,k*DT,[{x:20+wobble(k),y:3+.05*Math.sin(k)}]));const v=e.view(k*DT+AGE+30,{radius:60}).visible.filter(r=>r.cls==='Pedestrian');
    n++;if(v.length)shown++;if(v[0]?.motion==='moving')moving++;}
  assert.equal(moving,0);assert(shown/n>.95);});
test('insufficient evidence is "unresolved", not confirmed stationary, and is not extrapolated',()=>{
  const e=run(t=>({x:20+wobble(t/DT),y:3}),12),t=ped(e);assert.equal(t.motion,'unresolved');
  const v1=e.view(11*DT+AGE+200,{radius:60}).visible[0].pose,v2=e.view(11*DT+AGE+500,{radius:60}).visible[0].pose;assert(Math.hypot(v1.x-v2.x,v1.y-v2.y)<1e-9);});
test('standing pedestrian: with the measured 0.9 m range-noise prior the speed bound stays unresolved (never moving); stationary is reachable when range noise is small',()=>{
  assert.equal(ped(run(()=>({x:20,y:3}),20)).motion,'unresolved');
  assert.equal(ped(run(()=>({x:20,y:3}),20,{pedSigmaRadialM:.2})).motion,'stationary');});
test('a single 3 m range outlier neither starts motion nor jumps the glyph',()=>{
  const e=new T.TrackEngine();let prev=null,maxStep=0;
  for(let k=0;k<30;k++){e.observe(event(k+1,k*DT,[{x:20+(k===15?3:0)+wobble(k)*.3,y:3}]));
    for(let s=0;s<8;s++){const p=e.view(k*DT+AGE+s*15,{radius:60}).visible[0]?.pose;if(p&&prev)maxStep=Math.max(maxStep,Math.hypot(p.x-prev.x,p.y-prev.y));prev=p||prev;}}
  assert.notEqual(ped(e).motion,'moving');assert(maxStep<.5,String(maxStep));});
test('stop is reflected: moving walker settles to non-moving within 2 s',()=>{
  const path=t=>({x:20,y:-3+1.2*Math.min(t,3000)/1000});const e=new T.TrackEngine();let settled=null;
  for(let k=0;k<60;k++){e.observe(event(k+1,k*DT,[path(k*DT)]));if(k*DT>3000&&settled===null&&ped(e).motion!=='moving')settled=k*DT-3000;}
  assert(settled!==null&&settled<=2000,String(settled));});
test('brief missed updates keep identity and presence; prediction stays bounded',()=>{
  const e=new T.TrackEngine();for(let k=0;k<25;k++){if(k===14||k===15)e.observe(event(k+1,k*DT,[]));else e.observe(event(k+1,k*DT,[{x:20,y:-2+1.2*k*DT/1000}]));}
  assert.equal(e.stats.tentativeCreated,1);const t=ped(e);assert.equal(t.motion,'moving');
  const a=e.view(24*DT+AGE+400,{radius:60}).visible[0].pose,b=e.view(24*DT+AGE+560,{radius:60}).visible[0].pose;assert(Math.abs(b.y-a.y)<1e-6,'held beyond 500 ms horizon');
  assert.equal(e.view(24*DT+800,{radius:60}).visible.length,0);});
test('vehicle display is identical under v3.0 and v3.1 pedestrian policies',()=>{
  const path=(t,k)=>[{x:15+8*t/1000+wobble(k)*.4,y:.1*wobble(k+3),type:'Vehicle',yaw:0}];
  const go=pm=>{const e=new T.TrackEngine({policy:{pedestrianMotion:pm}});const out=[];for(let k=0;k<40;k++){e.observe(event(k+1,k*DT,path(k*DT,k)));out.push(JSON.stringify(e.view(k*DT+AGE+40,{radius:80}).visible.map(r=>[r.pose,r.yaw,r.motion,r.tier])));}return out;};
  assert.deepEqual(go('v3.1'),go('v3.0'));});
test('raw observations are not modified by the pedestrian path',()=>{const ev=event(1,0,[{x:20,y:3}]),before=JSON.stringify(ev);const e=new T.TrackEngine();e.observe(ev);e.observe(event(2,120,[{x:20.2,y:3}]));assert.equal(JSON.stringify(ev),before);});
console.log(JSON.stringify({status:'PASS',pedestrian_motion_checks:checks}));
