'use strict';
// Focused checks for the v3 presentation-only display-track policies.
const assert=require('assert/strict'),T=require('../../viewer_assets/replay_tracks_v3.js');
let checks=0;const test=(n,f)=>{try{f();checks++;}catch(e){console.error('FAIL',n);throw e;}};
const dims={length:4.8,width:1.9,height:1.5};
function obj(id,x,y,score,type='Vehicle',yaw=null){return {source:'a',track_id:id,heading_valid:yaw!==null,heading_yaw_deg:yaw,
  raw:{id,location:{x,y,z:.8},type,score,dimensions:type==='Pedestrian'?{length:.6,width:.6,height:1.7}:dims}};}
function event(frame,t,objects){const ns=String(1000000000+t*1e6);return {session:'r/c',source:'a',frame_id:frame,capture_ms:t,arrival_ms:t+190,capture_ns:ns,
  clock_domain:'unix_wall_ns',age_mapping:'RECORDED_SAME_HOST_WALL',
  pose:{x:0,y:0,yaw:0,source:'a',frame_id:frame,capture_ns:ns,binding:['r','c','a',frame,ns],sensor:{x:1.8,y:0,yaw:0},fov_deg:120,range_m:120,
    camera_matrix:[[1,0,0,1.8],[0,1,0,0],[0,0,1,1.55],[0,0,0,1]]},
  objects:objects.map((o,i)=>({...o,frame_id:frame,capture_ns:ns,raw:{...o.raw,id:`a:${frame}:${i}`,frame_id:frame}}))};}
const run=(frames,engine=new T.TrackEngine())=>{frames.forEach((objs,i)=>engine.observe(event(i+1,i*120,objs)));return engine;};
const shown=(e,t)=>e.view(t,{radius:60}).visible;

test('one confident detection is tentative; two in three frames establish it',()=>{
  const e=run([[obj('c',20,0,.6)]]);assert.equal(shown(e,200)[0].tier,'tentative');
  e.observe(event(2,120,[obj('c',20.1,0,.5)]));assert.equal(shown(e,320)[0].tier,'established');assert.equal(e.stats.confirmArrivalMs[0],120);});
test('low-confidence candidates never create tracks, only the labelled hypothesis field',()=>{
  const e=run(Array.from({length:8},()=>[obj('w',20,3,.05)]));assert.equal(e.tracks.size,0);
  const v=e.view(900,{radius:60});assert(v.field.some(c=>c.shown));assert.equal(v.dispositions[0].disposition,'LOW_CONFIDENCE_HYPOTHESIS');});
test('a score dip below threshold sustains an established track without a new ID',()=>{
  const e=run([[obj('c',20,0,.6)],[obj('c',20,0,.6)],[obj('c',20.2,0,.1)],[obj('c',20,0,.1)],[obj('c',20.1,0,.7)]]);
  assert.equal(e.stats.tentativeCreated,1);assert.equal(shown(e,700).length,1);});
test('weak-only support ends an established track after the bounded run',()=>{
  const e=run([[obj('c',20,0,.6)],[obj('c',20,0,.6)],...Array.from({length:7},()=>[obj('c',20,0,.1)])]);assert.equal(e.stats.endedWeakOnly,1);});
test('a duplicate centred inside a track footprint is absorbed, adjacent parked cars stay separate',()=>{
  const e=run([[obj('a',20,0,.6,'Vehicle',0),obj('b',21.5,.3,.4,'Vehicle',0),obj('n',20,2.4,.5,'Vehicle',0),obj('f',25.6,0,.5,'Vehicle',0)]]);
  const d=Object.fromEntries(e.lastDispositions.map(x=>[x.raw.score,x.disposition]));
  assert.equal(d[.4],'OVERLAPS_TRACK_FOOTPRINT');assert.equal(d[.5],'NEW_TENTATIVE');assert.equal(e.tracks.size,3);});
test('raw observations are never modified',()=>{const ev=event(1,0,[obj('a',20,0,.6),obj('w',30,2,.05)]),before=JSON.stringify(ev);
  new T.TrackEngine().observe(ev);assert.equal(JSON.stringify(ev),before);});
test('a noisy parked car stays anchored: no drift, no extrapolation',()=>{
  const noise=[0,.7,-.6,.9,-.8,.5,-.4,.8,-.9,.6,-.7,.3,.9,-.5,.2,-.8];
  const e=run(noise.map((n,i)=>[obj('p',20+n,.3*Math.sin(i),.6,'Vehicle',0)]));
  const last=e.tracks.values().next().value;assert.equal(last.motion,'stationary');
  const a=shown(e,15*120+190+160)[0].pose,b=shown(e,15*120+600)[0].pose;assert(Math.hypot(a.x-b.x,a.y-b.y)<1e-9);assert(Math.abs(a.x-20)<.6);});
test('genuine motion is recognized and extrapolated within the bounded horizon',()=>{
  const e=run(Array.from({length:12},(_,i)=>[obj('m',15+i*.96,0,.6,'Vehicle',0)]));const t=e.tracks.values().next().value;
  assert.equal(t.motion,'moving');assert(Math.abs(t.speed-8)<1);
  const p1=shown(e,11*120+200)[0].pose.x,p2=shown(e,11*120+700)[0].pose.x;assert(p2-p1>1&&p2-p1<=8*.3+.3);});
test('range-bias drift while the ego approaches is not called motion',()=>{
  // object fixed at 30 m; ego moves 6 m/s toward it; detection range error grows 10% of range rate
  const e=new T.TrackEngine();for(let i=0;i<14;i++){const ex=i*.72,ev=event(i+1,i*120,[obj('s',30-.1*(30-ex)*.2,0,.6,'Vehicle',0)]);
    ev.pose.x=ex;ev.pose.sensor.x=ex+1.8;ev.pose.camera_matrix[0][3]=ex+1.8;e.observe(ev);}
  assert.notEqual([...e.tracks.values()][0].motion,'moving');});
test('stationary heading flip in the producer swaps nothing until sustained; body never spins',()=>{
  const yaws=[0,0,0,180,0,180,0,0,180,0];const e=run(yaws.map(y=>[obj('h',20,0,.6,'Vehicle',y)]));
  const t=e.tracks.values().next().value,axis=((t.heading.yaw%180)+180)%180;assert(axis<1||axis>179);});
test('missed updates fade then expire; nothing persists past 750 ms',()=>{
  const e=run([[obj('c',20,0,.6)],[obj('c',20,0,.6)]]);const r=shown(e,120+190+300)[0];assert.equal(r.missed,true);
  assert.equal(shown(e,120+800).length,0);});
test('two established tracks in one footprint are marked as a conflict group',()=>{
  const f=[obj('a',20,0,.6,'Vehicle',0),obj('b',23.2,0,.6,'Vehicle',0)];const e=run([f,f,f]);
  // both established before they overlap; then b's evidence stays inside a's footprint
  for(let i=3;i<8;i++)e.observe(event(i+1,i*120,[obj('a',20,0,.6,'Vehicle',0),obj('b',21.6,0,.6,'Vehicle',0)]));
  const v=shown(e,7*120+360);assert.equal(v.length,2);assert(v.every(r=>r.conflict&&r.conflict===v[0].conflict));});
test('out-of-order and duplicate events are rejected without effect',()=>{
  const e=run([[obj('c',20,0,.6)],[obj('c',20,0,.6)]]);const s=e.tracks.size;
  assert.equal(e.observe(event(1,0,[obj('c',40,0,.9)])),'OUT_OF_ORDER');assert.equal(e.tracks.size,s);});
test('bounded memory under a flood of candidates',()=>{const e=new T.TrackEngine();
  for(let f=0;f<60;f++)e.observe(event(f+1,f*120,Array.from({length:120},(_,i)=>obj('x',5+(i%12)*3.5+f*.01,-20+Math.floor(i/12)*4,.5))));
  assert(e.tracks.size<=T.POLICY.maxTracks);assert(e.field.size<=T.POLICY.fieldMaxCells);});
console.log(JSON.stringify({status:'PASS',v3_track_checks:checks}));
