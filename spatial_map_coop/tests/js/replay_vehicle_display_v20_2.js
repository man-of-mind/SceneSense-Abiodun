'use strict';
// v20.2 vehicle DISPLAY orientation: glyph-only; association axis, positions and pedestrians untouched.
const assert=require('assert/strict'),T=require('../../viewer_assets/replay_tracks_v3.js');
let checks=0;const test=(n,f)=>{try{f();checks++;}catch(e){console.error(n);throw e;}};
const axis=(a,b)=>Math.abs(T.axisDelta(a,b));
const line=(x0,y0,x1,y1)=>{const n=Math.round(Math.hypot(x1-x0,y1-y0));return Array.from({length:n+1},(_,i)=>[x0+(x1-x0)*i/n,y0+(y1-y0)*i/n]);};
// One straight lane along +X at y=-3.5 (a parked car at y=0 is 3.5 m from its centreline), a crossing lane at x=60.
const ROADS={validation:{valid:true},roads:[line(-50,-3.5,100,-3.5),line(60,-40,60,40)]};
function fixture(policy={},at={x:20,y:0}){const e=new T.TrackEngine({roads:ROADS,policy}),t={cls:'Vehicle',hist:[],heading:null,motion:'stationary',speed:0,tstat:0,est:{vx:0,vy:0}};
 let time=0;return {e,t,step(yaw,{strong=true,moving=false,travel=0,speed=5,score=.7}={}){time+=140;
  t.hist.push({t:time,x:at.x,y:at.y,yaw,score:strong?score:.1,strong});if(t.hist.length>20)t.hist.shift();
  Object.assign(t,{motion:moving?'moving':'stationary',speed:moving?speed:0,tstat:moving?10:0,est:{vx:speed*Math.cos(travel*Math.PI/180),vy:speed*Math.sin(travel*Math.PI/180)}});
  e.updateHeading(t,time);return e.displayYaw({heading:t.display},time+2000);},get time(){return time;}};}

test('fewer than three qualified observations: orientation-neutral, never a guessed axis',()=>{
 const f=fixture();assert.equal(f.step(30),null);assert.equal(f.step(30),null);assert.equal(f.t.display.mode,'neutral');});
test('producer jitter around the lane axis draws the lane tangent, steadily',()=>{
 const f=fixture(),seen=[];for(const y of [18,-22,9,-15,24,-6,12,-19,3,21])seen.push(f.step(y));
 assert.equal(f.t.display.mode,'oriented');assert.match(f.t.display.source,/lane-tangent/);
 for(const y of seen.slice(2))assert(axis(y,0)<1e-6,'drawn axis is the lane, not the latest raw yaw');});
test('persistent near-perpendicular producer bias beside a lane: neutral, not diagonal and not forced straight',()=>{
 const f=fixture();for(let i=0;i<12;i++)f.step(i%2?84:-88);assert.equal(f.t.display.mode,'neutral');assert.match(f.t.display.source,/disagrees with lane/);
 assert(axis(f.t.heading.yaw,-88)<1e-6,'association axis keeps its own (held) value; the display rule does not edit it');});
test('consistent angled parking far from any lane keeps its observed angle',()=>{
 const f=fixture({},{x:20,y:30});let y;for(let i=0;i<8;i++)y=f.step(45+(i%2?3:-3));
 assert.equal(f.t.display.mode,'oriented');assert(axis(y,45)<4);assert.match(f.t.display.source,/no lane prior/);});
test('angled observation next to a lane is not snapped to the lane',()=>{
 const f=fixture();for(let i=0;i<10;i++)f.step(60+(i%2?2:-2));assert.equal(f.t.display.mode,'neutral');});
test('junction: no lane prior; inconsistent producer stays neutral',()=>{
 const f=fixture({},{x:60,y:-3.5});for(const y of [10,70,-40,25,85,-10,50,-70])f.step(y);
 assert.equal(f.t.display.mode,'neutral');assert.equal(f.t.display.road.junction,true);});
test('a real turn is followed through the qualified motion axis, not frozen to the lane',()=>{
 const f=fixture();for(let i=0;i<4;i++)f.step(0);
 let y;for(let a=0;a<=90;a+=10)y=f.step(null,{strong:false,moving:true,travel:a});
 assert.equal(f.t.display.mode,'oriented');assert.match(f.t.display.source,/motion axis/);assert(axis(y,90)<1,'display reaches the turned axis');});
test('stopping after motion is seeded by the trajectory axis, so the glyph stays oriented',()=>{
 const f=fixture({},{x:20,y:30});for(let i=0;i<5;i++)f.step(null,{strong:false,moving:true,travel:30});
 const y=f.step(null,{strong:false});assert.equal(f.t.display.mode,'oriented');assert(axis(y,30)<1);});
test('display policy never changes the association axis or heading state',()=>{
 const seq=[0,12,-30,85,-88,40,41,39,40,null,7];
 const a=fixture(),b=fixture({vehicleDisplay:'v20.1'});for(const y of seq){a.step(y);b.step(y);}
 const strip=h=>JSON.stringify({...h});assert.equal(strip(a.t.heading),strip(b.t.heading));assert.equal(b.t.display,undefined);});
test('missing or invalid geometry disables the lane prior',()=>{
 const e=new T.TrackEngine({roads:{roads:ROADS.roads}});assert.equal(e.roads.valid,false);assert.equal(e.roads.axisAt(20,0),null);});
test('pedestrian rows carry no vehicle display orientation; vehicles do',()=>{
 const e=new T.TrackEngine({roads:ROADS});
 const ev=(f,t)=>{const ns=String(1000000000+t*1e6);return {session:'r/c',source:'a',frame_id:f,capture_ms:t,arrival_ms:t+190,capture_ns:ns,clock_domain:'unix_wall_ns',age_mapping:'RECORDED_SAME_HOST_WALL',
  pose:{x:0,y:0,yaw:0,source:'a',frame_id:f,capture_ns:ns,binding:['r','c','a',f,ns],sensor:{x:1.8,y:0,yaw:0},fov_deg:120,range_m:120,camera_matrix:[[1,0,0,1.8],[0,1,0,0],[0,0,1,1.55],[0,0,0,1]]},
  objects:[['Pedestrian',12,1,{length:.6,width:.6,height:1.7}],['Vehicle',20,-6,{length:4.8,width:1.9,height:1.5}]].map(([type,x,y,dimensions],i)=>({source:'a',track_id:type,heading_valid:type==='Vehicle',heading_yaw_deg:type==='Vehicle'?3:null,frame_id:f,capture_ns:ns,
   raw:{id:`a:${f}:${i}`,frame_id:f,location:{x,y,z:.8},type,score:.9,dimensions}}))};};
 for(let i=1;i<7;i++)e.observe(ev(i,i*120));const rows=e.view(6*120+200,{radius:60}).visible;
 const ped=rows.filter(r=>r.cls==='Pedestrian'),veh=rows.filter(r=>r.cls==='Vehicle');assert(ped.length&&veh.length);
 for(const r of ped){assert.equal(r.orientation,null);assert.equal(r.displayYaw,r.yaw);}
 assert.equal(veh[0].orientation,'oriented');assert(axis(veh[0].displayYaw,0)<1e-6,'lane prior: observed 3 deg, lane 0 deg');});
console.log(JSON.stringify({status:'PASS',vehicle_display_v20_2_checks:checks}));
