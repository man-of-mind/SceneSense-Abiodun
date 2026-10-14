'use strict';
const assert=require('assert/strict');
const P=require('../../viewer_assets/presentation_predictor_v1.js');
const R=require('../../viewer_assets/replay_policy_v1.js');
let checks=0;
function test(name,fn){try{fn();checks++;}catch(e){console.error('FAIL',name);throw e;}}
function event(frame=1,time=0,objects=[{x:10,y:0}],extra={}){
 const source=extra.source||'a', capture_ns=String(1000000000+time*1e6);
 const e={source,session:'run/cell',frame_id:frame,capture_ms:time,arrival_ms:time+200,capture_ns,
 clock_domain:'unix_wall_ns',age_mapping:'RECORDED_SAME_HOST_WALL',...extra};
 e.pose={x:0,y:0,yaw:0,sensor:{x:0,y:0,yaw:0},fov_deg:90,range_m:40,
 source,frame_id:frame,capture_ns,binding:[e.session.split('/').slice(0,-1).join('/'),e.session.split('/').pop(),source,frame,capture_ns]};
 e.objects=objects.map((p,i)=>({raw:{id:source+':'+frame+':'+i,source_stream_id:source,frame_id:frame,
 location:{...p,z:0},type:'Vehicle',score:.02,dimensions:{length:4,width:2},yaw_deg:0},track_id:String(i),
 source,frame_id:frame,capture_ms:time,capture_ns,heading_valid:false}));return e;
}
const clone=x=>JSON.parse(JSON.stringify(x));
function freeze(x){if(x&&typeof x==='object'){Object.freeze(x);Object.values(x).forEach(freeze);}return x;}
test('front, behind, edge, range boundaries',()=>{
 for(const [x,y,reason]of[[10,0,null],[-1,0,'BEHIND_SENSOR'],[0,1,'BEHIND_SENSOR'],[10,10,null],[10,10.01,'OUTSIDE_HORIZONTAL_FOV'],[40,0,null],[40.01,0,'OUTSIDE_RANGE']])
 assert.equal(R.region({x,y},{x:0,y:0,yaw:0},90,40),reason);
});
test('yaw degrees and rotated mount',()=>{
 assert.equal(R.region({x:0,y:10},{x:0,y:0,yaw:90},90,40),null);
 assert.equal(R.region({x:10,y:0},{x:0,y:0,yaw:180},90,40),'BEHIND_SENSOR');
 assert.deepEqual(R.sensorAt({x:0,y:0,yaw:0,sensor:{x:2,y:0,yaw:0}},{x:10,y:20,yaw:90}),{x:10,y:22,yaw:90});
});
test('missing angle, range, timestamp or frame pose fail closed',()=>{
 for(const mutate of [e=>e.pose=null,e=>e.pose.fov_deg=null,e=>e.pose.range_m=null,e=>e.pose.frame_id++,e=>e.pose.capture_ns='bad',e=>e.capture_ms=null]){
 const e=event();mutate(e);assert.ok(R.poseReason(e));const s=new R.Engine();s.observe(e);assert.equal(s.view(210).visible.length,0);}
});
test('full 3D capture camera vertical frustum',()=>{
 const e=event();e.pose.camera_matrix=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
 const raw=e.objects[0].raw;raw.location.z=20;assert.equal(R.captureRegion(raw,e.pose),'OUTSIDE_VERTICAL_FOV');
});
test('capture geometry is checked before prediction',()=>{
 const s=new R.Engine();s.observe(event(1,0,[{x:-1,y:0}]));const v=s.view(210);assert.equal(v.visible.length,0);assert.equal(v.rows[0].reason,'BEHIND_EGO');
});
test('duplicate and out-of-order events cannot refresh age',()=>{
 const s=new R.Engine();s.observe(event(2,100));const before=JSON.stringify(s.predictor.states);
 assert.equal(s.observe(event(2,100,undefined,{arrival_ms:600})),'DUPLICATE');
 assert.equal(s.observe(event(1,0)),'OUT_OF_ORDER');assert.equal(JSON.stringify(s.predictor.states),before);
 assert.equal(s.view(900,{mode:'coast'}).visible.length,0);
});
test('same clock prearrival compensation & correction continuity',()=>{
 const p=new P.DisplayPredictor({horizonMs:500});
 p.observe('a',{x:0,y:0,yaw:0,measuredMs:0,ageAtArrivalMs:200},200);
 p.observe('a',{x:1,y:0,yaw:0,measuredMs:100,ageAtArrivalMs:200},300);
 assert.equal(p.sample('a',300).x,0); // continuous at adoption
 assert.ok(Math.abs(p.sample('a',420).x-4.2)<1e-10);
 assert.equal(p.sample('a',900).x,6);assert.equal(p.sample('a',900).held,true);
});
test('unmapped CARLA domain never gets wall age compensation',()=>{
 const e=event(1,100000000,undefined,{arrival_ms:20,clock_domain:'carla_simulation_ns',age_mapping:null});
 const s=new R.Engine();s.observe(e);const v=s.view(70);assert.equal(v.rows[0].ageMs,50);assert.equal(v.rows[0].pose.ageMs,null);
 assert.equal(v.clockMode,'arrival');assert.equal(v.visible.length,1);
 e.age_mapping='RECORDED_SAME_HOST_WALL';assert.equal(R.poseReason(e),'INVALID_CLOCK_MAPPING');
});
test('age advances with silence; strict excludes coast; fade bounded',()=>{
 const s=new R.Engine();s.observe(event());
 assert.equal(s.view(300).visible.length,1);assert.equal(s.view(351).visible.length,0);
 assert.equal(s.view(600,{mode:'coast'}).visible[0].alpha,.6);
 assert.equal(s.view(750,{mode:'coast'}).visible.length,0);
});
test('disappearing tracks are separate from current inventory',()=>{
 const s=new R.Engine();s.observe(event());s.observe(event(2,100,[]));
 assert.equal(s.view(310).visible.length,0);assert.equal(s.view(310,{mode:'coast'}).visible[0].state,'coasting');
});
test('display prediction is rechecked against moving display FoV',()=>{
 const s=new R.Engine(),a=event(1,0,[{x:4,y:2}]),b=event(2,100,[{x:4,y:2.5}]),c=event(3,200,[{x:4,y:3}]);
 s.observe(a);s.observe(b);s.observe(c);const v=s.view(540,{mode:'coast'});
 assert.equal(v.visible.length,0);assert.equal(v.rows[0].reason,'OUTSIDE_HORIZONTAL_FOV');
});
test('current display radius independent from sensor range',()=>{
 const s=new R.Engine();s.observe(event(1,0,[{x:35,y:0}]));assert.equal(s.view(210,{radius:20}).rows[0].reason,'OUTSIDE_DISPLAY_RADIUS');
});
test('selected source only and source change no cross velocity',()=>{
 const s=new R.Engine();s.observe(event());s.observe(event(1,0,[{x:30,y:0}],{source:'b'}));
 assert.equal(s.view(210).visible.length,1);assert.equal(s.view(210).rows[1].reason,'OTHER_SOURCE');
 s.source='b';assert.equal(s.view(210).visible[0].entry.source,'b');assert.equal(s.view(210).visible[0].pose.vx,0);
});
test('session changes and seeks reset identity and velocity',()=>{
 const s=new R.Engine();s.observe(event());s.observe(event(2,100,[{x:11,y:0}]));
 s.observe(event(1,0,[{x:30,y:0}],{session:'new/cell'}));assert.equal(s.tracks.size,1);assert.equal(s.view(210).visible[0].pose.vx,0);
 s.reset('a');s.observe(event());assert.equal(s.accepted,1);assert.equal(s.view(210).visible[0].pose.vx,0);
});
test('stopped motion does not drift; shortest turn wrap',()=>{
 const p=new P.DisplayPredictor({stationarySpeedMps:.25});
 p.observe('x',{x:0,y:0,yaw:179,measuredMs:0},0);p.observe('x',{x:.01,y:0,yaw:-179,measuredMs:100},100);
 const sample=p.sample('x',250);assert.equal(sample.vx,0);assert.ok(sample.yaw>-181&&sample.yaw<-175);
});
test('track teleport resets instead of implying extreme motion',()=>{
 const p=new P.DisplayPredictor({teleportDistanceM:5});p.observe('x',{x:0,y:0,yaw:0,measuredMs:0},0);
 assert.equal(p.observe('x',{x:20,y:0,yaw:0,measuredMs:100},100),'RESET');assert.equal(p.sample('x',120).vx,0);
});
test('FoV is unknown visibility; strict visible excludes unknown',()=>{
 const s=new R.Engine();s.observe(event());assert.equal(s.view(210).visible[0].visibility.state,'unknown');
 assert.equal(s.view(210,{mode:'visible'}).rows[0].reason,'VISIBILITY_UNKNOWN');
});
test('reviewed visible/occluded evidence is bound to exact measurement',()=>{
 const e=event(),o=e.objects[0];
 const proof={qualified:true,method:'FRAME_BOUND_REVIEW',source:o.source,frame_id:o.frame_id,capture_ns:o.capture_ns,
 observation_id:o.raw.id,evidence_ref:'fixture-only',pose_binding:JSON.stringify(e.pose.binding)};
 for(const state of ['visible','occluded']){o.visibility={...proof,state};assert.equal(R.visibility(o,e).state,state);}
 o.visibility.frame_id++;assert.equal(R.visibility(o,e).state,'unknown');
 o.visibility={...proof,state:'visible',method:'CENTER_RAY'};assert.equal(R.visibility(o,e).state,'unknown');
});
test('input observations immutable; low scores kept; per-object reasons',()=>{
 const e=freeze(event()),before=JSON.stringify(e),s=new R.Engine();s.observe(e);const v=s.view(230);
 assert.equal(v.visible.length,1);assert.equal(JSON.stringify(e),before);assert.equal(v.visible[0].entry.raw.score,.02);
});
test('bounded object and source queues',()=>{
 const s=new R.Engine();s.observe(event(1,0,Array.from({length:500},(_,i)=>({x:10,y:i/100}))));
 assert.equal(s.tracks.size,R.SETTINGS.maxTracks);assert.equal(s.predictor.size(),R.SETTINGS.maxTracks);
 for(let i=0;i<20;i++)s.observe(event(1,0,[],{source:'source'+i}));assert.ok(s.latest.size<=8);assert.ok(s.egoPredictor.size()<=8);
});
test('noisy two-sample displacement is insufficient motion evidence',()=>{
 const p=new P.DisplayPredictor({velocityConsistencyMps:6,stationarySpeedMps:.25});
 p.observe('x',{x:0,y:0,yaw:0,measuredMs:0},0);
 p.observe('x',{x:1,y:0,yaw:0,measuredMs:100},100);assert.equal(p.sample('x',200).vx,0);
 p.observe('x',{x:2,y:0,yaw:0,measuredMs:200},200);assert.equal(p.sample('x',300).vx,10);
 p.observe('x',{x:1,y:0,yaw:0,measuredMs:300},300);assert.equal(p.sample('x',400).vx,0);
});
test('large correction produces a labelled display reset, not a swept path',()=>{
 const s=new R.Engine();s.observe(event(1,0,[{x:10,y:0}]));
 s.observe(event(2,100,[{x:14,y:0}]));const row=s.view(310).visible[0];
 assert.equal(row.reset,true);assert.equal(row.generation,1);assert.ok(row.alpha<.1);
 assert.equal(row.entry.raw.location.x,14);assert.equal(row.pose.vx,0);
});
test('recorded legacy inventory reproduces source mixing, without altering the trace',()=>{
 const fs=require('fs'),path=require('path'),I=require('../../viewer_assets/presentation_inventory_v1.js');
 const lines=fs.readFileSync(path.join(__dirname,'../../recordings/two_ego_occl.jsonl'),'utf8').trim().split('\n');
 const snap=lines.map(l=>JSON.parse(l).snap).find(s=>s.frame_id===166812);
 const original=JSON.stringify(snap),view=I.projectInventory(snap,I.MODE_ASSOCIATED);
 assert.equal(view.objects.length,11);
 assert.equal(view.objects.filter(o=>o.object.source_stream_id==='fusion_ego_b').length,5);
 assert.equal(JSON.stringify(snap),original);
});
console.log(JSON.stringify({status:'PASS',focused_tests:checks,settings:R.SETTINGS},null,2));
