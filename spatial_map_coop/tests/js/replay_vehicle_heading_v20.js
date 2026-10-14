'use strict';
const assert=require('assert/strict'),T=require('../../viewer_assets/replay_tracks_v3.js'),G=require('../../viewer_assets/replay_geometry_v2.js');
let checks=0;const test=(n,f)=>{try{f();checks++;}catch(e){console.error(n);throw e;}};
const axis=(a,b)=>Math.abs(T.axisDelta(a,b));
function fixture(){const e=new T.TrackEngine(),t={cls:'Vehicle',hist:[],heading:null,motion:'stationary',speed:0,tstat:0,est:{vx:0,vy:0}};
 let time=0;return {e,t,step(yaw,{strong=true,moving=false,travel=0,speed=5}={}){time+=140;t.hist.push({t:time,x:20,y:0,yaw,score:strong?.7:.1,strong});if(t.hist.length>20)t.hist.shift();
  Object.assign(t,{motion:moving?'moving':'stationary',speed:moving?speed:0,tstat:moving?10:0,est:{vx:speed*Math.cos(travel*Math.PI/180),vy:speed*Math.sin(travel*Math.PI/180)}});
  e.updateHeading(t,time);return e.displayYaw(t,time);},get time(){return time;}};}
test('stationary jitter and contradictory above-threshold outliers cannot accumulate votes',()=>{
 const f=fixture();for(const y of [0,0,0,2,-2,1,45,-45,45,-45,45,-45])f.step(y);assert(axis(f.t.heading.yaw,0)<1e-9);assert(f.t.heading.pending<=1);});
test('a consistent changed axis is accepted, so orientation is not frozen',()=>{
 const f=fixture();for(const y of [0,0,0,40,41,39,40])f.step(y);assert(axis(f.t.heading.yaw,40)<1);assert(axis(f.e.displayYaw(f.t,f.time+500),40)<1);});
test('weak and missing headings break pending support; retain last axis without direction certainty',()=>{
 const f=fixture();[0,0,0,40,40].forEach(y=>f.step(y));f.step(null);f.step(40);assert.equal(f.t.heading.pending,1);
 f.step(40,{strong:false});assert.equal(f.t.heading.pending,0);assert.equal(f.t.heading.known,false);assert.equal(f.t.heading.yaw,0);});
test('missing heading at birth stays unknown, replacement inherits nothing',()=>{
 const f=fixture();f.step(null);assert.equal(f.t.heading.yaw,null);assert.equal(f.e.displayYaw(f.t,f.time),null);
 const g=fixture();g.step(75);f.step(null);assert.equal(f.t.heading.yaw,null);});
test('alternating 180-degree producer ambiguity never spins the body or claims a front',()=>{
 const f=fixture();const y=[];for(let i=0;i<20;i++)y.push(f.step(i%2?180:0));assert(y.every(a=>axis(a,0)<1e-9));assert.equal(f.t.heading.known,false);});
test('sustained front/back evidence swaps the marker only',()=>{
 const f=fixture();[0,0,0,...Array(8).fill(180)].forEach(y=>f.step(y));assert.equal(f.t.heading.yaw,0);assert.equal(f.t.heading.directionYaw,-180);});
test('smooth moving turn remains responsive',()=>{
 const f=fixture();for(let i=0;i<12;i++)f.step(0,{moving:true});
 for(let i=1;i<=15;i++){const angle=i*4;f.step(angle,{moving:true,travel:angle});assert(axis(f.t.heading.yaw,angle)<1e-9);assert(axis(f.e.displayYaw(f.t,f.time+140),angle)<1e-9);}});
test('speed oscillation across 2.5 m/s does not switch back to producer axis',()=>{
 const f=fixture();for(let i=0;i<8;i++)f.step(90,{moving:true,travel:0});
 for(const speed of [2.4,2.6,2.1,2.7,1.8]){f.step(90,{moving:true,travel:0,speed});assert.equal(f.t.heading.motionMode,true);assert(axis(f.t.heading.yaw,0)<1e-9);}});
test('reverse motion is separate from the producer front and does not rotate body',()=>{
 const f=fixture();for(let i=0;i<10;i++)f.step(0,{moving:true,travel:0});
 for(let i=0;i<10;i++){f.step(0,{moving:true,travel:180});assert(axis(f.t.heading.yaw,0)<1e-9);assert.equal(f.t.heading.directionYaw,0);}
 assert.equal(f.t.heading.travelYaw,180);});
test('noisy stationary-centroid estimate cannot manufacture travel/front confidence',()=>{
 const f=fixture();for(let i=0;i<12;i++)f.step(null,{moving:false,travel:i*40,speed:3});assert.equal(f.t.heading.travelYaw,null);assert.equal(f.t.heading.known,false);});
test('heading-sensitive matching is invariant to vehicle front/back',()=>{
 const e=new T.TrackEngine(),t={cls:'Vehicle',motion:'stationary',anchor:{x:20,y:0},hist:[{x:20,y:0}],dims:{l:4.8,w:1.9},heading:{yaw:0,known:true}};
 const c={cls:'Vehicle',x:20,y:0,dims:{l:4.8,w:1.9},yaw:0};const a=e.cost(t,c,0,2.5);c.yaw=180;assert.equal(e.cost(t,c,0,2.5),a);
 assert.equal(T.insideFootprint({x:20,y:2.4},t.anchor,t.dims,0),false);assert.equal(T.insideFootprint({x:25.6,y:0},t.anchor,t.dims,0),false);});
test('building edge, intersecting footprint and contained centre remain uncertain',()=>{
 const g=new G.Buildings({validation:{valid:true},buildings:[[[10,-5],[20,-5],[20,5],[10,5]]]});
 for(const x of [9,10,12])assert.equal(g.classify({x:0,y:0},{x,y:0},{length:4,width:2},0).state,'VISIBILITY_UNCERTAIN');
 assert.equal(g.classify({x:0,y:0},{x:30,y:0},{length:4,width:2},0).state,'STATIC_MAP_OCCLUDED');});
function event(frame,ms,objects){const ns=String(1000000000+ms*1e6);return {session:'r/c',source:'s',frame_id:frame,capture_ms:ms,arrival_ms:ms+190,capture_ns:ns,clock_domain:'unix_wall_ns',age_mapping:'RECORDED_SAME_HOST_WALL',
 pose:{x:0,y:0,yaw:0,source:'s',frame_id:frame,capture_ns:ns,binding:['r','c','s',frame,ns],sensor:{x:1.8,y:0,yaw:0},fov_deg:120,range_m:120,
 camera_matrix:[[1,0,0,1.8],[0,1,0,0],[0,0,1,1.55],[0,0,0,1]]},
 objects:objects.map((o,i)=>({source:'s',frame_id:frame,capture_ns:ns,heading_valid:o.valid!==false,heading_yaw_deg:o.yaw??0,
 raw:{id:`s:${frame}:${i}`,type:o.type||'Vehicle',score:o.score??.7,location:{x:o.x??20,y:o.y||0,z:1},dimensions:{length:4.8,width:1.9,height:1.5}}}))};}
test('adjacent parked neighbours retain two identities through contradictory yaw updates',()=>{
 const e=new T.TrackEngine();for(let i=0;i<30;i++){
  e.observe(event(i+1,i*140,[{yaw:i<4?0:i%2?45:-45},{y:2.4,yaw:i%2?180:0}]));
  if(i>0){assert.equal(e.stats.established,2);assert.equal(e.tracks.size,2);assert([...e.tracks.values()].every(t=>!t.conflict));}}
 assert.equal(e.stats.tentativeCreated,2);assert.equal(e.stats.absorbed,0);});
test('unqualified entries cannot use legacy yaw; expired replacement has no inherited orientation',()=>{
 const e=new T.TrackEngine();e.observe(event(1,0,[{yaw:75}]));e.observe(event(2,140,[{yaw:75}]));
 e.observe(event(3,280,[{yaw:90,valid:false}]));assert.equal([...e.tracks.values()][0].heading.yaw,75);assert.equal([...e.tracks.values()][0].heading.known,false);
 e.observe(event(4,1400,[{yaw:90,valid:false}]));const r=e.view(1590).visible[0];assert.equal(r.yaw,null);assert.equal(r.tier,'tentative');});
test('unresolved pedestrian presence is independent of the low-confidence field',()=>{
 const e=new T.TrackEngine();for(let i=0;i<4;i++)e.observe(event(i+1,i*140,[{type:'Pedestrian',x:15+i*.02,valid:false},{x:30,y:8,score:.05}]));
 const v=e.view(610);assert(v.field.length>0);assert(v.visible.some(r=>r.cls==='Pedestrian'&&r.motion==='unresolved'&&r.tier==='established'));});
console.log(JSON.stringify({status:'PASS',vehicle_heading_checks:checks}));
