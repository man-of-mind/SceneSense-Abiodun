'use strict';
// v20.3 map-freshness strip: values come only from supplied stamps; anything missing is "not available", never 0.
const assert=require('assert/strict'),Q=require('../../viewer_assets/replay_qos_v1.js');
let checks=0;const test=(n,f)=>{try{f();checks++;}catch(e){console.error(n);throw e;}};
const frame=(capture,mapping='RECORDED_SAME_HOST_WALL')=>({capture_ms:capture,age_mapping:mapping});

test('derives capture->install, frame age, rate and time since last update from supplied stamps',()=>{
 const q=Q.compute({now:1000,latest:frame(700),installMs:880,receipts:[400,520,640,760,880],visible:[{ageMs:300},{ageMs:300},{ageMs:540}]});
 assert.equal(q.captureToInstallMs,180);assert.equal(q.displayedDataAgeMs,300);assert.equal(q.sinceLastUpdateMs,120);
 assert(Math.abs(q.updateRateHz-4/0.48)<1e-9);assert.equal(q.objectAgeP50Ms,300);assert.equal(q.objectAgeMaxMs,540);});
test('object observation age is reported separately from frame age (a coasting object is older than the frame)',()=>{
 const q=Q.compute({now:1000,latest:frame(800),installMs:950,receipts:[950],visible:[{ageMs:200},{ageMs:620}]});
 assert.equal(q.displayedDataAgeMs,200);assert.equal(q.objectAgeMaxMs,620);assert.match(Q.format(q).objectAge,/oldest 620 ms/);});
test('nothing received yet: every value is "not available", never zero',()=>{
 const f=Q.format(Q.compute({now:0,latest:null,receipts:[],visible:[]}));
 for(const k of ['captureToInstall','displayedDataAge','updateRate','sinceLastUpdate'])assert.equal(f[k],'not available',k);
 assert.equal(f.objectAge,'not available (no objects shown)');assert.doesNotMatch(Object.values(f).join(' '),/\b0 ms|\b0\.0 \/s/);});
test('missing install stamp (e.g. a live event without a timeline) leaves capture->install not available',()=>{
 const q=Q.compute({now:1000,latest:frame(700),installMs:null,receipts:[700,820],visible:[]});
 assert.equal(q.captureToInstallMs,null);assert.equal(Q.format(q).captureToInstall,'not available');assert.equal(q.displayedDataAgeMs,300);});
test('capture and install on different clocks: no age is computed from them',()=>{
 const q=Q.compute({now:1000,latest:frame(700,'UNMAPPED'),installMs:880,receipts:[880],visible:[{ageMs:100}]});
 assert.equal(q.captureToInstallMs,null);assert.equal(q.displayedDataAgeMs,null);assert.equal(q.objectAgeP50Ms,null);
 assert.equal(q.sinceLastUpdateMs,120,'receipt timing is on the display clock and still valid');});
test('one update in the 5-s window is not a rate',()=>{
 assert.equal(Q.compute({now:9000,latest:frame(8800),receipts:[1000,8900],visible:[]}).updateRateHz,null);});
test('prediction / UE-feedback timing is never shown: the feed carries no joined stamp',()=>{
 const q=Q.compute({now:1000,latest:{...frame(700),timeline:{install_ms:880},feedback_ms:950},installMs:880,receipts:[880],visible:[]});
 assert.equal(q.predictionFeedbackMs,null);assert.match(Q.format(q).predictionFeedback,/^not available/);});
console.log(`replay_qos_v1: ${checks} checks passed`);
