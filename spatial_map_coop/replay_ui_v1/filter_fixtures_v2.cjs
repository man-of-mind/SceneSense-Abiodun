/* Synthetic truth distinguishes real movement from measurement jitter. */
const fs=require('fs'),path=require('path'),P=require('../viewer_assets/presentation_predictor_v1.js'),M=require('../viewer_assets/replay_motion_v2.js');
const rms=a=>Math.sqrt(a.reduce((s,x)=>s+x*x,0)/(a.length||1));
const mean=a=>a.reduce((s,x)=>s+x,0)/(a.length||1);
const cases={stationary:t=>[20,3,0],constant_motion:t=>[20+4*t,3,0],turn:t=>[20+10*Math.sin(t/2),3+10*(1-Math.cos(t/2)),t/2*180/Math.PI],stop:t=>[20+4*Math.min(t,4),3,0]};
const report={schema:'synthetic_filter_fixture.v2',input_cadence_ms:100,capture_to_arrival_ms:200,noise:'deterministic 0.18 m sinusoids; synthetic, not M3 truth',added_input_buffer_ms:0,cases:{}};
for(const [name,truth]of Object.entries(cases)){const fns={before:new P.DisplayPredictor({horizonMs:500,correctionMs:120,stationarySpeedMps:.25,maxCorrectionM:3,velocityConsistencyMps:6,teleportDistanceM:5}),after:new M.Filter()};
 const result={};for(const [version,f]of Object.entries(fns)){let next=0;const errors=[],samples=[],measurementErrors=[];let stopSettles=null;
  for(let now=0;now<=8000;now+=1000/60){while(next+200<=now){const t=next/1000,p=truth(t),x=p[0]+.18*Math.sin(next*.037),y=p[1]+.18*Math.cos(next*.051);f.observe('x',{x,y,yaw:p[2],measuredMs:next,sampleId:next,className:'Vehicle',ageAtArrivalMs:200},next+200);measurementErrors.push(Math.hypot(x-p[0],y-p[1]));next+=100;}
   const s=f.sample('x',now),target=truth(now/1000);if(s&&now>=2000){errors.push(Math.hypot(s.x-target[0],s.y-target[1]));samples.push({now,x:s.x,y:s.y,tx:target[0],ty:target[1]});if(name==='stop'&&now>=4000&&Math.hypot(s.x-target[0],s.y-target[1])<.15&&Math.hypot(s.vx,s.vy)<.5&&stopSettles===null)stopSettles=now-4000;}}
  const motion=samples.slice(1).map((s,i)=>Math.hypot(s.x-samples[i].x,s.y-samples[i].y));
  result[version]={display_error_rmse_m:rms(errors),raw_measurement_error_rmse_m:rms(measurementErrors),mean_frame_motion_m:mean(motion),first_stop_settle_after_truth_ms:stopSettles,
   constant_motion_signed_lag_ms:name==='constant_motion'?mean(samples.map(s=>(s.tx-s.x)/4*1000)):null};
 }report.cases[name]=result;
}
fs.writeFileSync(path.join(__dirname,'artifacts_v2/FILTER_FIXTURES.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
