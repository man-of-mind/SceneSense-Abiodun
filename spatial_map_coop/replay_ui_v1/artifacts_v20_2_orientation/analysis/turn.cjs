// Turn response: drawn axis, v20.1 vs v20.2, for moving established vehicles (frames 4770-5200). No GT.
const fs=require('fs'),readline=require('readline');
const V='/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/spatial_map_coop/';
const A=require(V+'replay_ui_v1/baseline_v20_1/replay_tracks_v3.js'),B=require(V+'viewer_assets/replay_tracks_v3.js');
const dir=process.argv[2],geo=JSON.parse(fs.readFileSync(dir+'/geometry.json'));
const a=new A.TrackEngine(),b=new B.TrackEngine({roads:geo}),ax=(x,y)=>Math.abs(B.axisDelta(x,y));
const series=new Map();let last=null;
const sample=t=>{const va=a.view(t),vb=b.view(t);va.tracks.forEach((ra,i)=>{const rb=vb.tracks[i];if(ra.cls!=='Vehicle'||!ra.shown||ra.tier!=='established')return;
  if(!series.has(ra.id))series.set(ra.id,[]);series.get(ra.id).push({t,motion:ra.motion,old:ra.yaw,neu:rb.displayYaw,fit:ra.motion==='moving'?Math.atan2(ra.track.est.vy,ra.track.est.vx)*180/Math.PI:null});});};
(async()=>{for await(const line of readline.createInterface({input:fs.createReadStream(dir+'/events.jsonl')})){const e=JSON.parse(line);
  if(e.frame_id>5200)break;if(last!==null&&e.frame_id>=4770)for(let t=last+1000/60;t<e.arrival_ms;t+=1000/60)sample(t);a.observe(e);b.observe(e);last=e.arrival_ms;}
 const out=[];for(const [id,s] of series){const mv=s.filter(r=>r.motion==='moving');if(mv.length<120)continue;
  const span=(()=>{const f=mv.map(r=>r.fit);let m=0;for(const x of f)for(const y of f)m=Math.max(m,ax(x,y));return m;})();
  const d=mv.filter(r=>r.neu!==null&&r.old!==null).map(r=>ax(r.old,r.neu)).sort((x,y)=>x-y);
  const lagOld=mv.filter(r=>r.old!==null).map(r=>ax(r.old,r.fit)).sort((x,y)=>x-y),lagNew=mv.filter(r=>r.neu!==null).map(r=>ax(r.neu,r.fit)).sort((x,y)=>x-y);
  out.push({id,movingSamples:mv.length,fitAxisRangeDeg:+span.toFixed(1),neutralWhileMoving:mv.filter(r=>r.neu===null).length,
   drawnVsV201_p50:+d[d.length>>1].toFixed(2),drawnVsV201_p95:+d[Math.floor(d.length*.95)].toFixed(2),
   offFit_v201_p95:+lagOld[Math.floor(lagOld.length*.95)].toFixed(2),offFit_v202_p95:+lagNew[Math.floor(lagNew.length*.95)].toFixed(2)});}
 out.sort((x,y)=>y.fitAxisRangeDeg-x.fitAxisRangeDeg);console.log(JSON.stringify(out.slice(0,6),null,1));fs.writeFileSync(process.argv[3],JSON.stringify(out,null,1));})();
