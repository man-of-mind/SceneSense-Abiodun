// Full-route, 60-Hz equivalence: v20.1 (frozen) vs current engine. No GT.
const fs=require('fs'),readline=require('readline');
const V='/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/spatial_map_coop/';
const A=require(V+'replay_ui_v1/baseline_v20_1/replay_tracks_v3.js'),B=require(V+'viewer_assets/replay_tracks_v3.js');
const dir=process.argv[2],geo=JSON.parse(fs.readFileSync(dir+'/geometry.json'));
const a=new A.TrackEngine(),b=new B.TrackEngine({roads:geo});
const n={samples:0,rows:0,diff:{},vehicleRows:0,pedRows:0,displayDiffers:0,neutral:0};
const fields=['key','id','cls','tier','missed','motion','speed','yaw','headingKnown','directionYaw','conflict','shown','reason','alpha'];
(async()=>{let pending=[],last=null;
 const sample=t=>{const va=a.view(t),vb=b.view(t);n.samples++;
  if(va.tracks.length!==vb.tracks.length){n.diff.trackCount=(n.diff.trackCount||0)+1;return;}
  va.tracks.forEach((ra,i)=>{const rb=vb.tracks[i];n.rows++;ra.cls==='Pedestrian'?n.pedRows++:n.vehicleRows++;
   for(const f of fields)if(JSON.stringify(ra[f])!==JSON.stringify(rb[f]))n.diff[f]=(n.diff[f]||0)+1;
   if(ra.pose.x!==rb.pose.x||ra.pose.y!==rb.pose.y)n.diff.pose=(n.diff.pose||0)+1;
   if(JSON.stringify(ra.motionUncertainty)!==JSON.stringify(rb.motionUncertainty))n.diff.motionUncertainty=(n.diff.motionUncertainty||0)+1;
   if(rb.cls==='Pedestrian'&&(rb.displayYaw!==rb.yaw||rb.orientation!==null))n.diff.pedestrianDisplay=(n.diff.pedestrianDisplay||0)+1;
   if(rb.cls==='Vehicle'){if(rb.displayYaw===null)n.neutral++;if(JSON.stringify(rb.displayYaw)!==JSON.stringify(ra.yaw))n.displayDiffers++;}});
  if(JSON.stringify(va.counts)!==JSON.stringify(vb.counts))n.diff.counts=(n.diff.counts||0)+1;
  if(va.field.length!==vb.field.length)n.diff.field=(n.diff.field||0)+1;};
 for await(const line of readline.createInterface({input:fs.createReadStream(dir+'/events.jsonl')})){
  const e=JSON.parse(line);
  if(last!==null)for(let t=last+1000/60;t<e.arrival_ms;t+=1000/60)sample(t);
  a.observe(e);b.observe(e);last=e.arrival_ms;sample(e.arrival_ms);}
 console.log(JSON.stringify(n));fs.writeFileSync(process.argv[3],JSON.stringify(n,null,1));})();
