// DIAGNOSIS ONLY: raw producer yaw (strong, qualified vehicles) vs GT; no display engine.
const fs=require('fs'),readline=require('readline');
const [dir,gtPath]=process.argv.slice(2),gt=JSON.parse(fs.readFileSync(gtPath));
const ax=(a,b)=>{let d=((b-a)%360+540)%360-180;return d>90?d-180:d<-90?d+180:d;};
const out=[];
(async()=>{for await(const line of readline.createInterface({input:fs.createReadStream(dir+'/events.jsonl')})){
 const e=JSON.parse(line),G=gt.frames[e.frame_id];if(!G)continue;const s=e.pose.sensor;
 for(const o of e.objects){const r=o.raw;if(r.type==='Pedestrian'||!(r.score>=.2))continue;
  let best=null;for(const b of G)if(b.label==='vehicle'){const d=Math.hypot(b.x-r.location.x,b.y-r.location.y);if(d<2.5&&(!best||d<best.d))best={...b,d};}
  if(!best)continue;const ray=Math.atan2(best.y-s.y,best.x-s.x)*180/Math.PI;
  out.push({f:e.frame_id,score:r.score,valid:o.heading_valid,yaw:o.heading_yaw_deg,gt:best.yaw,id:best.id,src:best.src,speed:best.speed,range:Math.hypot(best.x-s.x,best.y-s.y),
   rel:ax(ray,best.gt===undefined?best.yaw:best.yaw),err:o.heading_valid?ax(best.yaw,o.heading_yaw_deg):null,rayErr:o.heading_valid?ax(ray,o.heading_yaw_deg):null,
   reason:o.heading_reason});}
}fs.writeFileSync(process.argv[4],JSON.stringify(out));console.log(out.length);})();
