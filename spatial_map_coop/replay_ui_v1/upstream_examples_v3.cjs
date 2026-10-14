// Export the v3 display state at the arrival instant of selected frames (for upstream_examples_v3.py).
'use strict';const fs=require('fs'),path=require('path'),V3=require('../viewer_assets/replay_tracks_v3.js');
const [clipsDir,outPath]=process.argv.slice(2),want={departure:[1480],turn:[2020],stop:[2748]},out={};
for(const [name,frames] of Object.entries(want)){const clip=JSON.parse(fs.readFileSync(path.join(clipsDir,name+'.json'))),e=new V3.TrackEngine();
  for(const ev of clip.events){e.observe(ev);if(frames.includes(ev.frame_id)){const v=e.view(ev.arrival_ms+1,{radius:45});
    out[ev.frame_id]={ego:v.ego,tracks:v.visible.map(r=>({x:r.pose.x,y:r.pose.y,yaw:r.yaw,cls:r.cls,tier:r.tier,missed:r.missed,conflict:r.conflict,l:r.dims.l,w:r.dims.w})),
      field:v.field.filter(c=>c.shown).map(c=>({x:c.x,y:c.y,w:c.w})),raw:ev.objects.map(o=>({x:o.raw.location.x,y:o.raw.location.y,score:o.raw.score,cls:o.raw.type})),pose:ev.pose};}}}
fs.writeFileSync(outPath,JSON.stringify(out));console.log(Object.keys(out));
