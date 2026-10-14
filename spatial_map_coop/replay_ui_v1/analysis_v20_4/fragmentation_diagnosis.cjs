// DIAGNOSIS ONLY (GT labels detections for scoring; never enters the engine).
// Why do new display tracks start? For every new track of the ACCEPTED engine, classify by the GT actor of its first
// detection:
//   SPLIT     another live track of the same class already follows that actor (gate failure or double detection)
//   REACQUIRE a track that followed that actor ended within the last 3 s (gap > expiry)
//   FIRST     first track for that actor in the clip;  NO_GT  no GT actor within 3 m (false positive or unscored)
// For SPLIT/REACQUIRE: offset of the new detection from the old track's predicted position, radial/lateral to the sensor.
// Usage: node fragmentation_diagnosis.cjs <out.json> <dir:clip> ...
'use strict';
const fs=require('fs'),path=require('path');
const T=require(process.env.ENGINE||path.join(__dirname,'../baseline_v20_2/replay_tracks_v3.js'));
const [outPath,...specs]=process.argv.slice(2);
const q=(a,p)=>{if(!a.length)return null;const s=a.slice().sort((x,y)=>x-y);return +s[Math.min(s.length-1,Math.floor(s.length*p))].toFixed(2);};
function gtIndex(frames){const per={};for(const f of Object.values(frames))for(const p of f.positives)(per[p.id]=per[p.id]||[]).push({t:f.capture_ms,x:p.x,y:p.y,label:p.label});
  for(const v of Object.values(per))v.sort((a,b)=>a.t-b.t);
  const at=(v,t)=>{let i=0;while(i<v.length&&v[i].t<t)i++;if(i<v.length&&Math.abs(v[i].t-t)<1)return v[i];if(i===0||i>=v.length)return null;const a=v[i-1],b=v[i];if(b.t-a.t>400)return null;const k=(t-a.t)/(b.t-a.t);return {x:a.x+(b.x-a.x)*k,y:a.y+(b.y-a.y)*k};};
  return (cls,x,y,t)=>{let best=null,d=3;for(const [id,v] of Object.entries(per)){if((v[0].label==='person')!==(cls==='Pedestrian'))continue;const g=at(v,t);if(!g)continue;const e=Math.hypot(g.x-x,g.y-y);if(e<d){d=e;best=id;}}return best;};}
const result={schema:'display_track_fragmentation_diagnosis.v1',clips:{}};
for(const spec of specs){const i=spec.lastIndexOf(':'),dir=spec.slice(0,i),name=spec.slice(i+1);
  const clip=JSON.parse(fs.readFileSync(path.join(dir,name+'.json'))),geo=JSON.parse(fs.readFileSync(path.join(dir,'geometry.json')));
  const gtOf=gtIndex(JSON.parse(fs.readFileSync(path.join(dir,'gt_audit.json'))).clips[name]||{});
  const e=new T.TrackEngine({roads:geo}),label=new Map(),ended=[],rows=[];
  for(const ev of clip.events){const before=new Map([...e.tracks].map(([k,t])=>[k,{t,pred:e.predictAt(t,ev.capture_ms),gt:label.get(k)}]));
    e.observe(ev);
    for(const [k,o] of before)if(!e.tracks.has(k)&&o.gt)ended.push({gt:o.gt,cls:o.t.cls,t:ev.capture_ms,pred:o.pred,est:o.t.state==='established'});
    for(const [k,t] of e.tracks){if(t.lastT!==ev.capture_ms)continue;const h=t.hist[t.hist.length-1],g=gtOf(t.cls,h.x,h.y,h.t);
      if(!before.has(k)){ // new track
        const r={cls:t.cls,t:ev.capture_ms,gt:g,kind:g?'FIRST':'NO_GT'};
        if(g){const live=[...before].find(([kk,o])=>o.gt===g&&o.t.cls===t.cls&&e.tracks.has(kk));
          const old=live?{pred:live[1].pred,updated:live[1].t.lastT===ev.capture_ms,est:live[1].t.state==='established'}:null;
          const gone=old?null:ended.filter(x=>x.gt===g&&x.cls===t.cls&&ev.capture_ms-x.t<=3000).pop();
          const ref=old||gone;if(old)r.kind='SPLIT';else if(gone)r.kind='REACQUIRE';
          if(ref){const sx=h.sx,sy=h.sy,rr=Math.hypot(h.x-sx,h.y-sy)||1,ux=(h.x-sx)/rr,uy=(h.y-sy)/rr,dx=h.x-ref.pred.x,dy=h.y-ref.pred.y;
            Object.assign(r,{dist:Math.hypot(dx,dy),radial:Math.abs(dx*ux+dy*uy),lateral:Math.abs(-dx*uy+dy*ux),range:rr,oldEstablished:ref.est,
              oldUpdatedSameFrame:old?old.updated:null,gapMs:gone?ev.capture_ms-gone.t:null});}}
        rows.push(r);}
      if(g)label.set(k,g);}}
  const summ={};for(const cls of ['Pedestrian','Vehicle']){const R=rows.filter(r=>r.cls===cls),by=k=>R.filter(r=>r.kind===k);
    const sp=by('SPLIT'),re=by('REACQUIRE'),notUpd=sp.filter(r=>!r.oldUpdatedSameFrame);
    summ[cls]={new_tracks:R.length,FIRST:by('FIRST').length,NO_GT:by('NO_GT').length,SPLIT:sp.length,REACQUIRE:re.length,
      split_old_not_updated_same_frame:notUpd.length,split_old_established:sp.filter(r=>r.oldEstablished).length,
      split_offset:{dist_p50:q(sp.map(r=>r.dist),.5),radial_p50:q(sp.map(r=>r.radial),.5),lateral_p50:q(sp.map(r=>r.lateral),.5),lateral_p90:q(sp.map(r=>r.lateral),.9),
        radial_dominant:sp.filter(r=>r.radial>r.lateral).length,within_lateral_0_6:sp.filter(r=>r.lateral<=.6).length},
      reacquire:{gap_ms_p50:q(re.map(r=>r.gapMs),.5),gap_ms_p90:q(re.map(r=>r.gapMs),.9),dist_p50:q(re.map(r=>r.dist),.5),lateral_p50:q(re.map(r=>r.lateral),.5),old_established:re.filter(r=>r.oldEstablished).length}};}
  result.clips[name+'@'+path.basename(dir)]={summary:summ,rows};console.log(name.padEnd(13),JSON.stringify(summ.Pedestrian),'\n   veh',JSON.stringify({n:summ.Vehicle.new_tracks,FIRST:summ.Vehicle.FIRST,NO_GT:summ.Vehicle.NO_GT,SPLIT:summ.Vehicle.SPLIT,REACQ:summ.Vehicle.REACQUIRE,off:summ.Vehicle.split_offset}));}
if(outPath&&outPath!=='-')fs.writeFileSync(outPath,JSON.stringify(result,null,1)+'\n');
