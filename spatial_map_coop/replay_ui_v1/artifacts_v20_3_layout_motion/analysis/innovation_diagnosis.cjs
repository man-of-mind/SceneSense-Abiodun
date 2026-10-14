// DIAGNOSIS: per-update display corrections (innovations) of the CURRENT engine, by class x motion.
// Are they white noise (an EMA would help) or persistent drift (an EMA would only add lag)?
'use strict';
const fs=require('fs'),path=require('path');
const A=process.env.ENGINE||require('path').join(__dirname,'../../baseline_v20_2/replay_tracks_v3.js');
const T=require(A);
const q=(a,p)=>{if(!a.length)return null;const s=a.slice().sort((x,y)=>x-y);return +s[Math.min(s.length-1,Math.floor(s.length*p))].toFixed(3);};
const out={};
for(const spec of process.argv.slice(2)){const [dir,name]=spec.split(':');const clip=JSON.parse(fs.readFileSync(path.join(dir,name+'.json')));
  const geo=JSON.parse(fs.readFileSync(path.join(dir,'geometry.json')));
  const e=new T.TrackEngine({roads:geo});const per=new Map();const groups={};
  for(const ev of clip.events){e.observe(ev);
    for(const t of e.tracks.values()){if(t.lastT!==ev.capture_ms||!t.bridge||t.bridge.t0!==ev.arrival_ms||t.state!=='established')continue;
      const last=t.hist[t.hist.length-1],rx=last.x-last.sx,ry=last.y-last.sy,rn=Math.hypot(rx,ry)||1,ux=rx/rn,uy=ry/rn;
      const d={x:t.bridge.x,y:t.bridge.y,r:t.bridge.x*ux+t.bridge.y*uy,g:-t.bridge.x*uy+t.bridge.y*ux,motion:t.motion,cls:t.cls,t:ev.capture_ms};
      const k=t.cls+'/'+t.motion;(groups[k]=groups[k]||{m:[],r:[],g:[],corr:[],corrR:[]});const G=groups[k];
      G.m.push(Math.hypot(d.x,d.y));G.r.push(Math.abs(d.r));G.g.push(Math.abs(d.g));
      const p=per.get(t.id);if(p&&p.motion===d.motion&&d.t-p.t<=300){G.corr.push([p.x*d.x+p.y*d.y,p.x*p.x+p.y*p.y,d.x*d.x+d.y*d.y]);G.corrR.push([p.r,d.r]);}
      per.set(t.id,d);}}
  const res={};for(const [k,G] of Object.entries(groups)){const s=G.corr.reduce((a,c)=>[a[0]+c[0],a[1]+c[1],a[2]+c[2]],[0,0,0]);
    const mr=G.corrR.length?G.corrR.reduce((a,c)=>a+c[0],0)/G.corrR.length:0,mr2=G.corrR.length?G.corrR.reduce((a,c)=>a+c[1],0)/G.corrR.length:0;
    let sxy=0,sxx=0,syy=0;for(const [a,b] of G.corrR){sxy+=(a-mr)*(b-mr2);sxx+=(a-mr)**2;syy+=(b-mr2)**2;}
    res[k]={n:G.m.length,innov_m_p50:q(G.m,.5),p95:q(G.m,.95),radial_p50:q(G.r,.5),tangential_p50:q(G.g,.5),
      lag1_vector_corr:G.corr.length?+(s[0]/Math.sqrt(s[1]*s[2])).toFixed(2):null,lag1_radial_corr:G.corrR.length>3?+(sxy/Math.sqrt(sxx*syy)).toFixed(2):null};}
  out[name]=res;console.log(name,JSON.stringify(res));}
