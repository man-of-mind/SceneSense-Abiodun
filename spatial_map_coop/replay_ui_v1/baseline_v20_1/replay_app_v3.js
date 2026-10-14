/* Ego traffic view v3: tiered display tracks over recorded installed-map events.
 * Read-only replay. Raw observations are never edited; every candidate keeps an
 * explicit display disposition visible in the Inspect drawer. */
(function(){
'use strict';
const T=window.ScenesenseReplayTracks,I=window.ScenesenseMapInventory,G=window.ScenesenseBuildings;
const $=id=>document.getElementById(id),canvas=$('map'),ctx=canvas.getContext('2d');
const pedPolicy=new URLSearchParams(location.search).get('ped');
const engine=new T.TrackEngine(pedPolicy?{policy:{pedestrianMotion:pedPolicy}}:{}),intervals=new I.RollingBuffer(600),costs=new I.RollingBuffer(600);
const C={bg:'#0e161e',veh:'#dce6ec',vehDark:'#4f5f6c',edge:'#081017',ped:'#ffb07c',tent:'#aebbc5',miss:'#8796a3',hyp:'#e7b45c',ego:'#63e0bd',los:'#7cc8ef',raw:'#f0c27a'};
let data=null,meta=null,statics=null,buildings=null,backdrop=null,extent=null,hypSprite=null;
let now=0,cursor=0,playing=false,speed=1,heading=true,radius=45,mode='current',inspect=false;
let lastWall=null,lastHUD=-Infinity,lastView=null,loadGeneration=0,lastCamera=null,frameCount=0,selected=null,drawPose=null,loading=false;
let width=1,height=1,dpr=1;
const trails=new Map();
const time=t=>{t=Math.max(0,t)/1000;return String(Math.floor(t/60)).padStart(2,'0')+':'+(t%60).toFixed(1).padStart(4,'0');};
const text=(id,t)=>{$(id).textContent=t;};
const esc=t=>String(t).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function json(url){const r=await fetch(url);if(!r.ok)throw Error(`${url}: HTTP ${r.status}`);return r.json();}
function setPlaying(v){playing=v;text('play',v?'❚❚':'▶');$('play').setAttribute('aria-label',v?'Pause replay':'Play replay');}

// ---- time / data ---------------------------------------------------------
function seek(t){
  if(!data)return;
  now=Math.max(0,Math.min(data.end_ms,t));engine.reset($('source').value||data.sources[0]);cursor=0;drawPose=null;trails.clear();
  // Causal rebuild from bounded history only; nothing after `now` is read.
  while(cursor<data.events.length&&data.events[cursor].arrival_ms<now-T.POLICY.replayHistoryMs)cursor++;
  while(cursor<data.events.length&&data.events[cursor].arrival_ms<=now)engine.observe(data.events[cursor++]);
  lastHUD=-Infinity;lastCamera=null;
}
async function loadClip(name){
  const generation=++loadGeneration;setPlaying(false);loading=true;text('mapNotice','Loading recorded events…');
  try{
    const clip=await json('/api/replay/clip/'+encodeURIComponent(name));
    if(generation!==loadGeneration)return;
    if(clip.events.length>T.POLICY.maxEvents)throw Error('Replay event bound exceeded');
    data=clip;$('source').replaceChildren(...clip.sources.map((s,i)=>new Option('Ego '+(i+1)+' · '+s.replace(/^ue288_/,'').slice(-34),s)));
    $('seek').max=clip.end_ms;text('duration',time(clip.end_ms));seek(0);text('mapNotice','');
    text('footerStatus',`${clip.events.length} recorded events · ${clip.camera_frames.length} retained camera frames · pedestrian motion ${engine.policy.pedestrianMotion}`);
  }catch(e){text('mapNotice','Replay unavailable: '+e.message);}
  finally{if(generation===loadGeneration)loading=false;}
}

// ---- base map ------------------------------------------------------------
function resize(){const r=$('stage').getBoundingClientRect();width=r.width;height=r.height;dpr=Math.min(2,window.devicePixelRatio||1);
  canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);}
new ResizeObserver(resize).observe($('stage'));
function prepareMap(){
  buildings=new G.Buildings(statics);
  const points=[...statics.roads.flat(),...statics.buildings.flat()];
  hypSprite=document.createElement('canvas');hypSprite.width=hypSprite.height=64;
  const h=hypSprite.getContext('2d'),grad=h.createRadialGradient(32,32,0,32,32,32);
  grad.addColorStop(0,'rgba(231,180,92,0.95)');grad.addColorStop(.45,'rgba(231,180,92,0.35)');grad.addColorStop(1,'rgba(231,180,92,0)');h.fillStyle=grad;h.fillRect(0,0,64,64);
  if(!points.length){extent={x:-150,y:-150,size:300};backdrop=null;return;}
  const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]);
  extent={x:Math.min(...xs)-8,y:Math.min(...ys)-8};extent.size=Math.max(Math.max(...xs)-extent.x,Math.max(...ys)-extent.y)+8;
  backdrop=document.createElement('canvas');backdrop.width=backdrop.height=2048;
  const c=backdrop.getContext('2d'),s=2048/extent.size;
  c.fillStyle=C.bg;c.fillRect(0,0,2048,2048);c.setTransform(s,0,0,s,-extent.x*s,-extent.y*s);
  const path=pts=>{c.beginPath();pts.forEach((p,i)=>i?c.lineTo(p[0],p[1]):c.moveTo(p[0],p[1]));};
  // Quiet context: roads lighter than ground, buildings darker blocks; no lane
  // markings or outlines competing with traffic.
  c.lineCap='round';c.lineJoin='round';
  for(const [color,lw] of [['#16202a',7.6],['#1b2731',6.8]]){c.strokeStyle=color;c.lineWidth=lw;for(const r of statics.roads){path(r);c.stroke();}}
  c.fillStyle='#0a1016';c.strokeStyle='#18232d';c.lineWidth=.35;for(const b of statics.buildings){path(b);c.closePath();c.fill();c.stroke();}
}
function projection(ego){
  // Heading-up: ego low in the frame because only forward observations exist.
  const cx=width*.5,cy=heading?height*.8:height*.5,scale=heading?Math.min((cy-30)/radius,(width*.5-10)/radius):Math.min(width,height)/(2*radius);
  const a=heading?ego.yaw*Math.PI/180:0,ex=heading?-Math.sin(a):1,ey=heading?Math.cos(a):0,fx=heading?-Math.cos(a):0,fy=heading?-Math.sin(a):1;
  return {scale,cx,cy,project:(x,y)=>[cx+((x-ego.x)*ex+(y-ego.y)*ey)*scale,cy+((x-ego.x)*fx+(y-ego.y)*fy)*scale],
    matrix:[ex*scale,fx*scale,ey*scale,fy*scale,cx-(ego.x*ex+ego.y*ey)*scale,cy-(ego.x*fx+ego.y*fy)*scale]};
}
function screenAngle(project,x,y,yaw){const q=project(x,y),n=project(x+Math.cos(yaw*Math.PI/180),y+Math.sin(yaw*Math.PI/180));return Math.atan2(n[1]-q[1],n[0]-q[0]);}

// ---- glyphs --------------------------------------------------------------
function vehicle(q,angle,L,W,s){
  ctx.save();ctx.translate(q[0],q[1]);ctx.globalAlpha=s.alpha;
  if(angle===null){const d=Math.max(10,(L+W)/2*.85);ctx.beginPath();ctx.roundRect(-d/2,-d/2,d,d,d*.35);}
  else{ctx.rotate(angle);ctx.beginPath();ctx.roundRect(-L/2,-W/2,L,W,Math.min(W*.32,5));}
  if(s.fill){ctx.fillStyle=C.veh;ctx.fill();ctx.strokeStyle=C.edge;ctx.lineWidth=1;ctx.stroke();}
  else{ctx.setLineDash(s.dash||[]);ctx.strokeStyle=s.stroke;ctx.lineWidth=s.width||1.4;ctx.stroke();ctx.setLineDash([]);}
  if(angle!==null&&s.fill){ // windshield marks the travel direction only when it is known
    if(s.nose){ctx.fillStyle=C.vehDark;ctx.beginPath();ctx.roundRect(s.noseSign<0?-L*.26:L*.06,-W*.36,L*.2,W*.72,2);ctx.fill();}
    else{ctx.fillStyle='#b9c6cf';ctx.fillRect(-L*.18,-W*.3,L*.36,W*.6);}
  }
  ctx.restore();
}
function person(q,r,angle,s,halo){
  ctx.save();ctx.globalAlpha=s.alpha;
  if(halo){ // motion unresolved: measured 1-sigma detector error, long along the camera ray
    ctx.save();ctx.translate(q[0],q[1]);ctx.rotate(halo.angle);ctx.beginPath();ctx.ellipse(0,0,Math.max(r+3,halo.radial),Math.max(r+2,halo.lateral),0,0,Math.PI*2);
    ctx.fillStyle=C.ped+'20';ctx.fill();ctx.setLineDash([2,3]);ctx.strokeStyle=C.ped+'a8';ctx.lineWidth=1;ctx.stroke();ctx.restore();}ctx.beginPath();ctx.arc(q[0],q[1],r,0,Math.PI*2);
  if(s.fill){ctx.fillStyle=C.ped;ctx.fill();ctx.strokeStyle=C.edge;ctx.lineWidth=1;ctx.stroke();}
  else{ctx.setLineDash(s.dash||[]);ctx.strokeStyle=s.stroke;ctx.lineWidth=1.4;ctx.stroke();ctx.setLineDash([]);}
  if(angle!==null&&s.fill){ctx.strokeStyle=C.ped;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(q[0]+Math.cos(angle)*r,q[1]+Math.sin(angle)*r);ctx.lineTo(q[0]+Math.cos(angle)*(r+5),q[1]+Math.sin(angle)*(r+5));ctx.stroke();}
  ctx.restore();
}
function styleFor(row){
  if(row.conflict)return {stroke:C.hyp,dash:[4,3],alpha:.9*row.alpha,width:1.5};
  if(row.tier==='tentative')return {stroke:C.tent,dash:[3,3],alpha:.6*row.alpha};
  if(row.missed)return {stroke:C.miss,dash:[4,3],alpha:row.alpha};
  return {fill:true,alpha:row.alpha,nose:row.headingKnown,noseSign:Number.isFinite(row.directionYaw)&&Math.cos((row.directionYaw-row.yaw)*Math.PI/180)<0?-1:1};
}

// ---- frame drawing -------------------------------------------------------
function drawField(view,project,scale){
  if(!view.field.length)return;
  const cell=T.POLICY.fieldCellM,size=Math.max(14,cell*2.8*scale),shown=view.field.filter(c=>c.shown);
  for(const c of shown){const q=project(c.x,c.y);ctx.globalAlpha=Math.min(.22,.035+.06*c.w);ctx.drawImage(hypSprite,q[0]-size/2,q[1]-size/2,size,size);}
  ctx.globalAlpha=1;
  // Group labels name the field honestly; the groups are never drawn as a vehicle.
  const index=new Map(shown.map(c=>[c.key,c])),seen=new Set(),groups=[];
  for(const c of shown){if(seen.has(c.key))continue;const stack=[c],members=[];seen.add(c.key);
    while(stack.length){const m=stack.pop();members.push(m);const [ix,iy]=m.key.split(',').map(Number);
      for(let dx=-1;dx<=1;dx++)for(let dy=-1;dy<=1;dy++){const k=(ix+dx)+','+(iy+dy),n=index.get(k);if(n&&!seen.has(k)){seen.add(k);stack.push(n);}}}
    const w=members.reduce((s,m)=>s+m.w,0);if(w>=6)groups.push({w,x:members.reduce((s,m)=>s+m.x*m.w,0)/w,y:members.reduce((s,m)=>s+m.y*m.w,0)/w});}
  groups.sort((a,b)=>b.w-a.w);ctx.font='10px Inter,system-ui,sans-serif';ctx.textAlign='center';
  const egoQ=view.ego?project(view.ego.x,view.ego.y):null;
  for(const g of groups.slice(0,2)){const q=project(g.x,g.y);if(egoQ&&Math.hypot(q[0]-egoQ[0],q[1]-egoQ[1])<10*scale)continue;ctx.globalAlpha=Math.min(.7,.2+g.w/15);ctx.fillStyle=C.hyp;ctx.fillText('unresolved hypotheses',q[0],q[1]-size*.45);}
  ctx.globalAlpha=1;ctx.textAlign='left';
}
function drawMap(view){
  // ResizeObserver may not have delivered a usable canvas size yet.
  if(width<=20||height<=40)return;
  ctx.setTransform(dpr,0,0,dpr,0,0);ctx.fillStyle=C.bg;ctx.fillRect(0,0,width,height);
  if(view.ego)drawPose=view.ego;
  const ego=view.ego||drawPose||{x:0,y:0,yaw:0},{project,scale,matrix}=projection(ego);
  if(backdrop){ctx.save();ctx.transform(...matrix);ctx.drawImage(backdrop,extent.x,extent.y,extent.size,extent.size);ctx.restore();}
  const center=project(ego.x,ego.y);
  ctx.save();ctx.strokeStyle='#a4c6dd14';ctx.fillStyle='#8a9cab80';ctx.font='9px Inter,system-ui,sans-serif';ctx.lineWidth=1;
  for(const r of [15,30,radius])if(r<=radius){ctx.beginPath();ctx.arc(center[0],center[1],r*scale,0,Math.PI*2);ctx.stroke();if(heading)ctx.fillText(r+' m',center[0]+4,center[1]-r*scale-4);}
  ctx.restore();
  if(view.sensor&&!view.egoStale){
    const p=view.sensor,fov=view.selected.pose.fov_deg,len=Math.min(view.selected.pose.range_m,radius*1.9),los=buildings.polygon(p,fov,len);
    if(los){ctx.beginPath();los.forEach((v,i)=>{const q=project(v[0],v[1]);i?ctx.lineTo(...q):ctx.moveTo(...q);});ctx.closePath();ctx.fillStyle=C.los+'14';ctx.fill();}
    ctx.save();ctx.setLineDash([3,6]);ctx.strokeStyle=C.los+'40';ctx.lineWidth=1;
    for(const s of [-1,1]){const a=(p.yaw+s*fov/2)*Math.PI/180,q0=project(p.x,p.y),q1=project(p.x+len*Math.cos(a),p.y+len*Math.sin(a));ctx.beginPath();ctx.moveTo(...q0);ctx.lineTo(...q1);ctx.stroke();}
    ctx.restore();
  }
  if(mode==='current'&&inspect&&$('hypothesesOverlay').checked)drawField(view,project,scale);
  // Trails: the track's own recent display positions, only for moving tracks.
  for(const row of view.visible){const tr=trails.get(row.key);if(!tr||tr.length<2||row.motion!=='moving'||row.tier!=='established')continue;
    ctx.save();ctx.lineCap='round';for(let i=1;i<tr.length;i++){const a=project(tr[i-1].x,tr[i-1].y),b=project(tr[i].x,tr[i].y);
      ctx.globalAlpha=.32*row.alpha*i/tr.length;ctx.strokeStyle=row.cls==='Pedestrian'?C.ped:C.veh;ctx.lineWidth=row.cls==='Pedestrian'?2:Math.max(2,row.dims.w*scale*.35);
      ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();}ctx.restore();}
  // Conflict groups: a dashed envelope around incompatible tracks.
  const groups=new Map();for(const row of view.visible)if(row.conflict){if(!groups.has(row.conflict))groups.set(row.conflict,[]);groups.get(row.conflict).push(row);}
  for(const [,members] of groups){if(members.length<2)continue;const pts=members.map(m=>project(m.pose.x,m.pose.y)),cx=pts.reduce((s,p)=>s+p[0],0)/pts.length,cy=pts.reduce((s,p)=>s+p[1],0)/pts.length;
    const r=Math.max(...pts.map(p=>Math.hypot(p[0]-cx,p[1]-cy)))+Math.max(12,members[0].dims.l*scale*.6);
    ctx.save();ctx.setLineDash([2,4]);ctx.strokeStyle=C.hyp+'aa';ctx.beginPath();ctx.arc(cx,cy,r,0,Math.PI*2);ctx.stroke();ctx.setLineDash([]);
    ctx.fillStyle=C.hyp;ctx.font='10px Inter,system-ui,sans-serif';ctx.textAlign='center';ctx.fillText(`${members.length} conflicting hypotheses`,cx,cy-r-4);ctx.restore();}
  const order=r=>(r.tier==='tentative'?0:r.missed?1:2)+(r.cls==='Pedestrian'?3:0);
  for(const row of view.visible.slice().sort((a,b)=>order(a)-order(b))){
    const q=project(row.pose.x,row.pose.y);if(q[0]<-40||q[0]>width+40||q[1]<-40||q[1]>height+40)continue;
    const s=styleFor(row),angle=Number.isFinite(row.yaw)?screenAngle(project,row.pose.x,row.pose.y,row.yaw):null;
    if(row.cls==='Pedestrian'){let halo=null;const u=row.motionUncertainty;
      if(row.motion==='unresolved'&&u&&engine.policy.pedestrianMotion!=='v3.0'){const a=project(row.pose.x+u.ux,row.pose.y+u.uy);
        halo={angle:Math.atan2(a[1]-q[1],a[0]-q[0]),radial:engine.policy.pedSigmaRadialM*scale,lateral:Math.max(.4,engine.policy.pedSigmaTangentialM)*scale};}
      const dir=row.motion==='moving'?screenAngle(project,row.pose.x,row.pose.y,Math.atan2(row.track.est.vy,row.track.est.vx)*180/Math.PI):null;
      person(q,Math.max(4.5,.45*scale),dir,s,halo);}
    else vehicle(q,angle,Math.max(14,row.dims.l*scale),Math.max(7,row.dims.w*scale),s);
    if((inspect&&$('ids').checked&&row.tier==='established')||selected===row.key){ctx.fillStyle='#c9d6df';ctx.font='10px ui-monospace,monospace';ctx.fillText(row.id,q[0]+10,q[1]-10);}
    if(selected===row.key){ctx.beginPath();ctx.arc(q[0],q[1],Math.max(16,row.dims.l*scale*.75),0,Math.PI*2);ctx.strokeStyle='#ffffffa0';ctx.lineWidth=1;ctx.stroke();}
  }
  if(inspect&&$('rawOverlay').checked){ // unfiltered raw candidates of the latest frame
    for(const d of view.dispositions){const xy=d.raw&&d.raw.location;if(!xy)continue;const q=project(xy.x,xy.y),strong=T.isStrong(d.raw);
      ctx.strokeStyle=C.raw;ctx.globalAlpha=strong?.95:.45;ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(q[0]-3,q[1]-3);ctx.lineTo(q[0]+3,q[1]+3);ctx.moveTo(q[0]+3,q[1]-3);ctx.lineTo(q[0]-3,q[1]+3);ctx.stroke();}
    ctx.globalAlpha=1;ctx.fillStyle=C.raw;ctx.font='10px ui-monospace,monospace';ctx.fillText('RAW OVERLAY · latest frame · unfiltered · not tracks',14,height-14);
  }
  if(view.ego){const p=view.ego,q=project(p.x,p.y),a=screenAngle(project,p.x,p.y,p.yaw),L=Math.max(22,4.8*scale),W=Math.max(11,2*scale);
    ctx.save();ctx.globalAlpha=view.egoStale?.35:1;ctx.translate(...q);ctx.rotate(a);ctx.shadowBlur=16;ctx.shadowColor=C.ego+'80';
    ctx.fillStyle=C.ego;ctx.beginPath();ctx.roundRect(-L/2,-W/2,L,W,Math.min(W*.32,5));ctx.fill();ctx.shadowBlur=0;
    ctx.fillStyle='#14463a';ctx.beginPath();ctx.roundRect(L*.06,-W*.36,L*.2,W*.72,2);ctx.fill();ctx.restore();
    ctx.fillStyle=C.ego;ctx.font='600 10px Inter,system-ui,sans-serif';ctx.textAlign='center';ctx.fillText(view.egoStale?'EGO · HELD':'EGO',q[0],q[1]+W+14);ctx.textAlign='left';}
  $('scale').style.width=(10*scale)+'px';text('compass',heading?'↑ FORWARD':'↑ N (−Y)');
  drawOverview(view);
}
function drawOverview(view){
  const cv=$('overview'),c=cv.getContext('2d');c.clearRect(0,0,cv.width,cv.height);if(!backdrop)return;
  c.drawImage(backdrop,0,0,cv.width,cv.height);const pt=p=>[(p.x-extent.x)/extent.size*cv.width,(p.y-extent.y)/extent.size*cv.height];
  if(data){c.beginPath();let seen=false;for(const e of data.events)if(e.pose&&e.arrival_ms<=now){const q=pt(e.pose);seen?c.lineTo(...q):c.moveTo(...q);seen=true;}c.strokeStyle=C.ego+'90';c.lineWidth=1.5;c.stroke();}
  if(view.ego){const q=pt(view.ego);c.fillStyle=C.ego;c.beginPath();c.arc(...q,3,0,Math.PI*2);c.fill();}
}

// ---- view refinement, HUD, audit ----------------------------------------
function refine(view){
  for(const row of view.tracks){if(!row.shown||!view.sensor)continue;
    row.staticVisibility=buildings.classify(view.sensor,row.pose,{length:row.dims.l,width:row.dims.w},Number.isFinite(row.yaw)?row.yaw:null);
    if($('buildings').checked&&row.staticVisibility.state==='STATIC_MAP_OCCLUDED'){row.reason='STATIC_MAP_OCCLUDED';row.shown=false;}}
  view.visible=view.tracks.filter(r=>r.shown);
  for(const row of view.visible){if(row.motion!=='moving'||row.tier!=='established'){trails.delete(row.key);continue;}
    const tr=trails.get(row.key)||[];if(!tr.length||now-tr[tr.length-1].t>=80){tr.push({x:row.pose.x,y:row.pose.y,t:now});while(tr.length&&now-tr[0].t>1200)tr.shift();}trails.set(row.key,tr);}
  for(const key of trails.keys())if(!view.visible.some(r=>r.key===key))trails.delete(key);
  return view;
}
function hud(view){
  $('seek').value=now;text('time',time(now));
  const sel=view.selected,age=sel?now-sel.capture_ms:null;
  text('frameLabel',sel?'FRAME '+sel.frame_id:'NO FRAME');text('ageLabel',view.egoStale?'STREAM EXPIRED':age===null?'—':`capture age ${Math.round(age)} ms`);
  text('mapNotice',view.poseReason&&view.poseReason!=='AWAITING_POSE'?'Pose unavailable — ego-follow disabled ('+view.poseReason+')':
    mode==='visible'?'Strict view: no object has qualified visibility evidence, so nothing is shown.':view.egoStale?'No new recorded update: tracks have faded and expired.':'');
  const v=view.visible,est=v.filter(r=>r.tier==='established'&&!r.conflict);
  text('nVeh',est.filter(r=>r.cls==='Vehicle').length);text('nPed',est.filter(r=>r.cls==='Pedestrian').length);
  text('nTent',v.filter(r=>r.tier==='tentative').length);text('nHyp',mode==='current'?view.counts.hypotheses:'—');
  text('radiusTag',`${radius} m · ${heading?'forward':'north up'}`);
  const k=view.counts,withheld=view.tracks.filter(r=>r.reason==='STATIC_MAP_OCCLUDED').length,conflicts=new Set(v.filter(r=>r.conflict).map(r=>r.conflict)).size;
  text('rawLine',`Latest frame: ${k.raw} raw candidates = ${k.strong} confident + ${k.weak} low-confidence. ${k.absorbed} inside an existing track's footprint · ${k.outside} outside the camera view`+
    `${withheld?` · ${withheld} withheld behind a building`:''}${conflicts?` · ${conflicts} conflicting pair${conflicts>1?'s':''}`:''}. Missed-update tracks: ${k.missed}.`);
  const iv=intervals.summary(),cs=costs.summary(),fps=intervals.values.length?1000/(intervals.values.reduce((a,b)=>a+b,0)/intervals.values.length):0;
  text('performance',`${fps.toFixed(1)} fps · frame P95 ${Number(iv.p95||0).toFixed(1)} ms · draw P95 ${Number(cs.p95||0).toFixed(2)} ms`);
  if(inspect)audit(view);
  const cams=data.camera_frames.filter(c=>c.time_ms<=now),latest=cams[cams.length-1];
  if(latest){if(lastCamera!==latest.frame_id){lastCamera=latest.frame_id;$('camera').src='/api/replay/camera/'+lastCamera+'.png';$('camera').hidden=false;$('cameraEmpty').hidden=true;}
    const d=now-latest.time_ms;text('cameraLabel',`Frame ${latest.frame_id} · ${d<130?'synchronized with the map':(d/1000).toFixed(1)+' s old'} · restored model input`);}
  else{$('camera').hidden=true;$('cameraEmpty').hidden=false;text('cameraLabel','Exact frame joins only.');}
}
const dispClass=d=>/^(ESTABLISHED|TENTATIVE|NEW_)/.test(d)?'d-est':d==='LOW_CONFIDENCE_HYPOTHESIS'?'d-hyp':'d-out';
function audit(view){
  const rows=view.dispositions.slice().sort((a,b)=>(b.raw?.score||0)-(a.raw?.score||0));
  text('auditSummary',`${rows.length} raw · frame ${view.selected?.frame_id??'—'}`);
  $('rows').replaceChildren(...rows.map(d=>{const tr=document.createElement('tr'),r=d.raw||{};
    tr.innerHTML=`<td>${esc(String(r.id||'').split(':').pop())}</td><td>${esc(r.type)}</td><td>${Number(r.score).toFixed(3)}</td><td>${T.isStrong(r)?'confident':'low'}</td><td class="${dispClass(d.disposition)}">${esc(d.disposition)}</td><td>${esc(d.track?(d.track.displayId||d.track.id):'—')}</td>`;
    tr.onclick=()=>{selected=d.track?d.track.id:null;text('detail',JSON.stringify({disposition:d.disposition,display_track:d.track?{id:d.track.displayId||d.track.id,tier:d.track.state,motion:d.track.motion,
      heading:d.track.heading,displayed_heading_deg:engine.displayYaw(d.track,now),static_visibility:view.tracks.find(r=>r.key===d.track.id)?.staticVisibility,conflict:d.track.conflict||null,updates:d.track.updates,born_capture_ms:d.track.bornT,confirmed_capture_ms:d.track.confirmedT}:null,
      measurement_unchanged:r,heading_qualification:{valid:d.entry.heading_valid,yaw_deg:d.entry.heading_yaw_deg,reason:d.entry.heading_reason,producer:d.entry.upstream?.heading||null},provenance:d.entry.upstream?.provenance||null,
      pose_binding:view.selected?.pose?.binding,visibility:d.entry.visibility},null,2));};
    return tr;}));
}
function policyText(){const p=T.POLICY;
  text('policyText',`Confident = registered evaluator threshold (vehicle ≥ ${p.strongScore.Vehicle}, person ≥ ${p.strongScore.Pedestrian}). Tracked after ${p.confirmHits} confident detections within ${p.confirmWindow} frames; low-confidence detections may sustain a track (≤ ${p.weakOnlyMaxFrames} frames) but never create one. Missed update after ${p.freshMs} ms, expiry ${p.expiryMs} ms, prediction ≤ ${p.horizonMs} ms, ${p.bridgeMs} ms correction blend, 0 ms input buffer. Stopped objects are drawn at a median anchor; motion needs a significant fit (t ≥ ${p.motionSignificance}) beyond range-error drift. No lane snapping, no GT, no averaging of different candidates.`);}

function render(wall){
  const started=performance.now();
  if(lastWall!==null){const dt=wall-lastWall;if(dt>0&&dt<1000)intervals.push(dt);
    if(playing&&data&&!loading){now=Math.min(data.end_ms,now+dt*speed);if(now===data.end_ms)setPlaying(false);}}
  lastWall=wall;
  if(data&&!loading&&buildings){
    while(cursor<data.events.length&&data.events[cursor].arrival_ms<=now)engine.observe(data.events[cursor++]);
    lastView=refine(engine.view(now,{mode,radius}));drawMap(lastView);
    if(wall-lastHUD>160){hud(lastView);lastHUD=wall;}
  }
  costs.push(performance.now()-started);frameCount++;requestAnimationFrame(render);
}
$('play').onclick=()=>{if(data&&now>=data.end_ms)seek(0);setPlaying(!playing);};
$('seek').oninput=e=>seek(Number(e.target.value));
$('clip').onchange=e=>loadClip(e.target.value);
$('source').onchange=()=>seek(now);
$('speed').onchange=e=>{speed=Number(e.target.value);};
$('policy').onchange=e=>{mode=e.target.value;lastHUD=-Infinity;};
$('buildings').onchange=()=>{lastHUD=-Infinity;};
$('radius').oninput=e=>{radius=Number(e.target.value);text('radiusValue',radius+' m');lastHUD=-Infinity;};
$('heading').onclick=()=>{heading=true;$('heading').classList.add('active');$('north').classList.remove('active');};
$('north').onclick=()=>{heading=false;$('north').classList.add('active');$('heading').classList.remove('active');};
$('inspect').onclick=()=>{inspect=!inspect;$('inspect').setAttribute('aria-pressed',inspect);$('drawer').hidden=!inspect;lastHUD=-Infinity;if(inspect)$('drawer').scrollIntoView({behavior:'smooth'});};
window.addEventListener('keydown',e=>{if(['INPUT','SELECT','TEXTAREA','BUTTON'].includes(e.target.tagName))return;
  if(e.code==='Space'){e.preventDefault();$('play').click();}
  if(e.code==='ArrowLeft'||e.code==='ArrowRight'){e.preventDefault();seek(now+(e.code==='ArrowLeft'?-1000:1000));}});
document.addEventListener('visibilitychange',()=>{if(document.hidden)setPlaying(false);lastWall=null;});
// Read-only inspection seam for browser checks.
window.replayInspection={seek,getState:()=>({pedestrianMotion:engine.policy.pedestrianMotion,clip:data?.id,loading,now,playing,cursor,frameCount,mode,radius,heading,inspect,eventCount:data?.events.length,trackCount:engine.tracks.size,bufferDelayMs:0,bridgeMs:T.POLICY.bridgeMs,hypothesesOverlay:inspect&&$('hypothesesOverlay').checked,
  stats:{...engine.stats,bridges:undefined},confirmArrivalMs:engine.stats.confirmArrivalMs.slice(),
  view:lastView?{frame:lastView.selected?.frame_id,ego:lastView.ego,poseReason:lastView.poseReason,counts:lastView.counts,egoStale:lastView.egoStale,
    shown:lastView.visible.map(r=>({key:r.key,id:r.id,cls:r.cls,tier:r.tier,missed:r.missed,motion:r.motion,z:r.track.tstat,x:r.pose.x,y:r.pose.y,yaw:r.yaw,headingKnown:r.headingKnown,directionYaw:r.directionYaw,staticVisibility:r.staticVisibility,alpha:r.alpha,conflict:r.conflict})),
    withheld:lastView.tracks.filter(r=>!r.shown).map(r=>({key:r.key,reason:r.reason})),field:lastView.field.filter(c=>c.shown).length}:null,
  intervals:intervals.values.slice(),renderCosts:costs.values.slice()})};
Promise.all([json('/api/replay/meta'),json('/api/replay/static')]).then(async([m,s])=>{
  meta=m;if(m.label)document.querySelector('.brand .sub').textContent='EGO TRAFFIC VIEW · '+m.label;statics=s;prepareMap();policyText();$('clip').replaceChildren(...m.clips.map(c=>new Option(c.name,c.id)));
  await loadClip(m.clips[0].id);
}).catch(e=>{text('mapNotice','Cannot load replay: '+e.message);});
requestAnimationFrame(render);
})();
