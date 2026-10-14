/* Canvas replay UI. Isolated from live ingestion; at most 256 immutable events. */
(function() {
'use strict';
const R=window.ScenesenseReplay,P=window.ScenesenseMapPresentation,I=window.ScenesenseMapInventory;
const $=id=>document.getElementById(id), canvas=$('map'),ctx=canvas.getContext('2d');
const engine=new R.Engine({refined:true}), intervals=new I.RollingBuffer(600),costs=new I.RollingBuffer(600);
let data=null,meta=null,statics=null,roadIndex=null,buildings=null,backdrop=null,extent=null;
let now=0,cursor=0,playing=false,speed=1,heading=true,debug=false,radius=45,mode='current';
let lastWall=null,lastHUD=-Infinity,lastView=null,selectedKey=null,loadGeneration=0,lastCamera=null,frameCount=0;
let width=1,height=1,dpr=1,drawPose=null,loading=false;
const palette={ego:'#7ee7c5',car:'#72c8ef',person:'#e5a5cf',coast:'#edc587',raw:'#e6bd7b'};
const time=t=>{t=Math.max(0,t)/1000;return String(Math.floor(t/60)).padStart(2,'0')+':'+(t%60).toFixed(1).padStart(4,'0');};
const text=(id,t)=>{$(id).textContent=t;};
const esc=t=>String(t).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function json(url){const response=await fetch(url);if(!response.ok)throw Error(`${url}: HTTP ${response.status}`);return response.json();}
function setPlaying(value){playing=value;text('play',playing?'Ⅱ Pause':'▶ Play');$('play').setAttribute('aria-label',playing?'Pause replay':'Play replay');}
function seek(t){
  if(!data)return;
  now=Math.max(0,Math.min(data.end_ms,t));engine.reset($('source').value||data.sources[0]);cursor=0;drawPose=null;
  // Rebuild only bounded causal history. No future observation enters motion estimates.
  while(cursor<data.events.length&&data.events[cursor].arrival_ms<now-1000)cursor++;
  while(cursor<data.events.length&&data.events[cursor].arrival_ms<=now)engine.observe(data.events[cursor++]);
  selectedKey=null;lastHUD=-Infinity;lastCamera=null;
  text('detail','Select an object row to inspect its unchanged measurement, source, frame, pose binding, and display decision.');
}
async function loadClip(name){
  const generation=++loadGeneration;setPlaying(false);loading=true;text('mapNotice','Loading recorded events…');
  try{
    const clip=await json('/api/replay/clip/'+encodeURIComponent(name));
    if(generation!==loadGeneration)return;
    if(clip.events.length>R.SETTINGS.maxEvents)throw Error('Replay event bound exceeded');
    data=clip;$('source').replaceChildren(...clip.sources.map((s,i)=>new Option('Ego '+(i+1)+' · '+s.replace(/^ue288_/,'').slice(-34),s)));
    $('cameraFrame').replaceChildren(new Option('Select frame…',''),...clip.camera_frames.map(c=>new Option('Frame '+c.frame_id,c.frame_id)));
    $('seek').max=clip.end_ms;text('duration',time(clip.end_ms));seek(0);
    text('sequenceNote','Recorded install cadence · final 1.5 s shows stream expiry');
    text('footerStatus',`${clip.events.length} events buffered / ${R.SETTINGS.maxEvents} cap · local replay only`);
  }catch(e){text('mapNotice','Replay unavailable: '+e.message);text('footerStatus',e.message);}
  finally{if(generation===loadGeneration)loading=false;}
}
function resize(){
  const r=$('stage').getBoundingClientRect();width=r.width;height=r.height;dpr=Math.min(2,window.devicePixelRatio||1);
  canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);
}
new ResizeObserver(resize).observe($('stage'));
function prepareMap(){
  buildings=new window.ScenesenseBuildings.Buildings(statics);
  const points=[...statics.roads.flat(),...statics.buildings.flat()];
  if(!points.length){extent={x:-150,y:-150,size:300};backdrop=null;roadIndex=null;return;}
  const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]);
  extent={x:Math.min(...xs)-8,y:Math.min(...ys)-8};
  extent.size=Math.max(Math.max(...xs)-extent.x,Math.max(...ys)-extent.y)+8;
  backdrop=document.createElement('canvas');backdrop.width=backdrop.height=2048;
  const c=backdrop.getContext('2d'),s=2048/extent.size;
  c.fillStyle='#101c27';c.fillRect(0,0,2048,2048);c.setTransform(s,0,0,s,-extent.x*s,-extent.y*s);
  function path(points){c.beginPath();points.forEach((p,i)=>i?c.lineTo(p[0],p[1]):c.moveTo(p[0],p[1]));}
  c.fillStyle='#1b2a36';c.strokeStyle='#293d4a';c.lineWidth=.2;
  for(const b of statics.buildings){path(b);c.closePath();c.fill();c.stroke();}
  c.lineCap='round';c.lineJoin='round';
  for(const [color,lw] of [['#0b141d',5.3],['#263947',4.2],['#415361',.35]]){
    c.strokeStyle=color;c.lineWidth=lw;for(const road of statics.roads){path(road);c.stroke();}
  }
  roadIndex=new P.RoadHeadingIndex(statics.roads);
}
function projection(ego){
  const scale=Math.min(width,height)/(radius*2),cx=width*.5,cy=height*.56;
  const a=heading?ego.yaw*Math.PI/180:0;
  const ex=heading?-Math.sin(a):1,ey=heading?Math.cos(a):0;
  const fx=heading?-Math.cos(a):0,fy=heading?-Math.sin(a):1;
  const project=(x,y)=>[cx+((x-ego.x)*ex+(y-ego.y)*ey)*scale,cy+((x-ego.x)*fx+(y-ego.y)*fy)*scale];
  return {project,scale,matrix:[ex*scale,fx*scale,ey*scale,fy*scale,cx-(ego.x*ex+ego.y*ey)*scale,cy-(ego.x*fx+ego.y*fy)*scale]};
}
function shape(points,project){ctx.beginPath();points.forEach((p,i)=>{const q=project(p[0],p[1]);i?ctx.lineTo(...q):ctx.moveTo(...q);});}
function drawMap(view){
  ctx.setTransform(dpr,0,0,dpr,0,0);ctx.fillStyle='#101c27';ctx.fillRect(0,0,width,height);
  // Holding a last camera centre is explicitly distinct from a valid ego pose.
  if(view.ego)drawPose=view.ego;
  const ego=view.ego||drawPose||{x:0,y:0,yaw:0};
  const {project,scale,matrix}=projection(ego);
  if(backdrop){ctx.save();ctx.transform(...matrix);ctx.drawImage(backdrop,extent.x,extent.y,extent.size,extent.size);ctx.restore();}
  const center=project(ego.x,ego.y);
  ctx.save();ctx.strokeStyle='#a4c6dd13';ctx.lineWidth=1;
  for(const r of [15,30,45,60,75])if(r<=radius){ctx.beginPath();ctx.arc(...center,r*scale,0,Math.PI*2);ctx.stroke();}
  ctx.restore();
  if(view.sensor && !view.egoStale){
    const p=view.sensor,half=view.selected.pose.fov_deg/2;
    const length=Math.min(view.selected.pose.range_m,radius*1.8),vertices=[[p.x,p.y]];
    for(let angle=-half;angle<=half+.01;angle+=half/32){const a=(p.yaw+angle)*Math.PI/180;vertices.push([p.x+length*Math.cos(a),p.y+length*Math.sin(a)]);}
    shape(vertices,project);ctx.closePath();ctx.strokeStyle='#74b5d94d';ctx.lineWidth=1;ctx.setLineDash([4,5]);ctx.stroke();ctx.setLineDash([]);
    const los=buildings.polygon(p,view.selected.pose.fov_deg,length);
    if(los){shape(los,project);ctx.closePath();ctx.fillStyle='#77c8e821';ctx.fill();ctx.strokeStyle='#77c8e830';ctx.stroke();}
    ctx.fillStyle='#a0bac9';ctx.font='10px ui-sans-serif';ctx.fillText(los?'FILL · STATIC BUILDING LOS ESTIMATE':'STATIC LOS UNAVAILABLE',18,height-86);
    const [sx,sy]=project(p.x,p.y);ctx.fillStyle='#a8d6e6';ctx.beginPath();ctx.arc(sx,sy,2,0,Math.PI*2);ctx.fill();
  }
  const rows=debug?view.rawRows:view.visible;
  for(const row of rows){
    const raw=row.entry.raw,pose=debug?{...raw.location,yaw:raw.yaw_deg}:row.pose;
    if(!pose)continue;
    const q=project(pose.x,pose.y);if(q[0]<-30||q[0]>width+30||q[1]<-30||q[1]>height+30)continue;
    const ped=raw.type==='Pedestrian',color=debug?palette.raw:row.ambiguous?palette.coast:row.coast?palette.coast:ped?palette.person:palette.car;
    let yaw=pose.yaw;
    // Producer world yaw / bounded private motion heading; no road override.
    const direction=project(pose.x+Math.cos(yaw*Math.PI/180),pose.y+Math.sin(yaw*Math.PI/180));
    const angle=Math.atan2(direction[1]-q[1],direction[0]-q[0]);
    const dims=I.canonicalSize(raw.type,raw.dimensions),length=Math.max(12,Math.min(36,dims[0]*scale)),breadth=Math.max(7,Math.min(16,dims[1]*scale));
    ctx.save();ctx.globalAlpha=debug?.8:row.alpha;
    if(!debug&&row.state!=='measured'){
      const measured=project(raw.location.x,raw.location.y);ctx.beginPath();ctx.moveTo(...measured);ctx.lineTo(...q);ctx.strokeStyle=color+'65';ctx.lineWidth=1;ctx.stroke();
    }
    ctx.translate(...q);ctx.rotate(angle);ctx.strokeStyle=color;ctx.fillStyle=color+'22';ctx.lineWidth=1.5;
    if(!debug&&row.state!=='measured')ctx.setLineDash(row.coast?[3,3]:[5,2]);
    if(!debug&&row.ambiguous){
      ctx.setLineDash([]);ctx.beginPath();ctx.moveTo(0,-4);ctx.lineTo(4,0);ctx.lineTo(0,4);ctx.lineTo(-4,0);ctx.closePath();ctx.fill();ctx.stroke();
    }else if(ped){
      ctx.beginPath();ctx.arc(0,0,5,0,Math.PI*2);ctx.fill();ctx.stroke();ctx.setLineDash([]);
      ctx.beginPath();ctx.moveTo(-2,-2);ctx.lineTo(2,2);ctx.moveTo(2,-2);ctx.lineTo(-2,2);ctx.stroke();
    }else{
      ctx.beginPath();ctx.roundRect(-length/2,-breadth/2,length,breadth,3);ctx.fill();ctx.stroke();ctx.setLineDash([]);
      // Window/wheels make a recognizable glyph; directional nose only with a known heading.
      ctx.fillStyle=color+'50';ctx.fillRect(-length*.12,-breadth*.32,length*.28,breadth*.64);
      if(row.heading!=='unknown'&&!debug){ctx.beginPath();ctx.moveTo(length/2-3,-breadth*.3);ctx.lineTo(length/2-3,breadth*.3);ctx.stroke();}
    }
    ctx.restore();
    if(!debug&&row.entry.members?.length>1){ctx.fillStyle='#0b141d';ctx.fillRect(q[0]+5,q[1]-17,22,13);ctx.fillStyle='#c2e9fc';ctx.font='10px ui-monospace';ctx.fillText('×'+row.entry.members.length,q[0]+7,q[1]-7);}
    ctx.font='9px ui-monospace, monospace';ctx.fillStyle=color;
    const suffix=String(row.entry.track_id).split('_').pop().replace(/^0+/,'')||'0';
    // Dense model candidates stay intact; IDs live in the audit instead of
    // obscuring neighbouring glyphs. A selected track remains labelled.
    if(debug||selectedKey===row.key)ctx.fillText(debug?raw.id.split(':').pop():suffix+(row.coast?' C':' ?'),q[0]+9,q[1]-7);
    if(selectedKey===row.key){ctx.beginPath();ctx.arc(...q,22,0,Math.PI*2);ctx.strokeStyle='#ffffffa0';ctx.lineWidth=1;ctx.stroke();}
  }
  if(view.ego){
    const p=view.ego,q=project(p.x,p.y),n=project(p.x+Math.cos(p.yaw*Math.PI/180),p.y+Math.sin(p.yaw*Math.PI/180));
    ctx.save();ctx.globalAlpha=view.egoStale?.35:1;ctx.translate(...q);ctx.rotate(Math.atan2(n[1]-q[1],n[0]-q[0]));
    ctx.shadowBlur=18;ctx.shadowColor='#7ee7c560';ctx.fillStyle=palette.ego;ctx.strokeStyle='#c5ffea';ctx.lineWidth=1.5;
    ctx.beginPath();ctx.roundRect(-15,-7.5,30,15,4);ctx.fill();ctx.stroke();ctx.shadowBlur=0;
    ctx.fillStyle='#1d5144';ctx.fillRect(0,-5,6,10);ctx.beginPath();ctx.moveTo(19,0);ctx.lineTo(24,0);ctx.stroke();ctx.restore();
    ctx.fillStyle=palette.ego;ctx.font='bold 10px ui-sans-serif';ctx.textAlign='center';ctx.fillText(view.egoStale?'EGO · HELD':'EGO',q[0],q[1]+28);ctx.textAlign='left';
  }
  if(debug){ctx.fillStyle='#e6bd7b';ctx.font='11px ui-monospace';ctx.fillText('RAW MEASUREMENTS · UNFILTERED · NOT EGO OBSERVATIONS',20,height-65);}
  $('scale').style.width=(10*scale)+'px';text('compass',heading?'↑ FORWARD':'↑ N (−Y)');
  drawOverview(view);
}
function drawOverview(view){
  const cv=$('overview'),c=cv.getContext('2d');c.clearRect(0,0,cv.width,cv.height);
  if(!backdrop)return;c.drawImage(backdrop,0,0,cv.width,cv.height);
  const point=p=>[(p.x-extent.x)/extent.size*cv.width,(p.y-extent.y)/extent.size*cv.height];
  if(data){c.beginPath();let seen=false;for(const e of data.events)if(e.pose&&e.arrival_ms<=now){const q=point(e.pose);seen?c.lineTo(...q):c.moveTo(...q);seen=true;}c.strokeStyle='#7ee7c572';c.lineWidth=1.5;c.stroke();}
  if(view.ego){const q=point(view.ego);c.strokeStyle='#72c8ef';c.lineWidth=1;c.beginPath();c.arc(...q,radius/extent.size*cv.width,0,Math.PI*2);c.stroke();c.fillStyle=palette.ego;c.beginPath();c.arc(...q,3,0,Math.PI*2);c.fill();}
}
function detail(row){
  selectedKey=row.key;
  text('detail',JSON.stringify({display:{track:row.entry.track_id,reason:row.reason||'ADMITTED',state:row.state,
    visibility:row.visibility,age_ms:row.ageMs,clock:row.clockMode,heading:row.heading||'unknown',
    predicted:row.pose,display_reset:row.reset,display_generation:row.generation,
    pose_binding:row.event.pose?.binding,static_building_estimate:row.staticVisibility,ambiguous:row.ambiguous,
    representative:row.representative,members:row.entry.members?.map(m=>({id:m.raw.id,candidate:m.upstream?.candidate,provenance:m.upstream?.provenance})),
    heading_provenance:row.entry.upstream,backend_track:row.entry.backend_track_id},measurement:row.entry.raw},null,2));
}
function hud(view){
  $('seek').value=now;text('time',time(now));
  text('viewLabel',debug?'RAW / UNFILTERED':mode==='visible'?'STRICT VISIBLE ONLY':mode==='coast'?'FORWARD + COASTING':'FORWARD OBSERVATIONS');
  text('frameLabel',view.selected?'FRAME '+view.selected.frame_id:'NO FRAME');
  const notice=view.poseReason ? 'Pose unavailable — ego-follow and FoV disabled. '+view.poseReason :
    mode==='visible'?'No qualified visibility evidence: strict visible-only view withholds all unknown objects.':
    view.egoStale?'Stream expired · held ego reference; observations removed.':
    view.clockMode==='arrival'?'Clock mapping unavailable · animation uses arrival age only.':'';
  text('mapNotice',notice);
  const age=view.selected?(view.clockMode==='capture'?now-view.selected.capture_ms:now-view.selected.arrival_ms):null;
  text('count',debug?view.rawRows.length:view.visible.length);text('hidden',view.rawRows.filter(r=>!r.shown).length);
  text('age',age===null?'—':Math.round(age));$('agebar').style.width=Math.min(100,(age||0)/750*100)+'%';
  text('clockTag',view.clockMode==='capture'?'CAPTURE CLOCK':'ARRIVAL ONLY');
  const predicted=view.visible.filter(r=>r.state==='predicted').length,coast=view.visible.filter(r=>r.coast).length;
  text('stateLine',`${predicted} predicted · ${coast} coasting · ${view.visible.filter(r=>r.state==='reset').length} resetting · visibility unknown`);
  const consolidated=view.rows.filter(r=>r.reason==='CONSOLIDATED_HYPOTHESIS').length,occluded=view.rows.filter(r=>r.reason==='STATIC_MAP_OCCLUDED').length;
  text('auditSummary',`${view.rawRows.length} raw this frame · ${consolidated} consolidated · ${occluded} building-withheld · ${view.visible.filter(r=>r.ambiguous).length} uncertain diamonds`);
  text('coverage',`${view.visible.length} glyphs · ${consolidated} overlapping hypotheses represented by × badges · ${occluded} fully building-occluded. Diamonds mark conflicting image/world evidence.`);
  const interval=intervals.summary(),cost=costs.summary();
  const fps=intervals.values.length?1000/(intervals.values.reduce((a,b)=>a+b,0)/intervals.values.length):0;
  text('performance',`${fps.toFixed(1)} fps · frame P95 ${Number(interval.p95||0).toFixed(1)} ms · draw P95 ${Number(cost.p95||0).toFixed(2)} ms`);
  const rows=(debug?view.rawRows:view.rows).slice().sort((a,b)=>Number(b.shown)-Number(a.shown));
  $('rows').replaceChildren();
  for(const row of rows){const tr=document.createElement('tr'),raw=row.entry.raw;
    const id=String(row.entry.track_id).replace('coop_track_','T');
    tr.innerHTML=`<td>${esc(id)} / ${esc(raw.id.split(':').pop())}</td><td>${esc(raw.type)}</td><td>${esc(debug?'raw':row.state||'untracked')}</td><td>${esc(row.visibility.state)}</td><td>${row.ageMs===undefined?'—':Math.round(row.ageMs)+' ms'}</td><td class="${row.reason?'reason':'admitted'}">${esc(row.reason|| (row.staticVisibility?.state==='VISIBILITY_UNCERTAIN'?'VISIBILITY_UNCERTAIN':row.ambiguous?'AMBIGUOUS_LOCALIZATION':'FORWARD · ADMITTED'))}</td>`;
    tr.onclick=()=>detail(row);$('rows').append(tr);
  }
  if(selectedKey){const row=rows.find(r=>r.key===selectedKey);if(row)detail(row);}
  const cameras=data.camera_frames.filter(c=>c.time_ms<=now);
  const latest=cameras[cameras.length-1];
  if(latest){
    if(lastCamera!==latest.frame_id){lastCamera=latest.frame_id;$('camera').src='/api/replay/camera/'+lastCamera+'.png';$('camera').hidden=false;$('cameraEmpty').hidden=true;}
    const delta=now-latest.time_ms;
    text('cameraLabel',`Frame ${latest.frame_id} · ${delta<1?'synchronized map event':(delta/1000).toFixed(1)+' s since this sample'} · restored 768×432 RGB`);
  }else{$('camera').hidden=true;$('cameraEmpty').hidden=false;text('cameraLabel','Exact frame joins · select a retained sample');}
}
function refineView(view){
  const rows=view.rows;
  for(const row of rows){
    if(!row.shown)continue;
    row.staticVisibility=buildings.classify(view.sensor,row.pose,row.entry.raw.dimensions,row.entry.heading_valid?row.pose.yaw:null);
    if($('buildings').checked&&row.staticVisibility.state==='STATIC_MAP_OCCLUDED'){row.reason='STATIC_MAP_OCCLUDED';row.shown=false;}
    row.ambiguous=rows.some(other=>other!==row&&other.current&&row.current&&other.entry.raw.type===row.entry.raw.type&&
      window.ScenesenseReplayMotion.box(row.entry)&&window.ScenesenseReplayMotion.box(other.entry)&&
      Math.hypot(row.entry.raw.location.x-other.entry.raw.location.x,row.entry.raw.location.y-other.entry.raw.location.y)<1&&
      window.ScenesenseReplayMotion.overlap(window.ScenesenseReplayMotion.box(row.entry),window.ScenesenseReplayMotion.box(other.entry)).iou<.05);
  }
  view.visible=rows.filter(r=>r.shown);return view;
}
function render(wall){
  const started=performance.now();
  if(lastWall!==null){const dt=wall-lastWall;if(dt>0&&dt<1000)intervals.push(dt);
    if(playing&&data&&!loading){now=Math.min(data.end_ms,now+dt*speed);if(now===data.end_ms)setPlaying(false);}}
  lastWall=wall;
  if(data&&!loading){
    while(cursor<data.events.length&&data.events[cursor].arrival_ms<=now)engine.observe(data.events[cursor++]);
    lastView=refineView(engine.view(now,{mode,radius}));drawMap(lastView);
    if(wall-lastHUD>160){hud(lastView);lastHUD=wall;}
  }
  costs.push(performance.now()-started);frameCount++;requestAnimationFrame(render);
}
$('grouping').onchange=e=>{engine.consolidation=e.target.checked;seek(now);};
$('buildings').onchange=()=>{lastHUD=-Infinity;};
$('play').onclick=()=>{if(data&&now>=data.end_ms)seek(0);setPlaying(!playing);};
$('seek').oninput=e=>seek(Number(e.target.value));
$('clip').onchange=e=>loadClip(e.target.value);
$('source').onchange=()=>seek(now);
$('speed').onchange=e=>{speed=Number(e.target.value);};
$('policy').onchange=e=>{mode=e.target.value;lastHUD=-Infinity;};
$('radius').oninput=e=>{radius=Number(e.target.value);text('radiusValue',radius+' m');};
$('heading').onclick=()=>{heading=true;$('heading').classList.add('active');$('north').classList.remove('active');};
$('north').onclick=()=>{heading=false;$('north').classList.add('active');$('heading').classList.remove('active');};
$('debug').onclick=()=>{debug=!debug;$('debug').setAttribute('aria-pressed',debug);document.body.classList.toggle('debug',debug);lastHUD=-Infinity;};
$('cameraFrame').onchange=e=>{const frame=data.camera_frames.find(c=>String(c.frame_id)===e.target.value);if(frame){setPlaying(false);seek(frame.time_ms);}};
window.addEventListener('keydown',e=>{if(['INPUT','SELECT','TEXTAREA','BUTTON'].includes(e.target.tagName))return;
  if(e.code==='Space'){e.preventDefault();$('play').click();}
  if(e.code==='ArrowLeft'||e.code==='ArrowRight'){e.preventDefault();seek(now+(e.code==='ArrowLeft'?-1000:1000));}});
document.addEventListener('visibilitychange',()=>{if(document.hidden)setPlaying(false);lastWall=null;});
// Read-only inspection seam: actual rendered state and bounded timing windows.
window.replayInspection={getState:()=>({now,playing,cursor,frameCount,mode,debug,radius,heading,
  eventCount:data?.events.length,trackCount:engine.tracks.size,metrics:engine.predictor.metrics,churn:engine.churn,bufferDelayMs:0,bridgeMs:100,rejections:{...engine.rejections},
  view:lastView?{frame:lastView.selected?.frame_id,ego:lastView.ego,poseReason:lastView.poseReason,
    shown:lastView.visible.map(r=>({key:r.key,x:r.pose.x,y:r.pose.y,alpha:r.alpha,state:r.state,age:r.ageMs,raw_id:r.entry.raw.id,raw:r.entry.raw.location,yaw:r.pose.yaw,members:r.entry.members?.length||1,ambiguous:r.ambiguous})),
    rawCount:lastView.rawRows.length,rows:lastView.rows.map(r=>({key:r.key,id:r.entry.raw.id,reason:r.reason,shown:r.shown,staticVisibility:r.staticVisibility,representative:r.representative,members:r.entry.members?.map(m=>m.raw.id),backend:r.entry.backend_track_id})),reasons:lastView.rows.map(r=>r.reason),unknown:lastView.rows.filter(r=>r.visibility.state==='unknown').length}:null,
  intervals:intervals.values.slice(),renderCosts:costs.values.slice()})};
Promise.all([json('/api/replay/meta'),json('/api/replay/static')]).then(async([m,s])=>{
  meta=m;statics=s;prepareMap();$('clip').replaceChildren(...m.clips.map(c=>new Option(c.name,c.id)));
  text('calibration',`${m.calibration.fov_deg.toFixed(0)}° calibrated camera FoV · ${m.calibration.range_m} m configured radar envelope. View radius is independent. Dashed outline: calibrated FoV. Fill: first-building LOS approximation. Qualified visibility remains unknown.`);
  await loadClip(m.clips[0].id);
}).catch(e=>{text('mapNotice','Cannot load replay: '+e.message);text('footerStatus',e.message);});
requestAnimationFrame(render);
})();
