/* v20.3 user-facing check: map-freshness strip + "How to read" below the map, and the motion-smoothing
 * comparison (current vs rejected EMA candidate). Normal-speed recording, screenshots, layout/QoS assertions,
 * paused same-time equivalence of drawn objects v20.2 vs v20.3, live renderer on injected recorded frames.
 * Offline only: no capture, no live pipeline. Usage (recorded_tap_v20 server on REPLAY_URL):
 *   PLAYWRIGHT_MODULE=... PLAYWRIGHT_BROWSERS_PATH=... node browser_layout_motion_v20_3.cjs [--checks-only] */
'use strict';
const fs=require('fs'),path=require('path'),assert=require('assert/strict');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const base=process.env.REPLAY_URL||'http://127.0.0.1:8791',out=path.join(__dirname,'artifacts_v20_3_layout_motion');
fs.mkdirSync(out,{recursive:true});
const LABELS=()=>{window.diagLabels=0;const f=CanvasRenderingContext2D.prototype.fillText;
  CanvasRenderingContext2D.prototype.fillText=function(t,...a){if(/hypothes/i.test(String(t)))window.diagLabels++;return f.call(this,t,...a);};};
// [compare page, clip, start ms, duration ms, screenshot name]
const SEGMENTS=[['/compare-layout','turn',3000,9000,'01-layout-v20.2-vs-v20.3.png'],
  ['/compare-motion','turn',3000,11000,null],['/compare-motion','pedestrians',0,11000,'03-motion-current-vs-rejected-ema.png']];
const NUM=/^\d+(\.\d+)? (ms|s|\/s)$/;
(async()=>{
 const browser=await chromium.launch({headless:true}),errors=[],result={segments:[],layout:{},qos:{},equivalence:{},live:{}};
 const ready=(p,ids)=>p.waitForFunction(ids=>ids.every(id=>document.getElementById(id).contentWindow.replayInspection?.getState().eventCount>0),ids,{timeout:60000});
 const each=(p,src,arg)=>p.evaluate(([src,arg])=>['before','after'].map(id=>new Function('w','arg',src)(document.getElementById(id).contentWindow,arg)),[src,arg]);
 const loadClip=async(p,clip)=>{await each(p,"w.document.getElementById('clip').value=arg;w.document.getElementById('clip').dispatchEvent(new Event('change'));",clip);
   await p.waitForFunction(clip=>['before','after'].every(id=>{const s=document.getElementById(id).contentWindow.replayInspection.getState();return s.clip===clip&&!s.loading;}),clip);};
 if(!process.argv.includes('--checks-only')){
  const ctx=await browser.newContext({viewport:{width:2560,height:1000},recordVideo:{dir:out,size:{width:2560,height:1000}}});
  await ctx.addInitScript(LABELS);const page=await ctx.newPage();page.on('pageerror',e=>errors.push('compare: '+e.message));
  const start=Date.now();
  for(const [url,clip,from,duration,shot] of SEGMENTS){
   await page.goto(base+url);await ready(page,['before','after']);await loadClip(page,clip);
   await each(page,'w.replayInspection.seek(arg);',from);await page.waitForTimeout(400);
   const mark={page:url,clip,clip_start_ms:from,video_s:+((Date.now()-start)/1000).toFixed(1),duration_ms:duration};
   await each(page,"w.document.getElementById('play').click();");await page.waitForTimeout(duration);
   const st=await page.evaluate(()=>['before','after'].map(id=>document.getElementById(id).contentWindow.replayInspection.getState()));
   await each(page,"w.document.getElementById('play').click();");
   mark.clock_diff_ms=Math.abs(st[0].now-st[1].now);mark.after_labels=await page.evaluate(()=>document.getElementById('after').contentWindow.diagLabels);
   if(shot)await page.screenshot({path:path.join(out,shot)});
   result.segments.push(mark);}
  await ctx.close();fs.copyFileSync(await page.video().path(),path.join(out,'v20.3-layout-and-motion-1x.webm'));fs.unlinkSync(await page.video().path());
  for(const m of result.segments)assert.equal(m.after_labels,0,'default view drew a hypothesis/conflict label in '+m.clip);
 }
 // ---- layout: map stage unchanged, legend below the map, QoS strip under the counts ---------------------------------
 {const c=await browser.newContext({viewport:{width:1440,height:1000}});await c.addInitScript(LABELS);const p=await c.newPage();p.on('pageerror',e=>errors.push('replay: '+e.message));
  const geom=async url=>{await p.goto(base+url);await p.waitForFunction(()=>window.replayInspection?.getState().eventCount>0);
    return p.evaluate(()=>{const r=e=>{const b=document.querySelector(e)?.getBoundingClientRect();return b?{x:b.x,y:b.y,w:b.width,h:b.height,bottom:b.bottom}:null;};
      return {stage:r('#stage'),canvas:{w:document.getElementById('map').width,h:document.getElementById('map').height},legend:r('.legend'),legendInAside:!!document.querySelector('aside .legend'),
        legendInViewer:!!document.querySelector('.viewer .legend'),qos:r('.card.now .qos'),bignums:r('.card.now .bignums'),now:r('.card.now')};});};
  const before=await geom('/layout-before'),after=await geom('/');
  assert.deepEqual(after.stage,before.stage,'map stage geometry unchanged');assert.deepEqual(after.canvas,before.canvas,'map canvas size unchanged');
  assert(after.legendInViewer&&!after.legendInAside,'legend moved below the map');assert(after.legend.y>=after.stage.bottom,'legend starts below the map stage');
  assert(after.qos&&after.qos.y>=after.bignums.bottom&&after.qos.bottom<=after.now.bottom,'QoS strip sits under the object counts in the same card');
  assert(after.now.bottom<=1000,'counts + QoS card fits in the first viewport');
  result.layout={stage_before:before.stage,stage_after:after.stage,canvas_px:after.canvas,legend_top_px:after.legend.y,stage_bottom_px:after.stage.bottom,qos_box:after.qos,
    counts_card_bottom_px:after.now.bottom,legend_in_aside_before:before.legendInAside,legend_below_map_after:true};
  // QoS before any update: every value "not available" (never 0)
  await p.selectOption('#clip','turn');await p.waitForFunction(()=>{const s=window.replayInspection.getState();return s.clip==='turn'&&!s.loading;});
  await p.evaluate(()=>window.replayInspection.seek(0));await p.waitForTimeout(400);
  const empty=await p.evaluate(()=>window.replayInspection.getState());
  for(const id of ['qInstall','qAge','qRate','qSince'])assert.equal(empty.qosText[id],'not available',id+' before data');
  assert.match(empty.qosText.qObj,/^not available/);assert.match(empty.qosText.qFeedback,/^not available/);
  // QoS while playing: values present and equal to the engine-derived numbers
  await p.evaluate(()=>window.replayInspection.seek(3000));await p.click('#play');await p.waitForTimeout(5000);await p.click('#play');await p.waitForTimeout(300);
  const s=await p.evaluate(()=>window.replayInspection.getState()),q=s.qos;
  for(const id of ['qInstall','qAge','qRate','qSince'])assert.match(s.qosText[id],NUM,id);
  assert.equal(s.qosText.qAge,Math.round(q.displayedDataAgeMs)+' ms');assert.equal(s.qosText.qInstall,Math.round(q.captureToInstallMs)+' ms');
  assert(q.objectAgeMaxMs>=q.displayedDataAgeMs-1e-6,'oldest shown object observation is never newer than the latest frame');
  assert.match(s.qosText.qFeedback,/not available/);
  result.qos={before_any_update:empty.qosText,playing:s.qosText,values:q};
  // clean default and Inspect still behave as accepted in v20.2
  const vis=sel=>p.locator(sel).first().isVisible();
  const d={labels:await p.evaluate(()=>window.diagLabels),unresolvedTile:await vis('#nHyp'),rawLine:await vis('#rawLine'),hypLegend:await vis('.legend li.diag'),drawer:await vis('#drawer')};
  assert.deepEqual(d,{labels:0,unresolvedTile:false,rawLine:false,hypLegend:false,drawer:false});
  await p.screenshot({path:path.join(out,'02-v20.3-default-view.png')});
  await p.screenshot({path:path.join(out,'04-v20.3-full-page-legend-below-map.png'),fullPage:true});
  await p.click('#inspect');await p.check('#hypothesesOverlay');await p.waitForTimeout(400);
  const i={labels:await p.evaluate(()=>window.diagLabels),unresolvedTile:await vis('#nHyp'),rawLine:await vis('#rawLine'),hypLegend:await vis('.legend li.diag'),drawer:await vis('#drawer')};
  assert(i.labels>0&&i.unresolvedTile&&i.rawLine&&i.hypLegend&&i.drawer,'Inspect restores the diagnostics');
  result.clutter={default:d,inspect:i};await c.close();}
 // ---- paused same-time equivalence of drawn objects: v20.2 vs v20.3 (layout-only change) -----------------------------
 {const c=await browser.newContext({viewport:{width:2560,height:1000}}),p=await c.newPage();p.on('pageerror',e=>errors.push('equiv: '+e.message));
  await p.goto(base+'/compare-layout');await ready(p,['before','after']);let compared=0,rows=0,peds=0;
  for(const clip of ['turn','pedestrians','parked']){await loadClip(p,clip);
   for(let t=0;t<=12000;t+=250){
    const r=await p.evaluate(t=>Promise.all(['before','after'].map(id=>{const w=document.getElementById(id).contentWindow;w.replayInspection.seek(t);
      return new Promise(res=>w.requestAnimationFrame(()=>w.requestAnimationFrame(()=>res(w.replayInspection.getState().view.shown.map(x=>({key:x.key,cls:x.cls,x:x.x,y:x.y,motion:x.motion,tier:x.tier,alpha:x.alpha,displayYaw:x.displayYaw??null}))))));})),t);
    assert.deepEqual(r[1],r[0],`drawn objects differ at ${clip} ${t} ms`);compared++;rows+=r[0].length;peds+=r[0].filter(x=>x.cls==='Pedestrian').length;}}
  result.equivalence={paused_times_compared:compared,rows:rows,pedestrian_rows:peds,identical:true};await c.close();}
 // ---- live renderer, recorded frames injected locally (no running pipeline) -------------------------------------------
 {const c=await browser.newContext({viewport:{width:1440,height:1000}});await c.addInitScript(LABELS);const p=await c.newPage();p.on('pageerror',e=>errors.push('live: '+e.message));
  await p.route(base+'/live',r=>r.fulfill({contentType:'text/html',body:fs.readFileSync(path.join(__dirname,'../viewer_assets/live_v1.html'),'utf8')}));
  await p.route(base+'/api/live/**',r=>r.fulfill({status:200,contentType:r.request().url().endsWith('/stream')?'text/event-stream':'application/json',body:r.request().url().endsWith('/stream')?'':JSON.stringify({epoch:'offline-fixture',label:'OFFLINE V20 RENDERER CHECK'})}));
  const inject=withTimeline=>p.evaluate(async withTimeline=>{
   const clip=await (await fetch('/api/replay/clip/parked')).json(),ev=clip.events.filter(e=>e.arrival_ms<=8000).slice(-40),base=BigInt(clip.base_ns),epoch='offline-fixture-'+withTimeline;
   const send=m=>window.liveInspection._onMessage(JSON.stringify(m)),ns=t=>String(base+BigInt(Math.round(t*1e6)));
   send({type:'hello',epoch,session:ev[0].session,base_ns:String(base),server_ns:ns(ev[0].arrival_ms)});
   // The live feed's timeline carries the recorded install stamp; the fixture without it must show "not available".
   ev.forEach((e,i)=>send({...e,type:'event',seq:i+1,epoch,age_mapping:'LIVE_SAME_HOST_WALL',server_ns:ns(e.arrival_ms),...(withTimeline?{timeline:{install_ms:e.arrival_ms}}:{})}));
   await new Promise(r=>setTimeout(r,400));return window.liveInspection.getState();},withTimeline);
  await p.goto(base+'/live');await p.waitForFunction(()=>window.liveInspection&&window.ScenesenseBuildings);await p.waitForTimeout(500);
  const a=await inject(true);assert.equal(a.accepted,40);assert.match(a.qosText.qInstall,NUM);assert.match(a.qosText.qAge,NUM);assert.match(a.qosText.qRate,NUM);
  assert.match(a.qosText.qFeedback,/not available/);assert.equal(await p.evaluate(()=>window.diagLabels),0);
  const lg=await p.evaluate(()=>({inViewer:!!document.querySelector('.viewer .legend'),inAside:!!document.querySelector('aside .legend'),
    stageBottom:document.getElementById('stage').getBoundingClientRect().bottom,legendTop:document.querySelector('.legend').getBoundingClientRect().top}));
  assert(lg.inViewer&&!lg.inAside&&lg.legendTop>=lg.stageBottom,'live: legend below the map');
  await p.screenshot({path:path.join(out,'05-live-renderer-offline-fixture.png')});
  await p.goto(base+'/live');await p.waitForFunction(()=>window.liveInspection&&window.ScenesenseBuildings);await p.waitForTimeout(500);
  const b=await inject(false);assert.equal(b.qosText.qInstall,'not available','no install stamp -> not available');assert.match(b.qosText.qAge,NUM);
  result.live={injected_recorded_frames:40,with_install_stamp:a.qosText,without_install_stamp:b.qosText,legend_below_map:true,default_canvas_labels:0};await c.close();}
 assert.deepEqual(errors,[]);
 const report={status:'PASS',browser:browser.version(),conditions:'1x real rAF, 45 m heading-up; compare pages 2560x1000 recorded; checks 1440x1000. Offline replay of sealed V20 taps; live renderer fed recorded frames locally. No capture or live pipeline.',...result,page_errors:errors};
 fs.writeFileSync(path.join(out,'BROWSER_CHECK.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,1));await browser.close();
})().catch(e=>{console.error(e);process.exit(1);});
