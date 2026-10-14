/* Short, actual 1x browser comparison plus offline live-renderer fixture. No live capture. */
'use strict';
const fs=require('fs'),path=require('path'),assert=require('assert/strict');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const base=process.env.REPLAY_URL||'http://127.0.0.1:8776',out=path.join(__dirname,'artifacts_v20_heading');
const summary=a=>{const s=a.slice().sort((a,b)=>a-b);return {n:s.length,p50:s[Math.floor(s.length*.5)]||0,p95:s[Math.floor(s.length*.95)]||0,max:s.at(-1)||0};};
(async()=>{
 const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM?{executablePath:process.env.CHROMIUM}:{})}),errors=[],requests=[];
 const marks=[],clock=[];
 if(!process.argv.includes('--checks-only')){
 const ctx=await browser.newContext({viewport:{width:2560,height:1000},recordVideo:{dir:out,size:{width:2560,height:1000}}});
 const page=await ctx.newPage();page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>requests.push(r.url()));
 await page.goto(base+'/compare-heading');
 await page.waitForFunction(()=>['before','after'].every(id=>document.getElementById(id).contentWindow.replayInspection?.getState().eventCount>0),null,{timeout:60000});
 const each=(src,arg)=>page.evaluate(([src,arg])=>['before','after'].map(id=>new Function('w','arg',src)(document.getElementById(id).contentWindow,arg)),[src,arg]);
 let start=Date.now();
 for(const [clip,duration] of [['parked',9000],['turn',12000],['building',4300]]){
  await each("w.document.getElementById('clip').value=arg;w.document.getElementById('clip').dispatchEvent(new Event('change'));",clip);
  await page.waitForFunction(clip=>['before','after'].every(id=>{const s=document.getElementById(id).contentWindow.replayInspection.getState();return s.clip===clip&&!s.loading;}),clip);
  await each('w.replayInspection.seek(3000);');await page.waitForTimeout(200);
  marks.push({clip,clip_start_ms:3000,video_s:(Date.now()-start)/1000,duration_ms:duration});
  await each("w.document.getElementById('play').click();");
  const deltas=await page.evaluate(async ms=>{const a=[],end=performance.now()+ms;while(performance.now()<end){await new Promise(requestAnimationFrame);const s=['before','after'].map(id=>document.getElementById(id).contentWindow.replayInspection.getState());a.push(Math.abs(s[0].now-s[1].now));}return a;},duration);
  clock.push({clip,difference_ms:summary(deltas)});
  await each("w.document.getElementById('play').click();");
  if(clip==='building'){await each('w.replayInspection.seek(5240);');await page.waitForTimeout(200);}
  await page.screenshot({path:path.join(out,clip+'-comparison.png')});
 }
 await ctx.close();fs.copyFileSync(await page.video().path(),path.join(out,'v20-heading-before-after-1x.webm'));
 fs.writeFileSync(path.join(out,'VIDEO_TIMING.json'),JSON.stringify({marks,clock},null,2)+'\n');
 }
 // Standalone render timing, no recorder. Instrument the canvas label to verify its actual gate.
 const perf={};let overlay;
 for(const [name,url] of [['before','/heading-before'],['after','/']]){
  const c=await browser.newContext({viewport:{width:1440,height:1000}}),p=await c.newPage();p.on('pageerror',e=>errors.push(name+': '+e.message));
  await p.addInitScript(()=>{window.hypothesisLabels=0;const f=CanvasRenderingContext2D.prototype.fillText;CanvasRenderingContext2D.prototype.fillText=function(t,...a){if(t==='unresolved hypotheses')window.hypothesisLabels++;return f.call(this,t,...a);};});
  await p.goto(base+url);await p.waitForFunction(()=>window.replayInspection?.getState().eventCount>0);
  await p.evaluate(()=>window.replayInspection.seek(3000));await p.click('#play');await p.waitForTimeout(6500);await p.click('#play');
  let s=await p.evaluate(()=>window.replayInspection.getState());perf[name]={frame_ms:summary(s.intervals.slice(-350)),draw_ms:summary(s.renderCosts.slice(-350))};
  if(name==='after'){
   assert.equal(await p.evaluate(()=>window.hypothesisLabels),0,'default replay has no haze labels');
   // A stationary replay fixture with unresolved pedestrians; showing the low-score overlay cannot hide them.
   await p.evaluate(()=>window.replayInspection.seek(7500));await p.waitForTimeout(180);
   s=await p.evaluate(()=>window.replayInspection.getState());assert(s.view.shown.some(r=>r.cls==='Pedestrian'&&r.motion==='unresolved'));
   const before=s.view.shown.map(r=>r.key).sort();await p.click('#inspect');await p.check('#hypothesesOverlay');await p.waitForTimeout(250);
   const active=await p.evaluate(()=>({s:window.replayInspection.getState(),labels:window.hypothesisLabels}));
   assert(active.s.hypothesesOverlay);assert(active.labels>0);assert.deepEqual(active.s.view.shown.map(r=>r.key).sort(),before);
   await p.locator('#rows tr').first().click();const detail=await p.locator('#detail').innerText();assert(detail.includes('heading_qualification'));
   await p.screenshot({path:path.join(out,'inspect-overlay.png'),fullPage:true});
   await p.click('#inspect');const n=await p.evaluate(()=>window.hypothesisLabels);await p.waitForTimeout(200);
   assert.equal(await p.evaluate(()=>window.hypothesisLabels),n,'closing Inspect hides overlay again');
   overlay={replay_default_hidden:true,replay_restores_in_inspect:true,unresolved_pedestrian_visible:true,overlay_changes_presence:false};
  }
  await c.close();
 }
 // Load the actual live renderer with network responses stubbed locally. Inject only recorded frames.
 const c=await browser.newContext({viewport:{width:1440,height:1000}}),p=await c.newPage();p.on('pageerror',e=>errors.push('live fixture: '+e.message));
 await p.route(base+'/live',r=>r.fulfill({contentType:'text/html',body:fs.readFileSync(path.join(__dirname,'../viewer_assets/live_v1.html'),'utf8')}));
 await p.route(base+'/api/live/**',r=>r.fulfill({status:200,contentType:r.request().url().endsWith('/stream')?'text/event-stream':'application/json',body:r.request().url().endsWith('/stream')?'':JSON.stringify({epoch:'offline-fixture',label:'OFFLINE V20 RENDERER CHECK'})}));
 await p.goto(base+'/live');await p.waitForFunction(()=>window.liveInspection&&window.ScenesenseBuildings);
 await p.waitForTimeout(500);
 const live=await p.evaluate(async()=>{
  const clip=await (await fetch('/api/replay/clip/parked')).json(),ev=clip.events.filter(e=>e.arrival_ms<=7500).slice(-20),base=BigInt(clip.base_ns),epoch='offline-fixture';
  const send=m=>window.liveInspection._onMessage(JSON.stringify(m));
  // Constant virtual wall stamps replay historical arrival times; no source evidence is rewritten.
  const ns=t=>String(base+BigInt(Math.round(t*1e6)));
  send({type:'hello',epoch,session:ev[0].session,base_ns:String(base),server_ns:ns(ev[0].arrival_ms)});
  ev.forEach((e,i)=>send({...e,type:'event',seq:i+1,epoch,server_ns:ns(e.arrival_ms)}));
  await new Promise(requestAnimationFrame);return window.liveInspection.getState();
 });
 assert.equal(live.accepted,20);assert.equal(live.hypothesesOverlay,false);assert(live.view.shown.some(r=>r.cls==='Pedestrian'&&r.motion==='unresolved'));
 // Rapid toggles keep this fixed recorded fixture within the existing 750-ms expiry.
 await p.locator('#inspect').evaluate(e=>e.click());await p.locator('#hypothesesOverlay').evaluate(e=>{e.checked=true;e.dispatchEvent(new Event('change'));});
 assert((await p.evaluate(()=>window.liveInspection.getState())).hypothesesOverlay);await p.screenshot({path:path.join(out,'live-renderer-offline-fixture.png')});
 overlay.live_default_hidden=true;overlay.live_inspect_toggle=true;overlay.live_unresolved_pedestrian_visible=true;
 await c.close();assert.deepEqual(errors,[]);
 fs.writeFileSync(path.join(out,'BROWSER_CHECK.json'),JSON.stringify({status:'PASS',browser:browser.version(),conditions:'1x actual rAF; 45 m heading-up; short comparison recording at 2560x1000, standalone timing at 1440x1000. Offline live-renderer injection only; no running capture.',marks,clock,perf,overlay,page_errors:errors},null,2)+'\n');
 console.log(JSON.stringify({status:'PASS',perf,overlay,errors},null,2));await browser.close();
})().catch(e=>{console.error(e);process.exit(1);});
