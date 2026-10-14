/* v20.2 user-facing check: normal-speed v20.1 | v20.2 recording, screenshots, default-view clutter
 * checks, paused same-time pedestrian equivalence, and the live renderer on injected recorded frames.
 * Offline only: no capture, no live pipeline. Usage (offline server running on REPLAY_URL):
 *   PLAYWRIGHT_MODULE=... PLAYWRIGHT_BROWSERS_PATH=... node browser_orientation_v20_2.cjs [--checks-only] */
'use strict';
const fs=require('fs'),path=require('path'),assert=require('assert/strict');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const base=process.env.REPLAY_URL||'http://127.0.0.1:8777',out=path.join(__dirname,'artifacts_v20_2_orientation');
fs.mkdirSync(out,{recursive:true});
const summary=a=>{const s=a.slice().sort((a,b)=>a-b);return {n:s.length,p50:s[Math.floor(s.length*.5)]??null,p95:s[Math.floor(s.length*.95)]??null,max:s.at(-1)??null};};
// Count canvas text that names hypotheses/conflicts (the normal view must draw none).
const LABELS=()=>{window.diagLabels=0;const f=CanvasRenderingContext2D.prototype.fillText;
  CanvasRenderingContext2D.prototype.fillText=function(t,...a){if(/hypothes/i.test(String(t)))window.diagLabels++;return f.call(this,t,...a);};};
const SEGMENTS=[['parked',3000,9000],['parked_avenue',0,10000],['turn',3000,12000],['pedestrians',0,10000]];
(async()=>{
 const browser=await chromium.launch({headless:true}),errors=[],result={segments:[],clutter:{},pedestrians:{},live:{}};
 if(!process.argv.includes('--checks-only')){
  const ctx=await browser.newContext({viewport:{width:2560,height:1000},recordVideo:{dir:out,size:{width:2560,height:1000}}});
  await ctx.addInitScript(LABELS);
  const page=await ctx.newPage();page.on('pageerror',e=>errors.push('compare: '+e.message));
  await page.goto(base+'/compare-orientation');
  await page.waitForFunction(()=>['before','after'].every(id=>document.getElementById(id).contentWindow.replayInspection?.getState().eventCount>0),null,{timeout:60000});
  const each=(src,arg)=>page.evaluate(([src,arg])=>['before','after'].map(id=>new Function('w','arg',src)(document.getElementById(id).contentWindow,arg)),[src,arg]);
  const start=Date.now();
  for(const [clip,from,duration] of SEGMENTS){
   await each("w.document.getElementById('clip').value=arg;w.document.getElementById('clip').dispatchEvent(new Event('change'));",clip);
   await page.waitForFunction(clip=>['before','after'].every(id=>{const s=document.getElementById(id).contentWindow.replayInspection.getState();return s.clip===clip&&!s.loading;}),clip);
   await each('w.replayInspection.seek(arg);',from);await page.waitForTimeout(250);
   const mark={clip,clip_start_ms:from,video_s:(Date.now()-start)/1000,duration_ms:duration};
   await each("w.document.getElementById('play').click();");
   // Sample the user-facing state of both panes at display rate while it plays.
   mark.samples=await page.evaluate(async ms=>{const acc={before:{veh:0,neutral:0,labels:0},after:{veh:0,neutral:0,labels:0},clockDiff:[]},end=performance.now()+ms;
    while(performance.now()<end){await new Promise(requestAnimationFrame);const s={};
     for(const id of ['before','after']){const w=document.getElementById(id).contentWindow,st=w.replayInspection.getState();s[id]=st;
      for(const r of st.view?.shown||[])if(r.cls==='Vehicle'){acc[id].veh++;const y=r.displayYaw!==undefined?r.displayYaw:r.yaw;if(y===null)acc[id].neutral++;}
      acc[id].labels=w.diagLabels;}
     acc.clockDiff.push(Math.abs(s.before.now-s.after.now));}
    return acc;},duration);
   mark.samples.clockDiff=summary(mark.samples.clockDiff);
   await each("w.document.getElementById('play').click();");
   await page.screenshot({path:path.join(out,clip+'-v20.1-vs-v20.2.png')});
   result.segments.push(mark);
  }
  await ctx.close();fs.copyFileSync(await page.video().path(),path.join(out,'v20.1-vs-v20.2-1x.webm'));fs.unlinkSync(await page.video().path());
  for(const m of result.segments)assert.equal(m.samples.after.labels,0,'v20.2 default view drew a hypothesis/conflict label in '+m.clip);
 }
 // Default vs Inspect presentation on the current replay page.
 {const c=await browser.newContext({viewport:{width:1440,height:1000}});await c.addInitScript(LABELS);const p=await c.newPage();p.on('pageerror',e=>errors.push('replay: '+e.message));
  await p.goto(base+'/');await p.waitForFunction(()=>window.replayInspection?.getState().eventCount>0);
  await p.selectOption('#clip','parked');await p.waitForFunction(()=>{const s=window.replayInspection.getState();return s.clip==='parked'&&!s.loading;});
  await p.evaluate(()=>window.replayInspection.seek(3000));await p.click('#play');await p.waitForTimeout(7000);await p.click('#play');
  const vis=sel=>p.locator(sel).first().isVisible();
  const d={labels:await p.evaluate(()=>window.diagLabels),unresolvedTile:await vis('#nHyp'),rawLine:await vis('#rawLine'),hypLegend:await vis('.legend li.diag'),
   neutralLegend:await vis('.legend .g-neu'),state:await p.evaluate(()=>window.replayInspection.getState())};
  assert.equal(d.labels,0);assert.equal(d.unresolvedTile,false);assert.equal(d.rawLine,false);assert.equal(d.hypLegend,false);assert.equal(d.neutralLegend,true);
  const ped0=d.state.view.shown.filter(r=>r.cls==='Pedestrian').map(r=>r.key).sort();
  await p.screenshot({path:path.join(out,'default-view.png')});
  await p.click('#inspect');await p.check('#hypothesesOverlay');await p.waitForTimeout(400);
  const i={labels:await p.evaluate(()=>window.diagLabels),unresolvedTile:await vis('#nHyp'),rawLine:await vis('#rawLine'),hypLegend:await vis('.legend li.diag'),state:await p.evaluate(()=>window.replayInspection.getState())};
  assert(i.labels>0,'Inspect restores hypothesis labels');assert(i.unresolvedTile&&i.rawLine&&i.hypLegend);
  assert.deepEqual(i.state.view.shown.filter(r=>r.cls==='Pedestrian').map(r=>r.key).sort(),ped0,'Inspect does not change pedestrian presence');
  await p.locator('#rows tr').first().click();const detail=JSON.parse(await p.locator('#detail').innerText());
  await p.screenshot({path:path.join(out,'inspect-view.png'),fullPage:true});
  result.clutter={default:{hypothesis_or_conflict_canvas_labels:d.labels,low_confidence_tile_visible:d.unresolvedTile,raw_candidate_line_visible:d.rawLine,hypothesis_legend_visible:d.hypLegend,neutral_vehicle_legend_visible:d.neutralLegend},
   inspect:{canvas_labels:i.labels,low_confidence_tile_visible:i.unresolvedTile,raw_candidate_line_visible:i.rawLine,hypothesis_legend_visible:i.hypLegend,detail_has_display_orientation:'display_orientation' in (detail.display_track||{display_orientation:null})}};
  await c.close();}
 // Paused, same-time pedestrian equivalence between v20.1 and v20.2 panes (pose, motion, tier, presence).
 {const c=await browser.newContext({viewport:{width:2560,height:1000}}),p=await c.newPage();p.on('pageerror',e=>errors.push('ped: '+e.message));
  await p.goto(base+'/compare-orientation');
  await p.waitForFunction(()=>['before','after'].every(id=>document.getElementById(id).contentWindow.replayInspection?.getState().eventCount>0),null,{timeout:60000});
  let compared=0,peds=0,unresolved=0;
  for(const clip of ['pedestrians','parked']){
   await p.evaluate(clip=>['before','after'].forEach(id=>{const d=document.getElementById(id).contentWindow.document;d.getElementById('clip').value=clip;d.getElementById('clip').dispatchEvent(new Event('change'));}),clip);
   await p.waitForFunction(clip=>['before','after'].every(id=>{const s=document.getElementById(id).contentWindow.replayInspection.getState();return s.clip===clip&&!s.loading;}),clip);
   for(let t=0;t<=12000;t+=250){
    const rows=await p.evaluate(t=>Promise.all(['before','after'].map(id=>{const w=document.getElementById(id).contentWindow;w.replayInspection.seek(t);
      // seek() rebuilds causally; drive one render so the inspection view reflects time t.
      return new Promise(r=>w.requestAnimationFrame(()=>w.requestAnimationFrame(()=>r(w.replayInspection.getState().view.shown.filter(x=>x.cls==='Pedestrian').map(x=>({key:x.key,x:x.x,y:x.y,motion:x.motion,tier:x.tier,alpha:x.alpha}))))));})),t);
    const [a,b]=rows;assert.deepEqual(b,a,`pedestrians differ at ${clip} ${t} ms`);compared++;peds+=a.length;unresolved+=a.filter(x=>x.motion==='unresolved').length;}
  }
  result.pedestrians={paused_times_compared:compared,pedestrian_rows:peds,unresolved_rows:unresolved,identical:true};await c.close();}
 // Actual live renderer, recorded frames injected locally (no running pipeline).
 {const c=await browser.newContext({viewport:{width:1440,height:1000}});await c.addInitScript(LABELS);const p=await c.newPage();p.on('pageerror',e=>errors.push('live: '+e.message));
  await p.route(base+'/live',r=>r.fulfill({contentType:'text/html',body:fs.readFileSync(path.join(__dirname,'../viewer_assets/live_v1.html'),'utf8')}));
  await p.route(base+'/api/live/**',r=>r.fulfill({status:200,contentType:r.request().url().endsWith('/stream')?'text/event-stream':'application/json',body:r.request().url().endsWith('/stream')?'':JSON.stringify({epoch:'offline-fixture',label:'OFFLINE V20 RENDERER CHECK'})}));
  await p.goto(base+'/live');await p.waitForFunction(()=>window.liveInspection&&window.ScenesenseBuildings);await p.waitForTimeout(500);
  const live=await p.evaluate(async()=>{
   const clip=await (await fetch('/api/replay/clip/parked')).json(),ev=clip.events.filter(e=>e.arrival_ms<=8000).slice(-40),base=BigInt(clip.base_ns),epoch='offline-fixture';
   const send=m=>window.liveInspection._onMessage(JSON.stringify(m)),ns=t=>String(base+BigInt(Math.round(t*1e6)));
   send({type:'hello',epoch,session:ev[0].session,base_ns:String(base),server_ns:ns(ev[0].arrival_ms)});
   ev.forEach((e,i)=>send({...e,type:'event',seq:i+1,epoch,server_ns:ns(e.arrival_ms)}));
   await new Promise(r=>setTimeout(r,250));return window.liveInspection.getState();});
  const vis=sel=>p.locator(sel).first().isVisible();
  assert.equal(live.accepted,40);assert.equal(await p.evaluate(()=>window.diagLabels),0);assert.equal(await vis('#nHyp'),false);assert.equal(await vis('#rawLine'),false);
  const veh=live.view.shown.filter(r=>r.cls==='Vehicle');assert(veh.every(r=>r.orientation==='oriented'||r.orientation==='neutral'));
  await p.screenshot({path:path.join(out,'live-renderer-offline-fixture.png')});
  await p.locator('#inspect').evaluate(e=>e.click());await p.waitForTimeout(250);assert.equal(await vis('#nHyp'),true);
  result.live={injected_recorded_frames:40,default_canvas_labels:0,low_confidence_tile_hidden:true,inspect_shows_tile:true,vehicles_shown:veh.length,neutral:veh.filter(r=>r.orientation==='neutral').length,
   pedestrians_shown:live.view.shown.filter(r=>r.cls==='Pedestrian').length};
  await c.close();}
 assert.deepEqual(errors,[]);
 const report={status:'PASS',browser:browser.version(),conditions:'1x real rAF, 45 m heading-up, compare page 2560x1000 recorded; checks 1440x1000. Offline replay of sealed V20 taps; live renderer fed recorded frames locally. No capture or live pipeline.',...result,page_errors:errors};
 fs.writeFileSync(path.join(out,'BROWSER_CHECK.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,1));await browser.close();
})().catch(e=>{console.error(e);process.exit(1);});
