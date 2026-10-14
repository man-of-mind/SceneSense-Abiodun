/* Normal-speed side-by-side: pedestrian motion v3.0 (left) vs v3.1 (right), same clips, 1x, 45 m.
 * Usage (server running): PLAYWRIGHT_MODULE=... CHROMIUM=... node browser_compare_ped_v3.cjs   (dev tooling only) */
'use strict';
const fs=require('fs'),path=require('path'),assert=require('assert/strict');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const base=process.env.REPLAY_URL||'http://127.0.0.1:8765',out=path.join(__dirname,'artifacts_v3_ped');
const SEGMENTS=[['ped_pause',6500,6500],['ped_standing',7000,6000],['stop',2000,5000]];
(async()=>{fs.mkdirSync(out,{recursive:true});const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM?{executablePath:process.env.CHROMIUM}:{})}),errors=[],requests=[];
 const context=await browser.newContext({viewport:{width:2880,height:1240},recordVideo:{dir:out,size:{width:2880,height:1240}}});
 const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>requests.push(r.url()));
 await page.goto(base+'/compare-ped');
 await page.waitForFunction(()=>['before','after'].every(id=>document.getElementById(id).contentWindow.replayInspection?.getState().eventCount>0),null,{timeout:60000});
 const pol=await page.evaluate(()=>['before','after'].map(id=>document.getElementById(id).contentWindow.replayInspection.getState().pedestrianMotion));assert.deepEqual(pol,['v3.0','v3.1']);
 const each=(src,arg)=>page.evaluate(([src,arg])=>{const f=new Function('w','arg',src);return ['before','after'].map(id=>f(document.getElementById(id).contentWindow,arg));},[src,arg]);
 const marks=[],sections=[];const t0=Date.now();let shot=1;
 for(const [clip,start,dur] of SEGMENTS){await each("const s=w.document.getElementById('clip');s.value=arg;s.dispatchEvent(new Event('change'));",clip);
  await page.waitForFunction(c=>['before','after'].every(id=>{const s=document.getElementById(id).contentWindow.replayInspection.getState();return s.clip===c&&!s.loading;}),clip,{timeout:60000});await page.waitForTimeout(700);
  await each("const e=w.document.getElementById('seek');e.value=arg;e.dispatchEvent(new Event('input'));",start);await page.waitForTimeout(250);
  marks.push({clip,start_ms:start,duration_ms:dur,video_s:(Date.now()-t0)/1000});await each("w.document.getElementById('play').click();");
  const samples=await page.evaluate(async ms=>{const o=[],end=performance.now()+ms;while(performance.now()<end){await new Promise(requestAnimationFrame);
    o.push(['before','after'].map(id=>{const s=document.getElementById(id).contentWindow.replayInspection.getState();return {now:s.now,peds:(s.view?.shown||[]).filter(r=>r.cls==='Pedestrian').map(r=>({key:r.key,motion:r.motion}))};}));}return o;},dur);
  await each("w.document.getElementById('play').click();");sections.push({clip,start,dur,samples});
  await page.screenshot({path:path.join(out,String(shot++).padStart(2,'0')+`-${clip}-${start}-v3.0-v3.1.png`)});}
 await context.close();fs.renameSync(await page.video().path(),path.join(out,'pedestrian-v3.0-vs-v3.1-1x.webm'));
 const summary=sections.map(({clip,start,dur,samples})=>{const side=i=>{const n=samples.length,peds=samples.reduce((a,s)=>a+s[i].peds.length,0),m={};
   for(const s of samples)for(const p of s[i].peds)m[p.motion]=(m[p.motion]||0)+1;const keys=new Set(samples.flatMap(s=>s[i].peds.map(p=>p.key)));
   return {mean_pedestrians_shown:peds/n,motion_state_share:Object.fromEntries(Object.entries(m).map(([k,v])=>[k,v/Math.max(1,peds)])),distinct_pedestrian_display_ids:keys.size};};
  return {clip,start_ms:start,duration_ms:dur,samples:samples.length,clock_diff_max_ms:Math.max(...samples.map(s=>Math.abs(s[0].now-s[1].now))),v3_0:side(0),v3_1:side(1)};});
 assert.equal(errors.length,0,errors.join(';'));assert(requests.every(u=>u.startsWith(base)));
 const report={status:'PASS',browser:browser.version(),conditions:'Actual rAF 1x, both sides 45 m heading-up default view; left ?ped=v3.0, right v3.1',video_marks:marks,summary,page_errors:errors,external_requests:0};
 fs.writeFileSync(path.join(out,'BROWSER_CHECK_PED.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report.summary,null,1));await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
