/* Actual Chromium 1x side-by-side playback (v2 | v3), screenshots, standalone render cost.
 * Usage (server running):
 *   PLAYWRIGHT_MODULE=/path/to/playwright-core CHROMIUM=/path/to/headless_shell node browser_compare_v3.cjs
 * Development tooling only; not an application dependency. */
'use strict';
const fs=require('fs'),path=require('path'),assert=require('assert/strict');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const base=process.env.REPLAY_URL||'http://127.0.0.1:8765',out=path.join(__dirname,'artifacts_v3');
const summary=a=>{const s=a.slice().sort((x,y)=>x-y);return {n:s.length,mean:s.reduce((x,y)=>x+y,0)/(s.length||1),p50:s[Math.floor(s.length*.5)]||0,p95:s[Math.floor(s.length*.95)]||0,max:s.at(-1)||0};};
const launch=()=>chromium.launch({headless:true,...(process.env.CHROMIUM?{executablePath:process.env.CHROMIUM}:{})});
// Same clips, same 1x speed, same 45 m radius on both sides.
const SEGMENTS=[['departure',0,4500],['turn',8000,6500],['stop',2000,6000],['turn',17400,1700]];
(async()=>{fs.mkdirSync(out,{recursive:true});const browser=await launch(),errors=[],requests=[];
 const context=await browser.newContext({viewport:{width:2880,height:1240},recordVideo:{dir:out,size:{width:2880,height:1240}}});
 const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>requests.push(r.url()));
 await page.goto(base+'/compare-v3');
 await page.waitForFunction(()=>['before','after'].every(id=>document.getElementById(id).contentWindow.replayInspection?.getState().eventCount>0),null,{timeout:60000});
 const each=(fn,arg)=>page.evaluate(([src,arg])=>{const f=new Function('w','arg',src);return ['before','after'].map(id=>f(document.getElementById(id).contentWindow,arg));},[fn,arg]);
 async function load(clip){await each("w.document.getElementById('clip').value=arg;w.document.getElementById('clip').dispatchEvent(new Event('change'));",clip);
  await page.waitForFunction(clip=>['before','after'].every(id=>{const w=document.getElementById(id).contentWindow,d=w.document;
    return d.getElementById('clip').value===clip&&w.replayInspection.getState().eventCount>0&&!(w.replayInspection.getState().loading)&&(id==='before'?d.getElementById('cameraFrame').options.length>1:w.replayInspection.getState().clip===clip);}),clip,{timeout:60000});
  await page.waitForTimeout(700);}
 async function seek(t){await each("if(w.replayInspection.getState().playing)w.document.getElementById('play').click();const e=w.document.getElementById('seek');e.value=arg;e.dispatchEvent(new Event('input'));",t);await page.waitForTimeout(250);}
 const play=()=>each("w.document.getElementById('play').click();");
 async function sample(ms){return page.evaluate(async ms=>{const out=[],end=performance.now()+ms;while(performance.now()<end){await new Promise(requestAnimationFrame);
   out.push(['before','after'].map(id=>{const s=document.getElementById(id).contentWindow.replayInspection.getState();return {now:s.now,keys:(s.view?.shown||[]).map(o=>o.key),tiers:(s.view?.shown||[]).map(o=>o.tier||'glyph')};}));}return out;},ms);}
 const sections=[],marks=[];const t0=Date.now();let shot=1;
 for(const [clip,start,dur] of SEGMENTS){await load(clip);await seek(start);marks.push({clip,start_ms:start,duration_ms:dur,video_s:(Date.now()-t0)/1000});await play();
  const s=await sample(dur);await play();sections.push({clip,start,dur,s});
  await page.screenshot({path:path.join(out,String(shot++).padStart(2,'0')+`-${clip}-${start}-v2-v3.png`)});}
 const final=await page.evaluate(()=>document.getElementById('after').contentWindow.replayInspection.getState());
 assert(final.view.shown.length===0&&final.view.egoStale,'v3 tail expires');
 await context.close();const video=await page.video().path();fs.renameSync(video,path.join(out,'v2-vs-v3-1x.webm'));
 const comparisons=sections.map(({clip,start,dur,s})=>{const side=i=>{let prev=new Set(),appear=0,vanish=0,sum=0;
   for(const pair of s){const cur=new Set(pair[i].keys);sum+=cur.size;for(const k of cur)if(!prev.has(k))appear++;for(const k of prev)if(!cur.has(k))vanish++;prev=cur;}
   return {mean_glyphs:sum/s.length,appearances_per_s:appear/(dur/1000),disappearances_per_s:vanish/(dur/1000)};};
  const clock=summary(s.map(p=>Math.abs(p[0].now-p[1].now)));return {clip,start_ms:start,duration_ms:dur,samples:s.length,clock_difference_ms:clock,v2:side(0),v3:side(1)};});
 // Standalone: same renderer conditions for v2 and v3, no video.
 const standalone={};
 for(const [name,url] of [['v2','/v2'],['v3','/']]){const ctx=await browser.newContext({viewport:{width:1440,height:1000}}),p=await ctx.newPage();p.on('pageerror',e=>errors.push(name+': '+e.message));p.on('request',r=>requests.push(r.url()));
  await p.goto(base+url);await p.waitForFunction(()=>window.replayInspection?.getState().eventCount>0,null,{timeout:60000});
  await p.selectOption('#clip','turn');await p.waitForTimeout(2500);await p.locator('#seek').evaluate(e=>{e.value=6000;e.dispatchEvent(new Event('input'));});
  await p.click('#play');await p.waitForTimeout(8000);await p.click('#play');const st=await p.evaluate(()=>window.replayInspection.getState());
  standalone[name]={frame_ms:summary(st.intervals.slice(-400)),render_cpu_ms:summary(st.renderCosts.slice(-400))};
  if(name==='v3'){await p.screenshot({path:path.join(out,'05-v3-ego-traffic-view.png')});
   await p.click('#inspect');await p.check('#rawOverlay');await p.check('#ids');await p.waitForTimeout(400);await p.locator('#rows tr').first().click();await p.waitForTimeout(250);
   assert((await p.locator('#detail').innerText()).includes('measurement_unchanged'));await p.screenshot({path:path.join(out,'06-v3-inspect-raw-dispositions.png'),fullPage:true});
   await p.selectOption('#policy','visible');await p.waitForTimeout(300);assert.equal((await p.evaluate(()=>window.replayInspection.getState())).view.shown.length,0,'strict visible withholds all');
   await p.setViewportSize({width:760,height:1100});await p.waitForTimeout(300);assert(await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'narrow layout');}
  await ctx.close();}
 assert.equal(errors.length,0,errors.join('; '));assert(requests.every(u=>u.startsWith(base)),'no external requests');
 const report={status:'PASS',browser:browser.version(),conditions:'Actual rAF 1x playback, both sides 45 m radius, heading-up, default views. Side-by-side 2880x1240 with video; standalone 1440x1000 without video. Software rendering, host-specific.',
  video_marks:marks,comparisons,standalone,added_input_buffer_ms:0,v3_correction_blend_ms:150,page_errors:errors,external_requests:0};
 fs.writeFileSync(path.join(out,'BROWSER_CHECK_V3.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({comparisons,standalone},null,1));await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
