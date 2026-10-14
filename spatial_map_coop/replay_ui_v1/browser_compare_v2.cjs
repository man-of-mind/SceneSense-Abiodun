/* Actual Chromium 1x side-by-side playback, then standalone performance. */
const fs=require('fs'),path=require('path'),assert=require('assert/strict');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const base=process.env.REPLAY_URL||'http://127.0.0.1:8765',out=path.join(__dirname,'artifacts_v2');
const summary=a=>{const s=a.slice().sort((a,b)=>a-b);return {n:s.length,mean:s.reduce((a,b)=>a+b,0)/(s.length||1),p50:s[Math.floor(s.length*.5)]||0,p95:s[Math.floor(s.length*.95)]||0,max:s.at(-1)||0};};
(async()=>{fs.mkdirSync(out,{recursive:true});const browser=await chromium.launch({headless:true});
 const context=await browser.newContext({viewport:{width:2880,height:1240},deviceScaleFactor:1,recordVideo:{dir:out,size:{width:2880,height:1240}}});
 const page=await context.newPage(),errors=[],requests=[];page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>requests.push(r.url()));
 await page.goto(base+'/compare');
 await page.waitForFunction(()=>['before','after'].every(id=>document.getElementById(id).contentWindow.replayInspection?.getState().eventCount>0));
 const frames=()=>['before','after'].map(id=>page.frameLocator('#'+id));
 async function both(selector,value){for(const f of frames())await f.locator(selector).selectOption(value);}
 async function waitClip(frame){await page.waitForFunction(frame=>['before','after'].every(id=>document.getElementById(id).contentDocument.getElementById('cameraFrame').textContent.includes(frame)),String(frame));}
 async function seek(t){await page.evaluate(t=>{for(const id of ['before','after']){const w=document.getElementById(id).contentWindow,e=w.document.getElementById('seek');if(w.replayInspection.getState().playing)w.document.getElementById('play').click();e.value=t;e.dispatchEvent(new Event('input'));}},t);await page.waitForTimeout(200);}
 async function play(){await page.evaluate(()=>{for(const id of ['before','after'])document.getElementById(id).contentDocument.getElementById('play').click();});}
 async function sample(duration){return page.evaluate(async duration=>{const clipName=document.getElementById('before').contentDocument.getElementById('clip').value;const clip=await (await fetch('/api/replay/clip/'+clipName)).json();const counts=new Map(clip.events.map(e=>[e.frame_id,e.objects.length]));const samples=[],end=performance.now()+duration;while(performance.now()<end){await new Promise(requestAnimationFrame);const pair=['before','after'].map(id=>{const s=document.getElementById(id).contentWindow.replayInspection.getState();return {now:s.now,frame:s.view?.frame,ego:s.view?.ego,shown:s.view?.shown,raw:s.view?.rawCount??counts.get(s.view?.frame)??0,rows:s.view?.rows};});samples.push(pair);}return samples;},duration);}
 const checks=[],sections={};
 await both('#cameraFrame','1480');await page.waitForTimeout(600);await page.screenshot({path:path.join(out,'01-departure-before-after.png')});
 await play();sections.departure=await sample(4000);
 await both('#clip','turn');await waitClip(2020);await seek(0);await play();sections.turn=await sample(12000);
 await both('#cameraFrame','2020');await page.waitForTimeout(500);await page.screenshot({path:path.join(out,'02-turn-before-after.png')});
 await both('#clip','stop');await waitClip(2748);await both('#cameraFrame','2748');await page.waitForTimeout(500);await page.screenshot({path:path.join(out,'03-dense-before-after.png')});
 await play();sections.stop=await sample(6000);
 await both('#policy','coast');await seek(17400);await play();sections.interruption=await sample(2600);
 await page.screenshot({path:path.join(out,'04-interrupted-before-after.png')});
 const final=await page.evaluate(()=>['before','after'].map(id=>document.getElementById(id).contentWindow.replayInspection.getState()));
 assert(final.every(s=>s.view.shown.length===0),'no-message tail expires on both sides');
 assert(sections.interruption.some(pair=>pair[1].shown.some(o=>o.alpha>0&&o.alpha<1)),'coast fades');
 for(const [name,samples]of Object.entries(sections)){
  const sides=[0,1].map(side=>{let previous=new Map();const steps=[],secondDiff=[],oldDelta=new Map();let sum=0,raw=0,consolidated=0,occluded=0,ambiguous=0,disappeared=0,appeared=0,egoMovement=0,oldEgo=null;
   for(const pair of samples){const s=pair[side],current=new Map();sum+=s.shown.length;raw+=s.raw||0;
    consolidated+=(s.rows||[]).filter(r=>r.reason==='CONSOLIDATED_HYPOTHESIS').length;occluded+=(s.rows||[]).filter(r=>r.reason==='STATIC_MAP_OCCLUDED').length;ambiguous+=s.shown.filter(r=>r.ambiguous).length;
    if(oldEgo&&s.ego)egoMovement+=Math.hypot(s.ego.x-oldEgo.x,s.ego.y-oldEgo.y);oldEgo=s.ego;
    for(const o of s.shown){current.set(o.key,o);const p=previous.get(o.key);if(p){const d=[o.x-p.x,o.y-p.y];steps.push(Math.hypot(...d));const old=oldDelta.get(o.key);if(old)secondDiff.push(Math.hypot(d[0]-old[0],d[1]-old[1]));oldDelta.set(o.key,d);}else appeared++;}
    for(const key of previous.keys())if(!current.has(key))disappeared++;previous=current;
   }return {mean_shown:sum/samples.length,mean_raw_current:raw/samples.length,mean_consolidated:consolidated/samples.length,mean_building_withheld:occluded/samples.length,mean_uncertainty_diamonds:ambiguous/samples.length,step_m:summary(steps),second_difference_m:summary(secondDiff),appearances:appeared,disappearances:disappeared,ego_distance_m:egoMovement};});
  const clock=summary(samples.map(s=>Math.abs(s[0].now-s[1].now)));assert(clock.max<35,'comparison clocks differ');checks.push({name,samples:samples.length,frame_interval_ms:summary(samples.slice(1).map((s,i)=>s[0].now-samples[i][0].now).filter(x=>x>0)),clock_difference_ms:clock,before:sides[0],after:sides[1]});
 }
 await context.close();const video=await page.video().path();fs.renameSync(video,path.join(out,'before-after-1x.webm'));
 // Standalone full-size run, no concurrent video: same renderer at 1440x1100.
 const single=await browser.newContext({viewport:{width:1440,height:1100}}),p=await single.newPage();p.on('pageerror',e=>errors.push(e.message));p.on('request',r=>requests.push(r.url()));
 await p.goto(base);await p.waitForFunction(()=>window.replayInspection?.getState().eventCount>0);
 await p.selectOption('#clip','turn');await p.waitForFunction(()=>document.getElementById('cameraFrame').textContent.includes('2020'));
 await p.click('#play');await p.waitForTimeout(11000);await p.click('#play');const perf=await p.evaluate(()=>window.replayInspection.getState());
 await p.selectOption('#cameraFrame','2020');await p.waitForTimeout(300);const reference=await p.evaluate(()=>window.replayInspection.getState().view.shown);await p.screenshot({path:path.join(out,'05-improved-ui.png'),fullPage:true});
 await p.locator('#seek').evaluate(e=>{e.value=1000;e.dispatchEvent(new Event('input'));});await p.selectOption('#cameraFrame','2020');await p.waitForTimeout(250);assert.deepEqual(await p.evaluate(()=>window.replayInspection.getState().view.shown),reference,'deterministic seek');
 await p.selectOption('#policy','visible');await p.waitForTimeout(100);assert.equal((await p.evaluate(()=>window.replayInspection.getState())).view.shown.length,0,'static approximation cannot qualify visibility');
 await p.click('#debug');await p.waitForTimeout(200);await p.locator('#rows tr').first().click();assert((await p.locator('#detail').innerText()).includes('measurement'));await p.screenshot({path:path.join(out,'06-raw-provenance.png'),fullPage:true});
 await p.setViewportSize({width:760,height:1100});await p.waitForTimeout(250);assert(await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'narrow layout');
 assert.equal(errors.length,0);assert(requests.every(u=>u.startsWith(base)));assert(perf.trackCount<=256);
 const report={status:'PASS',browser:browser.version(),conditions:'Actual rAF 1x replay; side-by-side video at 2880x1240. Standalone timing at 1440x1100, no video. Software rendering; host-specific, not live qualification.',comparisons:checks,
  standalone:{frame_ms:summary(perf.intervals),render_cpu_ms:summary(perf.renderCosts),churn:perf.churn,motion:{...perf.metrics,corrections:summary(perf.metrics.corrections),innovations:summary(perf.metrics.innovations),filterOffsets:summary(perf.metrics.filterOffsets)}},
  added_input_buffer_ms:0,continuity_bridge_ms:100,expiry_ms:750,horizon_ms:500,strict_visibility_unchanged:true,raw_access:true,seek_deterministic:true,page_errors:errors,external_requests:0};
 fs.writeFileSync(path.join(out,'BROWSER_CHECK.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));await single.close();await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
