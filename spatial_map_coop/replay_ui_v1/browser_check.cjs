/* Real Chromium inspection, with real-time rAF playback and UI input events.
   PLAYWRIGHT_MODULE=/path/to/playwright PLAYWRIGHT_BROWSERS_PATH=/path node ... */
const fs=require('fs'),path=require('path'),assert=require('assert/strict');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const out=path.resolve(process.env.REPLAY_ARTIFACTS||path.join(__dirname,'artifacts'));
const base=process.env.REPLAY_URL||'http://127.0.0.1:8765';
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
(async()=>{
 fs.mkdirSync(out,{recursive:true});
 const browser=await chromium.launch({headless:true});
 const context=await browser.newContext({viewport:{width:1440,height:1100},deviceScaleFactor:1,
   recordVideo:{dir:out,size:{width:1440,height:1100}}});
 const page=await context.newPage(),errors=[],requests=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>requests.push(r.url()));
 const state=()=>page.evaluate(()=>window.replayInspection.getState());
 async function seek(ms){await page.locator('#seek').evaluate((e,t)=>{e.value=t;e.dispatchEvent(new Event('input',{bubbles:true}));},ms);await page.waitForTimeout(180);}
 await page.goto(base);await page.waitForFunction(()=>window.replayInspection?.getState().eventCount>0);
 const initial=await state();assert(initial.eventCount<=256);assert.equal(initial.playing,false);
 await page.selectOption('#cameraFrame','1480');await page.locator('#camera').waitFor({state:'visible'});
 await page.waitForFunction(()=>document.getElementById('camera').complete);
 await page.waitForTimeout(500);await page.screenshot({path:path.join(out,'01-forward.png'),fullPage:true});
 await page.locator('#rows tr').first().click();assert((await page.locator('#detail').innerText()).includes('measurement'));
 await page.click('#debug');await page.waitForTimeout(220);assert((await state()).debug);
 await page.screenshot({path:path.join(out,'02-raw-audit.png'),fullPage:true});
 await page.click('#debug');await page.selectOption('#policy','visible');await page.waitForTimeout(220);
 assert.equal((await state()).view.shown.length,0);assert((await page.locator('#mapNotice').innerText()).includes('No qualified'));
 await page.screenshot({path:path.join(out,'03-visibility-limitation.png'),fullPage:true});
 await page.selectOption('#policy','current');await page.click('#north');assert.equal((await state()).heading,false);
 await page.locator('#radius').fill('60');await page.locator('#radius').dispatchEvent('input');assert.equal((await state()).radius,60);
 await page.click('#heading');await page.locator('#radius').fill('45');await page.locator('#radius').dispatchEvent('input');
 // Play the actual recorded junction sequence, continuously. Sample the rendered
 // scene on every rAF to check independent motion between install events.
 await page.selectOption('#clip','turn');await page.waitForFunction(()=>document.getElementById('cameraFrame').textContent.includes('2020'));
 await page.click('#play');
 const motion=await page.evaluate(async()=>{
   const samples=[];const until=performance.now()+12500;
   while(performance.now()<until){await new Promise(requestAnimationFrame);const s=window.replayInspection.getState();
     samples.push({t:s.now,frame:s.view?.frame,ego:s.view?.ego,shown:s.view?.shown});}
   return samples;
 });
 await page.click('#play');const paused=await state();await page.waitForTimeout(300);assert.equal((await state()).now,paused.now);
 let betweenUpdates=0,headingDelta=0,continuitySamples=0,maxStep=0;
 for(let i=1;i<motion.length;i++){
   const a=motion[i-1],b=motion[i];if(!a.ego||!b.ego)continue;
   headingDelta+=Math.abs(((b.ego.yaw-a.ego.yaw+540)%360)-180);
   if(a.frame===b.frame&&Math.hypot(b.ego.x-a.ego.x,b.ego.y-a.ego.y)>1e-5)betweenUpdates++;
   const old=new Map((a.shown||[]).map(o=>[o.key,o]));
   for(const o of b.shown||[]){const before=old.get(o.key);if(before){maxStep=Math.max(maxStep,Math.hypot(o.x-before.x,o.y-before.y));continuitySamples++;}}
 }
 assert(betweenUpdates>50,'No independent smooth motion between arrivals');assert(headingDelta>10,'Recorded turn did not play');
 assert(maxStep<1.2,'Large within-track correction sweep; inspect motion before handoff');
 await page.selectOption('#cameraFrame','2020');await page.waitForTimeout(500);
 await page.screenshot({path:path.join(out,'04-junction.png'),fullPage:true});
 const reference=await state();await seek(1000);await page.selectOption('#cameraFrame','2020');await page.waitForTimeout(250);
 const rebuilt=await state();assert.deepEqual(rebuilt.view.shown,reference.view.shown,'seek must rebuild causal state deterministically');
 // Final interruption: actual no-message tail, watch coasting fade then disappear.
 await page.selectOption('#policy','coast');await seek(17400);await page.click('#play');
 const fade=await page.evaluate(async()=>{const samples=[];for(let i=0;i<150;i++){
   await new Promise(requestAnimationFrame);const s=window.replayInspection.getState();samples.push({t:s.now,shown:s.view.shown});}return samples;});
 const faded=fade.some(s=>s.shown.some(o=>o.alpha>0&&o.alpha<1));assert(faded,'No observed fading');
 assert.equal((await state()).view.shown.length,0,'Interrupted stream must expire');
 await page.selectOption('#clip','stop');await page.waitForFunction(()=>document.getElementById('cameraFrame').textContent.includes('2748'));
 await page.selectOption('#cameraFrame','2748');await page.waitForTimeout(400);
 await page.screenshot({path:path.join(out,'05-stopped-ego.png'),fullPage:true});
 // Separate warmed, moving run for performance, after control exercises.
 await page.selectOption('#clip','avenue');await page.waitForFunction(()=>document.getElementById('cameraFrame').textContent.includes('4510'));
 await page.click('#play');await page.waitForTimeout(11000);
 const measured=await state();await page.click('#play');
 const summary=a=>{const s=a.slice().sort((a,b)=>a-b);return {n:a.length,mean:a.reduce((s,x)=>s+x,0)/a.length,p50:s[Math.ceil(s.length*.5)-1],p95:s[Math.ceil(s.length*.95)-1],max:s[s.length-1]};};
 const frame=summary(measured.intervals),draw=summary(measured.renderCosts);
 assert.equal(errors.length,0);assert(requests.every(u=>u.startsWith(base)),'Unexpected external/live request');
 assert(measured.trackCount<=256);
 // Narrow viewport layout / keyboard control verification.
 await page.setViewportSize({width:760,height:1100});await page.waitForTimeout(200);
 assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
 await page.locator('h1').click();await page.keyboard.press('Space');assert((await state()).playing);await page.keyboard.press('Space');
 const report={status:'PASS',browser:browser.version(),viewport:{width:1440,height:1100,dpr:1},
   conditions:'Headless Chromium, real-time playback, software rendering; concurrent Playwright video capture. Not a hardware/live-service qualification.',
   pageErrors:errors,requests:requests.length,externalRequests:0,frame_interval_ms:frame,render_cpu_ms:draw,
   fps:1000/frame.mean,motion:{sampled_frames:motion.length,between_install_motion_frames:betweenUpdates,
   accumulated_ego_turn_deg:headingDelta,continuity_samples:continuitySamples,max_per_frame_object_step_m:maxStep},
   fading_observed:faded,interruption_expiry:true,seek_deterministic:true,pose_frames_inspected:[1480,2020,2748],
   max_events:measured.eventCount,final_tracks:measured.trackCount};
 fs.writeFileSync(path.join(out,'BROWSER_CHECK.json'),JSON.stringify(report,null,2)+'\n');
 await context.close();await browser.close();
 const video=await page.video().path();fs.renameSync(video,path.join(out,'replay-demo.webm'));
 console.log(JSON.stringify(report,null,2));
})().catch(e=>{console.error(e);process.exit(1)});
