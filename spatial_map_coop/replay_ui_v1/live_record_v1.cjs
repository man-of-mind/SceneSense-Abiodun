/* Normal-speed screen recording of the live view during ONE run (dev tooling, display only).
 * Usage: PLAYWRIGHT_MODULE=... CHROMIUM=... node live_record_v1.cjs <url> <outdir> <stopfile> [maxSeconds]
 * Records a 1280x800 video and samples window.liveInspection once per second into STATE_1HZ.jsonl.
 * Stops when <stopfile> exists or after maxSeconds; never talks to the pipeline (only the viewer URL). */
'use strict';
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const fs=require('fs'),path=require('path');
const [url,out,stopFile,maxArg]=process.argv.slice(2),maxSeconds=Number(maxArg||1800);
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
(async()=>{
  fs.mkdirSync(out,{recursive:true});
  const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM?{executablePath:process.env.CHROMIUM}:{})});
  const context=await browser.newContext({viewport:{width:1280,height:800},recordVideo:{dir:out,size:{width:1280,height:800}}});
  const page=await context.newPage(),errors=[],states=fs.openSync(path.join(out,'STATE_1HZ.jsonl'),'w');
  page.on('pageerror',e=>errors.push({t:Date.now(),error:String(e)}));
  page.on('console',m=>{if(m.type()==='error')errors.push({t:Date.now(),console:m.text()});});
  const started=Date.now();await page.goto(url+(url.includes('?')?'&':'?')+'client=recorder');
  let last=null;
  while(!fs.existsSync(stopFile)&&(Date.now()-started)/1000<maxSeconds){
    try{const s=await page.evaluate(()=>{const x=window.liveInspection.getState();return {state:x.state,frame:x.frame,received:x.received,accepted:x.accepted,
        refused:x.refused,reconnects:x.reconnects,tracks:x.trackCount,shown:x.view?x.view.shown.length:0,egoStale:x.view?.egoStale??null,
        ped:x.view?x.view.shown.filter(r=>r.cls==='Pedestrian').length:0,veh:x.view?x.view.shown.filter(r=>r.cls==='Vehicle').length:0,
        age:x.ages.length?x.ages[x.ages.length-1]:null,frames:x.frameCount};});
      last=s;fs.writeSync(states,JSON.stringify({wall_ms:Date.now(),video_s:(Date.now()-started)/1000,...s})+'\n');}
    catch(e){errors.push({t:Date.now(),sample:String(e)});}
    await sleep(1000);
  }
  const final=await page.evaluate(()=>{const x=window.liveInspection.getState();return {intervals:x.intervals,costs:x.renderCosts,ages:x.ages};}).catch(()=>null);
  const video=page.video();await context.close();await browser.close();fs.closeSync(states);
  const q=(a,p)=>{if(!a||!a.length)return null;const s=a.slice().sort((x,y)=>x-y);return s[Math.min(s.length-1,Math.floor(p*(s.length-1)))];};
  const summary={schema:'scenesense.live_view.recording.v1',url,started_wall_ms:started,duration_s:(Date.now()-started)/1000,
    video:video?await video.path():null,errors,last_state:last,
    tail_window:{note:'last 600 render intervals / draw costs / 240 display ages held by the page',
      interval_p50_ms:q(final?.intervals,.5),interval_p95_ms:q(final?.intervals,.95),draw_p95_ms:q(final?.costs,.95),
      capture_to_display_p50_ms:q(final?.ages,.5),capture_to_display_p95_ms:q(final?.ages,.95)}};
  fs.writeFileSync(path.join(out,'RECORDING.json'),JSON.stringify(summary,null,1));
  console.log(JSON.stringify({status:'RECORDED',video:summary.video,duration_s:summary.duration_s,errors:errors.length}));
})().catch(e=>{console.error(e);process.exit(1);});
