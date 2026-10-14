/* Live adapter browser check (dev tooling): drives the real /live page against a re-timed recorded feed.
 * Usage: PLAYWRIGHT_MODULE=... CHROMIUM=... node live_browser_check_v1.cjs <cell> <workdir> [port]
 * Phases: feed at 1x -> writer pause (STALE) -> viewer server killed (DISCONNECTED, expiry) -> server restarted
 * (new epoch, reset tracking, no backlog) -> injected stale/duplicate/other-session messages -> second tab opened
 * and closed abruptly.  Writes <workdir>/LIVE_BROWSER_CHECK.json and a normal-speed video. */
'use strict';
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const {spawn}=require('child_process'),fs=require('fs'),path=require('path');
const [cell,work,portArg]=process.argv.slice(2),port=Number(portArg||8775),root=path.resolve(__dirname,'../..');
const feedRoot=path.join(work,'feed'),url=`http://127.0.0.1:${port}/live`;
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
const env={...process.env};delete env.PYTHONPATH;
function server(tag){const out=fs.openSync(path.join(work,`server_${tag}.log`),'w');
  return spawn('python3',['-B','-m','spatial_map_coop.replay_ui_v1.server','--port',String(port),'--live-root',feedRoot,
    '--live-log',path.join(work,`VIEWER_EVENTS_${tag}.jsonl`)],{cwd:root,env,stdio:['ignore',out,out]});}
async function waitHttp(){for(let i=0;i<200;i++){try{const r=await fetch(`http://127.0.0.1:${port}/api/live/status`);if(r.ok)return;}catch(e){}await sleep(150);}throw Error('server did not start');}
(async()=>{
  fs.mkdirSync(work,{recursive:true});
  let s1=server('s1');await waitHttp();
  const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM?{executablePath:process.env.CHROMIUM}:{})});
  const context=await browser.newContext({viewport:{width:1440,height:900},recordVideo:{dir:work,size:{width:1440,height:900}}});
  const page=await context.newPage(),errors=[],external=[],networkErrors=[];
  page.on('pageerror',e=>errors.push(String(e)));page.on('console',m=>{if(m.type()!=='error')return;
    // The deliberate server kill produces resource/network errors in the console; anything else is a failure.
    (/Failed to load resource: net::ERR_(INCOMPLETE_CHUNKED_ENCODING|CONNECTION_REFUSED|EMPTY_RESPONSE)/.test(m.text())?networkErrors:errors).push(m.text());});
  page.on('request',r=>{if(!r.url().startsWith(`http://127.0.0.1:${port}/`))external.push(r.url());});
  await page.goto(url);
  const feeder=spawn('python3',['-B','-m','spatial_map_coop.replay_ui_v1.live_smoke_v1','--cell',cell,'--out',feedRoot,
    '--start-s','60','--duration-s','46','--yaw-from-scorer','--duplicate-every','30','--out-of-order-at','80',
    '--old-session-at','120','--pause-at-s','12','--pause-s','3'],{cwd:root,env,stdio:['ignore','pipe','pipe']});
  const t0=Date.now(),timeline=[],marks={};
  const sample=async()=>{const s=await page.evaluate(()=>window.liveInspection.getState());
    timeline.push({t:(Date.now()-t0)/1000,state:s.state,shown:s.view?s.view.shown.length:0,tracks:s.trackCount,frame:s.frame,
      epoch:s.epoch,received:s.received,accepted:s.accepted,reconnects:s.reconnects,egoStale:s.view?.egoStale??null,ego:!!s.view?.ego});return s;};
  const until=async(sec)=>{while((Date.now()-t0)/1000<sec){await sample();await sleep(100);}};
  await until(8);marks.live=await sample();
  await page.screenshot({path:path.join(work,'01-live.png')});
  await until(14.6);marks.stalePause=await sample();
  await page.screenshot({path:path.join(work,'02-stale-writer-pause.png')});
  await until(24);
  // Second tab opened and closed abruptly while streaming.
  const page2=await context.newPage();await page2.goto(url);await sleep(1500);await page2.close();
  await until(26);marks.beforeKill=await sample();
  s1.kill('SIGKILL');const killedAt=(Date.now()-t0)/1000;
  await until(27.8);marks.disconnected=await sample();
  await page.screenshot({path:path.join(work,'03-disconnected.png')});
  await until(29.5);marks.afterExpiry=await sample();
  const s2=server('s2');await waitHttp();const restartedAt=(Date.now()-t0)/1000;
  await until(38);marks.reconnected=await sample();
  await page.screenshot({path:path.join(work,'04-reconnected.png')});
  // Injected messages that must never reach the display engine or revive objects.
  marks.injection=await page.evaluate(()=>{const L=window.liveInspection,s=L.getState(),before={...s.refused},tracks=s.trackCount;
    const base={type:'event',epoch:s.epoch,session:s.session,seq:s.lastSeq,objects:[],source:'x',frame_id:1,capture_ms:0,capture_ns:'1',server_ns:String(BigInt(Date.now())*1000000n)};
    L._onMessage(JSON.stringify(base));                                   // duplicate seq
    L._onMessage(JSON.stringify({...base,seq:s.lastSeq-3}));              // delayed / out of order
    L._onMessage(JSON.stringify({...base,seq:s.lastSeq+1000,session:'OLD/SESSION'}));  // old session
    L._onMessage(JSON.stringify({...base,seq:s.lastSeq+1000,epoch:'deadbeef0000'}));    // previous viewer epoch
    const after=L.getState();return {before,after:after.refused,tracksBefore:tracks,tracksAfter:after.trackCount,lastSeqAfter:after.lastSeq,lastSeqBefore:s.lastSeq};});
  await new Promise(r=>feeder.on('exit',r));
  await until((Date.now()-t0)/1000+2.5);marks.end=await sample();
  await page.screenshot({path:path.join(work,'05-end-stale.png')});
  const final=await page.evaluate(()=>window.liveInspection.getState());
  const video=page.video();await context.close();await browser.close();s2.kill('SIGTERM');
  const q=(a,p)=>{const s=a.slice().sort((x,y)=>x-y);return s.length?s[Math.min(s.length-1,Math.floor(p*(s.length-1)))]:null;};
  const states=[...new Set(timeline.map(x=>x.state))],stateAt=t=>timeline.filter(x=>x.t>=t).map(x=>x.state)[0];
  const disc=timeline.find(x=>x.t>killedAt&&x.state==='DISCONNECTED'),rec=timeline.find(x=>x.t>restartedAt&&x.state==='LIVE');
  const zeroAfterKill=timeline.find(x=>x.t>killedAt&&x.shown===0);
  const epochs=[...new Set(timeline.map(x=>x.epoch).filter(Boolean))];
  const result={schema:'scenesense.live_view.browser_check.v1',url,errors,expected_network_errors_during_kill:networkErrors.length,external_requests:external,states_seen:states,
    killed_at_s:killedAt,restarted_at_s:restartedAt,disconnected_detected_after_s:disc?disc.t-killedAt:null,
    objects_zero_after_kill_s:zeroAfterKill?zeroAfterKill.t-killedAt:null,live_again_after_restart_s:rec?rec.t-restartedAt:null,
    epochs,marks,render:{frames:final.frameCount,interval_p50_ms:q(final.intervals,.5),interval_p95_ms:q(final.intervals,.95),
      interval_max_ms:final.intervals.length?Math.max(...final.intervals):null,draw_p95_ms:q(final.renderCosts,.95),
      capture_to_display_p50_ms:q(final.ages,.5),capture_to_display_p95_ms:q(final.ages,.95)},
    video:video?await video.path():null,timeline};
  const checks={no_page_errors:errors.length===0,no_external_requests:external.length===0,
    live_reached:marks.live.state==='LIVE'&&marks.live.view&&marks.live.view.shown.length>0&&!!marks.live.view.ego,
    writer_pause_marked_stale:timeline.some(x=>x.t>12&&x.t<17&&x.state==='STALE'&&x.shown===0)&&timeline.some(x=>x.t>15&&x.t<20&&x.state==='LIVE'),
    disconnect_detected_within_2s:!!disc&&disc.t-killedAt<=2.0,
    objects_expired_after_disconnect:!!zeroAfterKill&&zeroAfterKill.t-killedAt<=2.0,
    reconnect_new_epoch_reset:epochs.length===2&&marks.reconnected.state==='LIVE'&&marks.reconnected.reconnects>=1,
    injected_messages_refused:marks.injection.lastSeqAfter===marks.injection.lastSeqBefore&&marks.injection.tracksAfter===marks.injection.tracksBefore,
    end_of_feed_stale:marks.end.state==='STALE'&&(marks.end.view?marks.end.view.shown.length:0)===0};
  result.checks=checks;result.status=Object.values(checks).every(Boolean)?'PASS':'FAIL';
  fs.writeFileSync(path.join(work,'LIVE_BROWSER_CHECK.json'),JSON.stringify(result,null,1));
  console.log(JSON.stringify({status:result.status,checks,render:result.render,disconnected_after:result.disconnected_detected_after_s,
    expired_after:result.objects_zero_after_kill_s,live_again_after:result.live_again_after_restart_s}));
})().catch(e=>{console.error(e);process.exit(1);});
