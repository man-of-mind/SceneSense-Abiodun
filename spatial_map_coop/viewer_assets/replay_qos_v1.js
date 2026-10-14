/* Display QoS strip v1 (replay and live): read-only values derived from stamps the feed already carries.
 *
 *   capture -> install   latest frame: map-install stamp minus capture stamp (same host wall clock only)
 *   displayed data age   frame/map age: display time minus the latest displayed frame's capture stamp
 *   update rate          map updates per second over the last 5 s (needs >= 2 updates)
 *   since last update    display time minus the time the latest update reached this display
 *   object observations  age of each SHOWN object's own latest observation (>= frame age when coasting)
 *   prediction/feedback  not in this feed: no prediction or UE-feedback stamp arrives with a frame join
 *
 * Anything that cannot be derived is null and renders as "not available"; nothing is estimated or zeroed.
 * Display timing only; unrelated to the 170-ms scientific deadline. */
(function(root){'use strict';
const NA='not available',RATE_WINDOW_MS=5000;
const finite=x=>typeof x==='number'&&Number.isFinite(x);
const sameHostWall=e=>!!e&&(e.age_mapping==='RECORDED_SAME_HOST_WALL'||e.age_mapping==='LIVE_SAME_HOST_WALL');
const median=a=>{const s=a.slice().sort((x,y)=>x-y),n=s.length;return n%2?s[(n-1)/2]:(s[n/2-1]+s[n/2])/2;};
// installMs: map-install time of `latest` on the display clock (null if not supplied).
// receipts: times (display clock, ascending) at which recent updates reached this display.
function compute({now,latest,installMs=null,receipts=[],visible=[]}){
  const ok=finite(now)&&latest&&sameHostWall(latest)&&finite(latest.capture_ms);
  const recent=receipts.filter(t=>finite(t)&&t<=now&&now-t<=RATE_WINDOW_MS),last=receipts.length?receipts[receipts.length-1]:null;
  const ages=visible.map(r=>r.ageMs).filter(finite);
  return {
    captureToInstallMs:ok&&finite(installMs)&&installMs>=latest.capture_ms?installMs-latest.capture_ms:null,
    displayedDataAgeMs:ok&&now>=latest.capture_ms?now-latest.capture_ms:null,
    updateRateHz:recent.length>=2&&recent[recent.length-1]>recent[0]?(recent.length-1)/((recent[recent.length-1]-recent[0])/1000):null,
    sinceLastUpdateMs:finite(now)&&finite(last)&&now>=last?now-last:null,
    objectAgeP50Ms:ok&&ages.length?median(ages):null,objectAgeMaxMs:ok&&ages.length?Math.max(...ages):null,objectCount:ages.length,
    predictionFeedbackMs:null,predictionFeedbackReason:'not in this feed (no prediction or UE-feedback timing with a frame join)'};}
const ms=x=>finite(x)?(x>=10000?(x/1000).toFixed(1)+' s':Math.round(x)+' ms'):NA;
function format(q){return {captureToInstall:ms(q.captureToInstallMs),displayedDataAge:ms(q.displayedDataAgeMs),
  updateRate:finite(q.updateRateHz)?q.updateRateHz.toFixed(1)+' /s':NA,sinceLastUpdate:ms(q.sinceLastUpdateMs),
  objectAge:finite(q.objectAgeP50Ms)?`median ${ms(q.objectAgeP50Ms)} · oldest ${ms(q.objectAgeMaxMs)}`:NA+(q.objectCount===0?' (no objects shown)':''),
  predictionFeedback:NA+' — '+q.predictionFeedbackReason};}
const api={compute,format,NOT_AVAILABLE:NA,RATE_WINDOW_MS};
root.ScenesenseDisplayQos=api;if(typeof module!=='undefined'&&module.exports)module.exports=api;
})(typeof globalThis!=='undefined'?globalThis:this);
