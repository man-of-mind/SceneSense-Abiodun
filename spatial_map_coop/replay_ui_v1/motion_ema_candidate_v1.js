/* EVALUATION CANDIDATE (display only): causal per-track EMA of the per-update position correction.
 *
 * The current engine applies every update's full correction (drawn position before the update minus the
 * fresh fit/anchor) over the 150-ms bridge. This candidate applies only a fraction
 *   alpha = 1 - exp(-dt_capture / tau)
 * through that bridge and keeps the rest as a per-track world-XY offset that rides with the fit (a causal
 * EMA with constant-velocity prediction). It is reset (full correction) when the motion state changes,
 * when the correction exceeds a per-class gate (a real manoeuvre is never deferred), and on a snap.
 * State lives on the display track, so identity, source and session changes reset it by construction.
 * Association, fits, motion state, tiers, visibility rules and expiry never read the offset.
 *
 * Load after replay_tracks_v3.js (browser) or require() it (node); it re-exports the same API with
 * TrackEngine replaced. Parameters were fixed before evaluation (no search).
 */
(function(root){'use strict';
const node=typeof module!=='undefined'&&module.exports;
const base=node?require(process.env.TRACKS_BASE||'../viewer_assets/replay_tracks_v3.js'):root.ScenesenseReplayTracks;
const EMA=Object.freeze({tauMs:250,gateM:Object.freeze({Vehicle:1.5,Pedestrian:1.0})});

class EmaTrackEngine extends base.TrackEngine {
  constructor(options={}){super(options);this.ema={...EMA,...(options.ema||{})};this.emaStats={partial:0,reset:0};}
  display(t,now){const d=super.display(t,now);if(t.ema){d.x+=t.ema.x;d.y+=t.ema.y;}return d;}
  update(t,c,e,strong,fresh=false){
    const ta=e.arrival_ms,before=fresh?null:this.display(t,ta),held=t.ema,motion=t.motion,lastT=t.lastT;
    // The base engine decides snap vs bridge exactly as before (on its own correction, without the offset).
    t.ema=null;const alive=super.update(t,c,e,strong,fresh);
    if(!alive||!before||!t.bridge)return alive;
    const after=this.base(t,ta),dx=before.x-after.x,dy=before.y-after.y,m=Math.hypot(dx,dy);
    const reset=t.motion!==motion||m>this.ema.gateM[t.cls];
    if(reset&&!held)return alive;              // identical to the base engine
    const a=reset?1:1-Math.exp(-Math.max(0,e.capture_ms-lastT)/this.ema.tauMs);
    this.emaStats[reset?'reset':'partial']++;
    t.bridge={x:a*dx,y:a*dy,t0:ta};t.ema=a<1?{x:(1-a)*dx,y:(1-a)*dy}:null;
    return alive;}
}
const api={...base,TrackEngine:EmaTrackEngine,BaseTrackEngine:base.TrackEngine,EMA_CANDIDATE:EMA};
root.ScenesenseReplayTracks=api;if(node)module.exports=api;
})(typeof globalThis!=='undefined'?globalThis:this);
