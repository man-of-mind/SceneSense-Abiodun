/* EVALUATION CANDIDATE (association only, pedestrians only): range-aware gate along the camera ray.
 *
 * The accepted engine gates pedestrian association with a 1.5-m circle, while the measured pedestrian detector
 * error is anisotropic (radial sd ~0.9 m, lag-correlated, 10% beyond 1.5 m; lateral sd ~0.2 m). Offline diagnosis
 * (analysis_v20_4) found that 94 of 175 GT-backed new pedestrian tracks were splits of a live track, 87 of them
 * radial-dominant. This candidate measures a pedestrian candidate's offset from a track's predicted and last
 * positions in the candidate's sensor-ray frame and uses the equivalent distance
 *     d_eff = gate * sqrt((radial / Rr)^2 + (lateral / gate)^2),   Rr = pedOutlierK * pedSigmaRadialM (= 2.7 m)
 * in the UNCHANGED gate test and cost formula. Purely lateral offsets are scored exactly as before; the old circle is
 * a strict subset of the new ellipse. Only the strong pass for pedestrians (gate = policy.gate) changes; weak updates,
 * vehicles, footprint absorption, conflicts, fits, display, expiry and visibility rules are the accepted engine's.
 * No new tunable: Rr reuses the accepted pedestrian outlier bound. Load after replay_tracks_v3.js or require().
 */
(function(root){'use strict';
const node=typeof module!=='undefined'&&module.exports;
const base=node?require(process.env.TRACKS_BASE||'../viewer_assets/replay_tracks_v3.js'):root.ScenesenseReplayTracks;
const P=node?require('../viewer_assets/presentation_predictor_v1.js'):root.ScenesenseMapPresentation;
const M=node?require('../viewer_assets/replay_motion_v2.js'):root.ScenesenseReplayMotion;
class RayGateTrackEngine extends base.TrackEngine {
  constructor(options={}){super(options);this.rayRadialM=options.rayRadialM??null;this.rayStats={pedestrianStrongPairsGatedOnlyByRay:0};} // rayRadialM: test hook only
  observe(e){this._sensor=e&&e.pose&&e.pose.sensor&&Number.isFinite(e.pose.sensor.x)?e.pose.sensor:null;return super.observe(e);}
  cost(t,c,ms,gate){ // assign() passes the per-class gate value: strong 1.5 m, weak 1.0 m for pedestrians
    const p=this.policy;
    if(t.cls!=='Pedestrian'||c.cls!=='Pedestrian'||gate!==p.gate.Pedestrian||!this._sensor)return super.cost(t,c,ms,gate);
    const s=this._sensor,rr=Math.hypot(c.x-s.x,c.y-s.y)||1,ux=(c.x-s.x)/rr,uy=(c.y-s.y)/rr,Rr=this.rayRadialM??p.pedOutlierK*p.pedSigmaRadialM;
    const eff=(x,y)=>{const dx=c.x-x,dy=c.y-y;return gate*Math.hypot((dx*ux+dy*uy)/Rr,(-dx*uy+dy*ux)/gate);};
    const pr=this.predictAt(t,ms),l=t.hist[t.hist.length-1],d=eff(pr.x,pr.y),dl=eff(l.x,l.y);
    if(Math.min(d,dl)>gate)return null;
    if(Math.min(Math.hypot(pr.x-c.x,pr.y-c.y),Math.hypot(l.x-c.x,l.y-c.y))>gate)this.rayStats.pedestrianStrongPairsGatedOnlyByRay++;
    // remainder identical to the accepted cost: size ratio, box overlap, heading (pedestrian: full-circle delta)
    if(Math.min(t.dims.l,c.dims.l)/Math.max(t.dims.l,c.dims.l)<.5)return null;
    const box=t.box&&c.box?1-M.overlap(t.box,c.box).iou:.5;
    const yaw=t.heading&&t.heading.known&&Number.isFinite(c.yaw)?Math.abs(P.angleDeltaDeg(t.heading.yaw,c.yaw))/180:0;
    return .7*d+.3*dl+.4*box+.3*yaw;}
}
const api={...base,TrackEngine:RayGateTrackEngine,BaseTrackEngine:base.TrackEngine,RAY_GATE_CANDIDATE:{radialBound:'pedOutlierK*pedSigmaRadialM',scope:'pedestrian strong pass'}};
root.ScenesenseReplayTracks=api;if(node)module.exports=api;
})(typeof globalThis!=='undefined'?globalThis:this);
