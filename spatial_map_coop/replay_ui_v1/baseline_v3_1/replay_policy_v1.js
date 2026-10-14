// Pure replay presentation policy. Measurements and scientific association are read-only.
(function(root) {
  'use strict';
  const P = typeof module !== 'undefined' && module.exports ? require('./presentation_predictor_v1.js') : root.ScenesenseMapPresentation;
  const M = typeof module !== 'undefined' && module.exports ? require('./replay_motion_v2.js') : root.ScenesenseReplayMotion;
  const SETTINGS = Object.freeze({horizonMs:500, freshMs:350, fadeMs:500, expiryMs:750,
    correctionMs:120, stationarySpeedMps:.25, maxCorrectionM:3, velocityConsistencyMps:6,
    maxTracks:256, maxEvents:256, radiusM:45});
  const finite = x => typeof x === 'number' && Number.isFinite(x);
  const validXY = p => p && finite(p.x) && finite(p.y);
  function poseReason(e) {
    const p=e.pose;
    if (!p) return e.pose_error || 'MISSING_FRAME_POSE';
    if (p.source!==e.source || p.frame_id!==e.frame_id || p.capture_ns!==e.capture_ns ||
        !p.binding || p.binding[0]+'/'+p.binding[1]!==e.session ||
        p.binding[2]!==e.source || p.binding[3]!==e.frame_id || p.binding[4]!==e.capture_ns)
      return 'POSE_IDENTITY_MISMATCH';
    if (!validXY(p) || !finite(p.yaw) || !validXY(p.sensor) || !finite(p.sensor.yaw)) return 'INVALID_POSE';
    if (!finite(p.fov_deg) || p.fov_deg<=0 || p.fov_deg>=180 || !finite(p.range_m) || p.range_m<=0)
      return 'MISSING_CALIBRATED_FOV_OR_RANGE';
    if (!finite(e.capture_ms) || !finite(e.arrival_ms) || !e.clock_domain ||
        typeof e.capture_ns!=='string' || !/^[1-9][0-9]*$/.test(e.capture_ns)) return 'MISSING_TIMESTAMP';
    // Same-host wall-clock mappings: the recorded replay and the live view (browser clock synchronized to the
    // viewer host, which shares the capture host clock). Both require unix wall ns and arrival after capture.
    if ((e.age_mapping === 'RECORDED_SAME_HOST_WALL' || e.age_mapping === 'LIVE_SAME_HOST_WALL') &&
        (e.clock_domain!=='unix_wall_ns' || e.arrival_ms<e.capture_ms)) return 'INVALID_CLOCK_MAPPING';
    return null;
  }
  function region(point, sensor, fov, range) {
    if (!validXY(point) || !validXY(sensor) || !finite(sensor.yaw) || !finite(fov) || !finite(range)) return 'INVALID_GEOMETRY';
    const dx=point.x-sensor.x, dy=point.y-sensor.y, a=sensor.yaw*Math.PI/180;
    const forward=dx*Math.cos(a)+dy*Math.sin(a), lateral=-dx*Math.sin(a)+dy*Math.cos(a);
    if (forward<=0) return 'BEHIND_SENSOR';
    if (Math.hypot(dx,dy)>range+1e-7) return 'OUTSIDE_RANGE';
    if (Math.abs(lateral)>forward*Math.tan(fov*Math.PI/360)+1e-7) return 'OUTSIDE_HORIZONTAL_FOV';
    return null;
  }
  function captureRegion(raw, pose) {
    const point=raw.location;
    const a=pose.yaw*Math.PI/180;
    if((point.x-pose.x)*Math.cos(a)+(point.y-pose.y)*Math.sin(a)<=0) return 'BEHIND_EGO';
    const horizontal=region(point,pose.sensor,pose.fov_deg,pose.range_m);
    if (horizontal) return horizontal;
    // 3D capture-time camera coordinates, CARLA forward/right/up axes. The
    // top-down wedge alone is not a vertical-frustum or visibility check.
    const m=pose.camera_matrix;
    if (m && finite(point.z)) {
      const v=[point.x-m[0][3],point.y-m[1][3],point.z-m[2][3]];
      const local=[0,1,2].map(j=>v.reduce((s,x,i)=>s+x*m[i][j],0));
      if (local[0]<=0) return 'BEHIND_CAMERA';
      if (Math.abs(local[1])>local[0]*Math.tan(pose.fov_deg*Math.PI/360)+1e-7) return 'OUTSIDE_CAMERA_FOV';
      if (Math.abs(local[2])>local[0]*Math.tan(pose.fov_deg*Math.PI/360)*720/1280+1e-7) return 'OUTSIDE_VERTICAL_FOV';
    }
    return null;
  }
  function visibility(entry,e) {
    // FoV, map buildings, model confidence, and single centre rays cannot set
    // visibility. Optional reviewed evidence must name this exact observation.
    const v=entry.visibility;
    if (v && ['visible','occluded'].includes(v.state) && v.qualified===true &&
        v.method==='FRAME_BOUND_REVIEW' && v.source===entry.source && v.frame_id===entry.frame_id &&
        v.capture_ns===entry.capture_ns && v.observation_id===entry.raw.id && v.evidence_ref &&
        e.pose && v.pose_binding===JSON.stringify(e.pose.binding))
      return {state:v.state,reason:v.reason || 'FRAME_BOUND_REVIEW',evidence_ref:v.evidence_ref};
    return {state:'unknown',reason:'NO_QUALIFIED_VISIBILITY_EVIDENCE'};
  }
  function sensorAt(pose, ego) {
    const d=P.angleDeltaDeg(pose.yaw,ego.yaw)*Math.PI/180;
    const dx=pose.sensor.x-pose.x,dy=pose.sensor.y-pose.y;
    return {x:ego.x+dx*Math.cos(d)-dy*Math.sin(d),y:ego.y+dx*Math.sin(d)+dy*Math.cos(d),
      yaw:pose.sensor.yaw+P.angleDeltaDeg(pose.yaw,ego.yaw)};
  }
  class Engine {
    constructor(options={}) { this.refined=!!options.refined; this.consolidation=options.consolidation!==false; this.reset(); }
    reset(source=null) {
      this.source=source; this.session=null; this.latest=new Map(); this.tracks=new Map();
      this.predictor=new P.DisplayPredictor({...SETTINGS,teleportDistanceM:5,maxSampleGapMs:500});
      this.egoPredictor=new P.DisplayPredictor({...SETTINGS,maxCorrectionM:4,velocityConsistencyMps:null,teleportDistanceM:8,maxSampleGapMs:500});
      if(this.refined){this.predictor=new M.Filter();this.egoPredictor=new M.Filter({ego:true});}
      this.nextID=0;this.qualifiedHistory=new Map();this.churn={created:0,backend_id_changes:0};
      this.rejections={}; this.accepted=0;
    }
    reject(reason) {this.rejections[reason]=(this.rejections[reason]||0)+1;return reason;}
    observe(e) {
      if (!e || !Array.isArray(e.objects) || !finite(e.capture_ms) || !finite(e.arrival_ms) ||
          !e.source || !e.session) return this.reject('INVALID_EVENT');
      if (this.session!==null && this.session!==e.session) this.reset(this.source);
      this.session=e.session;
      const old=this.latest.get(e.source);
      if (old && (e.frame_id===old.frame_id || e.capture_ms===old.capture_ms)) return this.reject('DUPLICATE');
      if (old && (e.frame_id<old.frame_id || e.capture_ms<old.capture_ms)) return this.reject('OUT_OF_ORDER');
      if(this.refined){
        const original=e.objects;
        const prior=this.qualifiedHistory.get(e.source);
        const representatives=M.match(this,M.consolidate(original,this.consolidation,prior&&e.capture_ms-prior.time<=500?prior.entries:[]),e);
        if(original.some(o=>o.upstream)){this.qualifiedHistory.set(e.source,{time:e.capture_ms,entries:original});if(this.qualifiedHistory.size>8)this.qualifiedHistory.delete(this.qualifiedHistory.keys().next().value);}
        e={...e,objects:representatives,original_objects:original};
      }
      this.latest.set(e.source,e);
      if (!this.source) this.source=e.source;
      const reason=poseReason(e);
      const age=e.age_mapping==='RECORDED_SAME_HOST_WALL' && !reason ? e.arrival_ms-e.capture_ms : null;
      if (!reason) this.egoPredictor.observe(e.source,{x:e.pose.x,y:e.pose.y,yaw:e.pose.yaw,
        measuredMs:e.capture_ms,sampleId:e.frame_id,className:'Vehicle',ageAtArrivalMs:age},e.arrival_ms);
      else delete this.egoPredictor.states[e.source];
      // Prune before insertion; interrupted streams cannot grow the store.
      for (const [key,t] of this.tracks) if (e.arrival_ms-t.event.arrival_ms>SETTINGS.expiryMs) this.tracks.delete(key);
      for (const entry of e.objects) {
        const raw=entry.raw, key=e.session+'|'+entry.source+'|'+entry.track_id;
        const before=this.tracks.get(key), xy=raw.location;
        if (!validXY(xy)) continue;
        let yaw=entry.heading_valid && finite(this.refined?entry.heading_yaw_deg:raw.yaw_deg) ? (this.refined?entry.heading_yaw_deg:raw.yaw_deg) : (before ? before.yaw : 0);
        let heading=entry.heading_valid ? (this.refined?'producer world yaw':'measured') : (before ? before.heading : 'unknown');
        const dt=before ? (e.capture_ms-before.event.capture_ms)/1000 : 0;
        if (!entry.heading_valid && !(this.refined&&before?.heading.startsWith('producer')) && dt>0 && dt<=.5) {
          const dx=xy.x-before.entry.raw.location.x,dy=xy.y-before.entry.raw.location.y;
          if (Math.hypot(dx,dy)/dt>.5 && Math.hypot(dx,dy)<5) {
            const motion=Math.atan2(dy,dx)*180/Math.PI;
            yaw=heading==='unknown' ? motion : yaw+Math.max(-60*dt,Math.min(60*dt,P.angleDeltaDeg(yaw,motion)));
            heading='motion estimate';
          }
        }
        if(this.refined&&!entry.heading_valid&&before?.heading.startsWith('producer'))heading='producer world yaw (held; current missing)';
        const status=this.predictor.observe(key,{x:xy.x,y:xy.y,yaw,className:raw.type,
          measuredMs:e.capture_ms,sampleId:e.frame_id,ageAtArrivalMs:age},e.arrival_ms);
        const generation=(before?.generation||0)+(status==='RESET'?1:0);
        this.tracks.set(key,{key,entry,event:e,yaw,heading,reset:status==='RESET',generation,
          resetAt:status==='RESET'?e.arrival_ms:(before?.resetAt ?? null)});
      }
      while (this.tracks.size>SETTINGS.maxTracks) this.tracks.delete(this.tracks.keys().next().value);
      this.predictor.retain([...this.tracks.keys()]);
      // Source inventory is bounded independently from observation count.
      if (this.latest.size>8) this.latest.delete(this.latest.keys().next().value);
      this.egoPredictor.retain([...this.latest.keys()]);
      this.accepted++; return reason || 'ACCEPTED';
    }
    view(now, options={}) {
      const selected=this.latest.get(this.source), mode=options.mode || 'current';
      const radius=options.radius || SETTINGS.radiusM, pReason=selected ? poseReason(selected) : 'AWAITING_POSE';
      const ego=!pReason ? this.egoPredictor.sample(this.source,now) : null;
      const sensor=ego ? sensorAt(selected.pose,ego) : null;
      const rows=[], rawRows=[];
      for (const [key,t] of this.tracks) {
        const {entry,event}=t, raw=entry.raw;
        const mapped=event.age_mapping==='RECORDED_SAME_HOST_WALL' && !poseReason(event);
        const age=mapped ? now-event.capture_ms : now-event.arrival_ms;
        const predicted=this.predictor.sample(key,now);
        if (!predicted) continue;
        let reason=poseReason(event);
        if (entry.source!==this.source || event.source!==this.source) reason='OTHER_SOURCE';
        else if (entry.frame_id!==event.frame_id || entry.capture_ns!==event.capture_ns) reason='OBJECT_FRAME_MISMATCH';
        else if (pReason) reason=pReason;
        else if (!reason) reason=captureRegion(raw,event.pose);
        const current=selected && event.frame_id===selected.frame_id;
        const coast=!current || age>SETTINGS.freshMs;
        if (!reason && age<0) reason='FUTURE_TIMESTAMP';
        if (!reason && age>=SETTINGS.expiryMs) reason='EXPIRED';
        if (!reason && mode==='current' && coast) reason='COASTING_NOT_CURRENT';
        if (!reason && sensor) reason=region(predicted,sensor,selected.pose.fov_deg,selected.pose.range_m);
        if (!reason && ego) {
          const a=ego.yaw*Math.PI/180;
          if((predicted.x-ego.x)*Math.cos(a)+(predicted.y-ego.y)*Math.sin(a)<=0) reason='BEHIND_DISPLAY_EGO';
        }
        if (!reason && ego && Math.hypot(predicted.x-ego.x,predicted.y-ego.y)>radius) reason='OUTSIDE_DISPLAY_RADIUS';
        const vis=visibility(entry,event);
        if (!reason && mode==='visible' && vis.state!=='visible') reason=vis.state==='occluded' ? 'OCCLUDED' : 'VISIBILITY_UNKNOWN';
        const resetting=t.resetAt!==null && now-t.resetAt<SETTINGS.correctionMs;
        const state=coast ? 'coasting' : resetting ? 'reset' : predicted.held ? 'held' : age>0 ? 'predicted' : 'measured';
        const row={...t,key:key+':display'+t.generation,pose:predicted,ageMs:age,clockMode:mapped?'capture age':'arrival age only',
          current,coast,state,visibility:vis,reason,shown:!reason,
          alpha:Math.max(0,Math.min(1,(SETTINGS.expiryMs-age)/(SETTINGS.expiryMs-SETTINGS.fadeMs),
            t.resetAt===null?1:(now-t.resetAt)/SETTINGS.correctionMs))};
        rows.push(row);
      }
      // Debug shows the latest RAW measurements, never extrapolated or relabelled as observations.
      for (const event of this.latest.values()) for (const entry of (event.original_objects||event.objects)) {
        const match=rows.find(r=>r.entry.raw.id===entry.raw.id);
        const representative=rows.find(r=>r.entry.members?.some(m=>m.raw.id===entry.raw.id));
        if(!match&&representative){const suppressed={...representative,entry,key:entry.raw.id,reason:'CONSOLIDATED_HYPOTHESIS',shown:false,representative:representative.entry.raw.id};rawRows.push(suppressed);rows.push(suppressed);continue;}
        rawRows.push(match || {entry,event,key:entry.raw.id,reason:'NOT_TRACKED',visibility:visibility(entry,event)});
      }
      return {ego,sensor,selected,poseReason:pReason,rows,rawRows,
        visible:rows.filter(r=>r.shown),clockMode:selected?.age_mapping==='RECORDED_SAME_HOST_WALL'?'capture':'arrival',
        egoStale:!selected || now-selected.arrival_ms>=SETTINGS.expiryMs};
    }
  }
  const api={SETTINGS,Engine,region,captureRegion,poseReason,visibility,sensorAt};
  root.ScenesenseReplay=api;
  if(typeof module!=='undefined'&&module.exports) module.exports=api;
})(typeof globalThis!=='undefined'?globalThis:this);
