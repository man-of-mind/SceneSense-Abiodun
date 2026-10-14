/* Presentation-only display tracks, v3.
 *
 * A private, causal display layer over recorded installed-map events. It never
 * edits raw observations, backend/scientific association, GT, scores or metrics.
 * Every raw candidate keeps its original record and gets an explicit display
 * disposition. Policies are documented in replay_ui_v1/README.md (v3 section).
 *
 * Tiers:
 *   established  - a display track confirmed by >=2 strong detections in 3 frames
 *   tentative    - a strong detection without temporal confirmation yet
 *   hypothesis   - weak (sub-threshold) candidates not explained by a track; shown
 *                  only as a decaying "unresolved hypotheses" field, never as cars
 * Strong/weak uses the project's registered evaluator detection thresholds
 * (rl_agent/splitfusion_minimal_quality_scorer_v1/scorer.py: vehicle .20, person .25).
 */
(function(root){'use strict';
const node=typeof module!=='undefined'&&module.exports;
const P=node?require('./presentation_predictor_v1.js'):root.ScenesenseMapPresentation;
const R=node?require('./replay_policy_v1.js'):root.ScenesenseReplay;
const M=node?require('./replay_motion_v2.js'):root.ScenesenseReplayMotion;

const POLICY=Object.freeze({
  strongScore:Object.freeze({Vehicle:.20,Pedestrian:.25}),
  confirmHits:2, confirmWindow:3, tentativeMaxMisses:2, weakOnlyMaxFrames:6,
  freshMs:350, fadeStartMs:450, expiryMs:750, horizonMs:500,
  gate:Object.freeze({Vehicle:2.5,Pedestrian:1.5}), weakGate:Object.freeze({Vehicle:1.5,Pedestrian:1.0}),
  motionWindowMs:1200, historyMax:20, minMotionSamples:4, motionSignificance:3, stopSignificance:1.5, noiseFloorM:.25,
  directionFlipVote:.3, directionKnownVote:.6,
  moveEnter:Object.freeze({Vehicle:2.0,Pedestrian:1.0}), moveExit:Object.freeze({Vehicle:.6,Pedestrian:.3}),
  moveDisplacement:Object.freeze({Vehicle:2.0,Pedestrian:1.5}), rangeBiasSlope:.15, anchorSamples:20, motionHeadingSpeed:Object.freeze({Vehicle:2.5,Pedestrian:.8}),
  headingHoldDeg:20, headingHoldUpdates:3, yawRateDps:90, footprintMarginM:.6,
  vehicleHeadingWindowMs:800, vehicleAxisConsistencyDeg:12, vehicleAxisUpdates:4, vehicleAxisSpanMs:350,
  vehicleMotionHeadingUpdates:3,
  bridgeMs:150, snapM:4, fieldCellM:1.5, fieldTauMs:350, fieldMinWeight:.12, fieldMaxCells:2048,
  maxTracks:256, maxEvents:256, replayHistoryMs:3000,
  // Pedestrian motion v3.1 (display only). Priors measured on the registered capture:
  // confident pedestrian detections vs archived GT, radial sd .96 m (lag-240 ms corr .42),
  // tangential sd .23 m. 'v3.0' restores the previous pedestrian policy for comparison.
  pedestrianMotion:'v3.1', pedWindowMs:2500, pedMinSpanMs:500, pedSigmaRadialM:.9, pedSigmaTangentialM:.2, pedPriorDof:4,
  pedRadialRho:.65, pedOutlierK:3, pedEnterZ:3, pedEnterUpdates:2, pedExitZ:1.5, pedExitUpdates:2, pedStationaryBoundMps:.3, pedShrinkK:2,
  pedChangeWindowMs:800, pedChangeZ:2.5, pedChangeUpdates:2, pedCentroid:'mean'});

const finite=x=>typeof x==='number'&&Number.isFinite(x);
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
const smooth=u=>{u=clamp(u,0,1);return 1-u*u*(3-2*u);};
const cls=t=>t==='Pedestrian'?'Pedestrian':'Vehicle';
const median=a=>{const s=a.slice().sort((x,y)=>x-y),n=s.length;return n?(n%2?s[(n-1)/2]:(s[n/2-1]+s[n/2])/2):NaN;};
const delta=(a,b)=>P.angleDeltaDeg(a,b);
const wrap=a=>((a%360)+540)%360-180;
const axisDelta=(a,b)=>{const d=wrap(b-a);return d>90?d-180:d<-90?d+180:d;};
function axisMean(items){if(!items.length)return null;let sx=0,sy=0,w=0;for(const m of items){const r=m.yaw*Math.PI/90,k=Math.max(.05,m.score);sx+=Math.cos(r)*k;sy+=Math.sin(r)*k;w+=k;}
  return {deg:Math.atan2(sy,sx)*90/Math.PI,resultant:Math.hypot(sx,sy)/w};}
function circularMean(items){let sx=0,sy=0,w=0;for(const [deg,wt] of items){const r=deg*Math.PI/180;sx+=Math.cos(r)*wt;sy+=Math.sin(r)*wt;w+=wt;}
  if(!w)return null;const len=Math.hypot(sx,sy)/w;return {deg:Math.atan2(sy,sx)*180/Math.PI,resultant:len};}
function isStrong(raw){return finite(raw.score)&&raw.score>=POLICY.strongScore[cls(raw.type)];}
// Physical exclusion: a candidate centre inside another object's (expanded)
// footprint cannot be a separate vehicle. Adjacent/parked neighbours do not
// overlap footprints and are never merged by centroid proximity alone.
function insideFootprint(p,center,dims,yaw){const dx=p.x-center.x,dy=p.y-center.y,m=POLICY.footprintMarginM;
  if(!finite(yaw))return Math.hypot(dx,dy)<=dims.w/2+m;
  const a=yaw*Math.PI/180,u=dx*Math.cos(a)+dy*Math.sin(a),v=-dx*Math.sin(a)+dy*Math.cos(a);
  return Math.abs(u)<=dims.l/2+m&&Math.abs(v)<=dims.w/2+m;}
function candidateDims(raw){const d=raw.dimensions||{},c=cls(raw.type);
  return c==='Pedestrian'?{l:.8,w:.8}:{l:clamp(Number(d.length)||4.6,3.2,14),w:clamp(Number(d.width)||2,1.4,3.4)};}

class TrackEngine {
  constructor(options={}){this.policy={...POLICY,...(options.policy||{})};this.reset(options.source||null);}
  reset(source=null){
    this.source=source;this.session=null;this.latest=null;this.tracks=new Map();this.field=new Map();
    this.ego=new M.Filter({ego:true});this.nextTentative=0;this.nextEstablished=0;this.rejections={};
    this.lastDispositions=[];
    this.stats={tentativeCreated:0,established:0,endedExpired:0,endedWeakOnly:0,tentativeDropped:0,
      confirmCaptureMs:[],confirmArrivalMs:[],establishedLifetimeMs:[],snaps:0,bridges:[],absorbed:0,hypotheses:0,events:0};
  }
  reject(r){this.rejections[r]=(this.rejections[r]||0)+1;return r;}
  // ---- motion / pose -----------------------------------------------------
  base(t,now){ // undisplaced estimate at replay time `now` (capture clock)
    if(t.motion!=='moving')return {x:t.anchor.x,y:t.anchor.y,vx:0,vy:0};
    const dt=clamp(now-t.est.t,0,this.policy.horizonMs)/1000;return {x:t.est.x+t.est.vx*dt,y:t.est.y+t.est.vy*dt,vx:t.est.vx,vy:t.est.vy};}
  display(t,now){const b=this.base(t,now),o=t.bridge,k=o?smooth((now-o.t0)/this.policy.bridgeMs):0;
    return {x:b.x+(o?o.x*k:0),y:b.y+(o?o.y*k:0),vx:b.vx,vy:b.vy,bridgeM:o?Math.hypot(o.x,o.y)*k:0};}
  displayYaw(t,now){const h=t.heading;if(!h||!finite(h.yaw))return null;if(!finite(h.from))return h.yaw;
    const d=delta(h.from,h.yaw),dur=Math.abs(d)/this.policy.yawRateDps*1000;if(!dur)return h.yaw;
    return h.from+d*clamp((now-h.t0)/dur,0,1);}
  fitMotion(t){
    // Weighted least-squares constant-velocity fit over a short window. Movement
    // must be statistically significant (|v|/se >= 3), fast enough AND displaced
    // from the stopped anchor: detector range noise of 1-2 m otherwise makes
    // parked cars look like they move at 1-3 m/s.
    const c=t.cls,p=this.policy;
    if(c==='Pedestrian'&&p.pedestrianMotion!=='v3.0')return this.fitPedestrian(t);
    const last=t.hist[t.hist.length-1],win=t.hist.filter(h=>last.t-h.t<=p.motionWindowMs);
    let vx=0,vy=0,x=last.x,y=last.y,tstat=0;
    if(win.length>=3){let sw=0,st=0,sx=0,sy=0;for(const h of win){sw+=1;st+=h.t/1000;sx+=h.x;sy+=h.y;}
      const mt=st/sw,mx=sx/sw,my=sy/sw;let stt=0,stx=0,sty=0;for(const h of win){const d=h.t/1000-mt;stt+=d*d;stx+=d*(h.x-mx);sty+=d*(h.y-my);}
      if(stt>0){vx=stx/stt;vy=sty/stt;}x=mx+vx*(last.t/1000-mt);y=my+vy*(last.t/1000-mt);
      let rss=0;for(const h of win){const d=h.t/1000-mt;rss+=(h.x-mx-vx*d)**2+(h.y-my-vy*d)**2;}
      const sigma=Math.max(p.noiseFloorM,Math.sqrt(rss/Math.max(1,2*win.length-4)));
      tstat=stt>0?Math.hypot(vx,vy)/(sigma/Math.sqrt(stt)):0;}
    [vx,vy]=P.clampVelocity(vx,vy,P.maxDisplaySpeed(c));
    const speed=Math.hypot(vx,vy);t.tstat=tstat;
    // Detector range error is mostly radial and changes with range, so a parked
    // object appears to slide radially while the ego approaches. Radial world
    // speed must exceed enter + slope*|range rate| (both measured, no GT).
    let radialOk=true;
    if(win.length>=3&&finite(last.sx)){const r=win.map(h=>Math.hypot(h.x-h.sx,h.y-h.sy)),ts=win.map(h=>h.t/1000),mt=ts.reduce((a,b)=>a+b)/ts.length,mr=r.reduce((a,b)=>a+b)/r.length;
      let stt=0,str=0;ts.forEach((x,i)=>{stt+=(x-mt)**2;str+=(x-mt)*(r[i]-mr);});const rangeRate=stt>0?str/stt:0;
      const ux=(last.x-last.sx)/(r[r.length-1]||1),uy=(last.y-last.sy)/(r[r.length-1]||1),vr=vx*ux+vy*uy,vt=Math.abs(-vx*uy+vy*ux);
      radialOk=vt>p.moveEnter[c]||Math.abs(vr)>p.moveEnter[c]+p.rangeBiasSlope*Math.abs(rangeRate);t.rangeRate=rangeRate;}
    const stopAnchor=()=>{const still=t.hist.slice(-p.anchorSamples);t.anchor={x:median(still.map(h=>h.x)),y:median(still.map(h=>h.y))};};
    if(t.motion!=='moving'){
      const away=Math.hypot(last.x-t.anchor.x,last.y-t.anchor.y);
      if(win.length>=p.minMotionSamples&&speed>p.moveEnter[c]&&tstat>=p.motionSignificance&&away>p.moveDisplacement[c]&&radialOk){t.motion='moving';t.stillCount=0;}
      else{t.motion=win.length>=p.minMotionSamples||t.hist.length>=p.minMotionSamples?'stationary':'init';stopAnchor();}
    }else if(speed<p.moveExit[c]||tstat<p.stopSignificance){if(++t.stillCount>=2){t.motion='stationary';t.hist=t.hist.slice(-2);stopAnchor();}}
    else t.stillCount=0;
    t.est={x,y,vx:t.motion==='moving'?vx:0,vy:t.motion==='moving'?vy:0,t:last.t};t.speed=t.motion==='moving'?speed:0;
  }
  // Pedestrians: object presence is independent of this classification. Motion is
  // 'moving' only on statistically supported displacement (no speed floor),
  // 'stationary' only when the speed upper bound is small, otherwise 'unresolved'
  // (drawn with an uncertainty halo, no forward prediction). Noise is anisotropic:
  // range (radial) error is large and correlated, lateral error small.
  fitPedestrian(t){
    const p=this.policy,last=t.hist[t.hist.length-1];
    let win=t.hist.filter(h=>last.t-h.t<=p.pedWindowMs);
    const sx=finite(last.sx)?last.sx:last.x-1,sy=finite(last.sy)?last.sy:last.y,rr=Math.hypot(last.x-sx,last.y-sy)||1;
    const ux=(last.x-sx)/rr,uy=(last.y-sy)/rr; // radial unit vector at the newest sample
    const toRT=h=>({t:h.t/1000,r:(h.x-sx)*ux+(h.y-sy)*uy,g:-(h.x-sx)*uy+(h.y-sy)*ux,s:finite(h.sx)?(h.sx-sx)*ux+(h.sy-sy)*uy:0});
    const fit=pts=>{const n=pts.length,mt=pts.reduce((a,q)=>a+q.t,0)/n,mr=pts.reduce((a,q)=>a+q.r,0)/n,mg=pts.reduce((a,q)=>a+q.g,0)/n,ms=pts.reduce((a,q)=>a+q.s,0)/n;
      let stt=0,str=0,stg=0,sts=0;for(const q of pts){const d=q.t-mt;stt+=d*d;str+=d*(q.r-mr);stg+=d*(q.g-mg);sts+=d*(q.s-ms);}
      const vr=stt>0?str/stt:0,vg=stt>0?stg/stt:0;let rr2=0,rg2=0;for(const q of pts){const d=q.t-mt;rr2+=(q.r-mr-vr*d)**2;rg2+=(q.g-mg-vg*d)**2;}
      return {n,mt,mr,mg,vr,vg,stt,rr2,rg2,egoRadial:stt>0?sts/stt:0};};
    let pts=win.map(toRT),f=pts.length>=3?fit(pts):null;
    if(f){const kept=pts.filter(q=>{const d=q.t-f.mt;return Math.abs(q.r-f.mr-f.vr*d)<=p.pedOutlierK*p.pedSigmaRadialM&&Math.abs(q.g-f.mg-f.vg*d)<=p.pedOutlierK*p.pedSigmaTangentialM;});
      if(kept.length>=3&&kept.length<pts.length){pts=kept;f=fit(pts);}}
    // Causal change point: if the recent window's velocity is inconsistent with the
    // long fit for consecutive updates (a stop, start or turn), forget older samples.
    const seOf=(q)=>{const dof=Math.max(0,q.n-2),infl=Math.sqrt((1+p.pedRadialRho)/(1-p.pedRadialRho));
      return [Math.sqrt((p.pedPriorDof*p.pedSigmaRadialM**2+q.rr2)/(p.pedPriorDof+dof))*infl/Math.sqrt(q.stt),Math.sqrt((p.pedPriorDof*p.pedSigmaTangentialM**2+q.rg2)/(p.pedPriorDof+dof))/Math.sqrt(q.stt)];};
    const recent=pts.filter(q=>last.t/1000-q.t<=p.pedChangeWindowMs/1000);
    if(t.motion==='moving'&&f&&recent.length>=4&&recent.length<pts.length){const g=fit(recent);if(g.stt>0){const [a,b]=seOf(g),dz=Math.hypot((g.vr-f.vr)/a,(g.vg-f.vg)/b);
      t.changeCount=dz>=p.pedChangeZ?(t.changeCount||0)+1:0;
      if(t.changeCount>=p.pedChangeUpdates){t.changeCount=0;t.hist=t.hist.filter(h=>last.t-h.t<=p.pedChangeWindowMs);pts=recent;f=g;t.changePoints=(t.changePoints||0)+1;}}}
    const span=pts.length?(pts[pts.length-1].t-pts[0].t)*1000:0,world=(r,g)=>({x:sx+r*ux-g*uy,y:sy+r*uy+g*ux});
    let z=0,vr=0,vg=0,ser=Infinity,seg=Infinity,pos;
    if(f&&span>=p.pedMinSpanMs&&f.stt>0){const dof=Math.max(0,f.n-2),infl=Math.sqrt((1+p.pedRadialRho)/(1-p.pedRadialRho));
      const sr=Math.sqrt((p.pedPriorDof*p.pedSigmaRadialM**2+f.rr2)/(p.pedPriorDof+dof)),sg=Math.sqrt((p.pedPriorDof*p.pedSigmaTangentialM**2+f.rg2)/(p.pedPriorDof+dof));
      // range-bias drift while the ego approaches is systematic radial uncertainty
      ser=Math.hypot(sr*infl/Math.sqrt(f.stt),p.rangeBiasSlope*Math.abs(f.egoRadial));seg=sg/Math.sqrt(f.stt);
      vr=f.vr;vg=f.vg;z=Math.hypot(vr/ser,vg/seg);}
    t.tstat=z;t.motionUncertainty={radialSe:ser,tangentialSe:seg,ux,uy};
    // state machine with persistence on entry and exit
    if(t.motion==='moving'){t.exitCount=z<p.pedExitZ?(t.exitCount||0)+1:0;if(t.exitCount>=p.pedExitUpdates){t.motion='unresolved';t.enterCount=0;}}
    else{t.enterCount=z>=p.pedEnterZ?(t.enterCount||0)+1:0;if(t.enterCount>=p.pedEnterUpdates){t.motion='moving';t.exitCount=0;}}
    if(t.motion!=='moving'){const bounded=f&&span>=1000&&Math.abs(vr)+2*ser<p.pedStationaryBoundMps&&Math.abs(vg)+2*seg<p.pedStationaryBoundMps;
      t.motion=bounded?'stationary':'unresolved';}
    if(t.motion==='moving'){let [wx,wy]=[vr*ux-vg*uy,vr*uy+vg*ux];[wx,wy]=P.clampVelocity(wx,wy,P.maxDisplaySpeed('Pedestrian'));
      const tl=last.t/1000,at=world(f.mr+vr*(tl-f.mt),f.mg+vg*(tl-f.mt));t.est={x:at.x,y:at.y,vx:wx,vy:wy,t:last.t};t.speed=Math.hypot(wx,wy);t.anchor={x:at.x,y:at.y};return;}
    if(!f){const n=pts.length||1;pos={x:win.reduce((a,h)=>a+h.x,0)/n,y:win.reduce((a,h)=>a+h.y,0)/n};}
    else if(t.motion==='stationary')pos=world(median(pts.map(q=>q.r)),median(pts.map(q=>q.g)));
    else{const k=z*z/(z*z+p.pedShrinkK**2),tl=last.t/1000,mr=p.pedCentroid==='mean'?f.mr:median(pts.map(q=>q.r)),mg=p.pedCentroid==='mean'?f.mg:median(pts.map(q=>q.g));pos=world(mr+k*vr*(tl-f.mt),mg+k*vg*(tl-f.mt));}
    t.anchor=pos;t.est={x:pos.x,y:pos.y,vx:0,vy:0,t:last.t};t.speed=0;
  }
  updateHeading(t,arrival){
    if(t.cls==='Vehicle')return this.updateVehicleHeading(t,arrival);
    // Body axis (mod 180) and travel direction are separate. Stationary objects
    // rotate only after a sustained axis change; an opposed direction majority
    // swaps the nose marker without spinning the outline.
    const p=this.policy,produced=t.hist.filter(h=>finite(h.yaw)).slice(-8);
    const motion=t.cls==='Pedestrian'&&p.pedestrianMotion!=='v3.0'?t.motion==='moving':t.motion==='moving'&&t.speed>=p.motionHeadingSpeed[t.cls]&&t.tstat>=p.motionSignificance;
    const h=t.heading||{yaw:null,from:null,t0:arrival,known:false,source:'unknown',pending:0,flip:0};
    const current=this.displayYaw(t,arrival);
    const set=(yaw,from,known,source,extra={})=>{t.heading={...h,yaw:wrap(yaw),from:finite(from)?wrap(yaw)-delta(from,yaw):null,t0:arrival,known,source,...extra};};
    if(motion){let target=Math.atan2(t.est.vy,t.est.vx)*180/Math.PI,source='motion';
      const ax=axisMean(produced);if(ax&&Math.abs(axisDelta(ax.deg,target))<=40){source='motion+producer';}
      set(target,current,true,source,{pending:0,flip:0});return;}
    const ax=axisMean(produced);
    if(!ax){if(t.heading)t.heading={...h,known:false};return;}
    // Direction along the axis: keep the end nearest the current glyph.
    let dir=finite(current)?(Math.abs(delta(current,ax.deg))<=90?ax.deg:ax.deg+180):ax.deg;
    const total=produced.reduce((s,m)=>s+Math.max(.05,m.score),0),vote=produced.reduce((s,m)=>s+Math.max(.05,m.score)*Math.cos((m.yaw-dir)*Math.PI/180),0)/total;
    if(!finite(current)&&vote<0){dir+=180;}
    let flip=vote< -p.directionFlipVote?h.flip+1:0;
    if(flip>=p.headingHoldUpdates){dir+=180;flip=0;}
    const known=produced.length>=3&&Math.abs(vote)>=p.directionKnownVote&&ax.resultant>=.8;
    if(!finite(current)){set(dir,null,known,'producer world yaw',{pending:0,flip});return;}
    const nose=Math.abs(delta(current,dir))>90?current+180:current; // nose swap only
    if(Math.abs(axisDelta(nose,dir))<p.headingHoldDeg){set(nose,null,known,'producer world yaw (held)',{pending:0,flip});return;}
    const pending=h.pending+1;
    if(pending<p.headingHoldUpdates){set(nose,null,known,'producer world yaw (held)',{pending,flip});return;}
    set(dir,nose,known,'producer world yaw',{pending:0,flip});
  }
  updateVehicleHeading(t,arrival){
    // Body orientation is an AXIS. The producer's front/back vote is a separate
    // marker; reversing travel must never spin the body or invent a front end.
    const p=this.policy,last=t.hist[t.hist.length-1],current=this.displayYaw(t,arrival);
    const h={yaw:null,from:null,t0:arrival,known:false,source:'unknown',pending:0,pendingAxis:null,
      pendingSince:null,motionMode:false,motionVotes:0,...t.heading};
    const fresh=last.strong&&finite(last.yaw);
    const produced=t.hist.filter(q=>last.t-q.t<=p.vehicleHeadingWindowMs&&q.strong&&finite(q.yaw));
    const enter=t.motion==='moving'&&t.speed>=p.motionHeadingSpeed.Vehicle&&t.tstat>=p.motionSignificance;
    h.motionVotes=enter?h.motionVotes+1:0;
    if(h.motionMode&&(t.motion!=='moving'||t.speed<p.moveExit.Vehicle||t.tstat<p.stopSignificance))h.motionMode=false;
    if(h.motionVotes>=p.vehicleMotionHeadingUpdates)h.motionMode=true;
    const motionYaw=h.motionMode?Math.atan2(t.est.vy,t.est.vx)*180/Math.PI:null;
    h.travelYaw=motionYaw;h.travelSource=h.motionMode?'qualified motion fit':null;
    h.rawYaw=finite(last.yaw)?last.yaw:null;h.rawQualified=finite(last.yaw);h.rawStrong=last.strong;
    const clear=()=>{h.pending=0;h.pendingAxis=null;h.pendingSince=null;};
    const target=axis=>{
      // Shortest modulo-180 path; front/back is never included in interpolation.
      const yaw=finite(current)?current+axisDelta(current,axis):axis;
      h.yaw=wrap(yaw);h.from=finite(current)?h.yaw-axisDelta(current,yaw):null;h.t0=arrival;clear();
    };
    if(h.motionMode){
      h.source='motion axis (hysteresis)';
      // The causal velocity fit already spans 1.2 s. Following its axis directly
      // preserves turns; keep using it through small speed-threshold crossings.
      target(motionYaw);
    }else if(fresh){
      if(!finite(h.yaw)){target(last.yaw);h.source='producer axis';}
      else if(Math.abs(axisDelta(h.yaw,last.yaw))<p.headingHoldDeg){clear();}
      else{
        // Compare new qualified observations, not overlapping rolling means.
        // Missing/weak observations and contradictory axes break the sequence.
        if(!finite(h.pendingAxis)||Math.abs(axisDelta(h.pendingAxis,last.yaw))>p.vehicleAxisConsistencyDeg){
          h.pending=1;h.pendingAxis=last.yaw;h.pendingSince=last.t;
        }else h.pending++;
        if(h.pending>=p.vehicleAxisUpdates&&last.t-h.pendingSince>=p.vehicleAxisSpanMs){
          const support=produced.filter(q=>q.t>=h.pendingSince);
          const ax=axisMean(support);
          if(ax&&ax.resultant>=.9){target(ax.deg);h.source='producer axis (consistent temporal support)';}
        }
      }
    }else clear(); // Hold the last qualified axis; never substitute 0 or legacy 90.
    h.axisSupport=produced.length;
    // Only aligned, strong producer evidence can locate the front of a vehicle.
    // Motion provides travel direction, which differs from front direction in reverse.
    const aligned=finite(h.yaw)?produced.filter(q=>Math.abs(axisDelta(h.yaw,q.yaw))<=p.vehicleAxisConsistencyDeg):[];
    const vote=aligned.length?aligned.reduce((s,q)=>s+Math.cos((q.yaw-h.yaw)*Math.PI/180),0)/aligned.length:0;
    h.directionVote=vote;h.directionSource='producer front/back vote';
    h.known=fresh&&aligned.length>=3&&aligned.length>=produced.length*.75&&Math.abs(vote)>=.8;
    h.directionYaw=h.known?wrap(h.yaw+(vote<0?180:0)):null;
    // Holding retains the active interpolation, instead of truncating it at each update.
    t.heading=h;
  }
  // ---- association -------------------------------------------------------
  predictAt(t,ms){return t.motion==='moving'?{x:t.est.x+t.est.vx*clamp(ms-t.est.t,0,500)/1000,y:t.est.y+t.est.vy*clamp(ms-t.est.t,0,500)/1000}:{...t.anchor};}
  cost(t,c,ms,gate){if(t.cls!==c.cls)return null;const p=this.predictAt(t,ms),l=t.hist[t.hist.length-1];
    const d=Math.hypot(p.x-c.x,p.y-c.y),dl=Math.hypot(l.x-c.x,l.y-c.y);if(Math.min(d,dl)>gate)return null;
    if(Math.min(t.dims.l,c.dims.l)/Math.max(t.dims.l,c.dims.l)<.5)return null;
    const box=t.box&&c.box?1-M.overlap(t.box,c.box).iou:.5;
    const yaw=t.heading&&t.heading.known&&finite(c.yaw)?Math.abs(t.cls==='Vehicle'?axisDelta(t.heading.yaw,c.yaw):delta(t.heading.yaw,c.yaw))/180:0;
    return .7*d+.3*dl+.4*box+.3*yaw;}
  assign(tracks,cands,ms,gate,onPair){if(!tracks.length||!cands.length)return;
    const big=1e4,matrix=tracks.map(t=>[...cands.map(c=>{const g=gate[t.cls],v=this.cost(t,c,ms,g);return v===null?big:v;}),...tracks.map(()=>5)]);
    const pairs=M.assignment(matrix);pairs.forEach((j,i)=>{if(j>=0&&j<cands.length&&matrix[i][j]<5)onPair(tracks[i],cands[j]);});}
  // ---- event ingestion ---------------------------------------------------
  observe(e){
    if(!e||!Array.isArray(e.objects)||!finite(e.capture_ms)||!finite(e.arrival_ms)||!e.source||!e.session)return this.reject('INVALID_EVENT');
    if(this.source&&e.source!==this.source)return this.reject('OTHER_SOURCE');
    if(this.session!==null&&this.session!==e.session)this.reset(this.source);
    this.session=e.session;if(!this.source)this.source=e.source;
    const old=this.latest;
    if(old&&(e.frame_id===old.frame_id||e.capture_ms===old.capture_ms))return this.reject('DUPLICATE');
    if(old&&(e.frame_id<old.frame_id||e.capture_ms<old.capture_ms))return this.reject('OUT_OF_ORDER');
    const reason=R.poseReason(e),p=this.policy,ms=e.capture_ms,ta=e.arrival_ms;this.stats.events++;
    const age=sameHostWall(e)&&!reason?ta-ms:null;
    if(!reason)this.ego.observe(e.source,{x:e.pose.x,y:e.pose.y,yaw:e.pose.yaw,measuredMs:ms,sampleId:e.frame_id,className:'Vehicle',ageAtArrivalMs:age},ta);
    else delete this.ego.states[e.source];
    this.latest=e;
    // End tracks whose last measurement is older than expiry (capture clock).
    for(const [id,t] of this.tracks)if(ms-t.lastT>=p.expiryMs){this.end(t,'EXPIRED',ms);this.tracks.delete(id);}
    const dispositions=e.objects.map(o=>({entry:o,raw:o.raw,disposition:null,track:null}));
    const cands=[];
    dispositions.forEach((d,i)=>{const raw=d.raw,xy=raw&&raw.location;
      if(!raw||!xy||!finite(xy.x)||!finite(xy.y)){d.disposition='INVALID_MEASUREMENT';return;}
      if(reason){d.disposition='POSE_UNAVAILABLE:'+reason;return;}
      const region=R.captureRegion(raw,e.pose);if(region){d.disposition='OUTSIDE_CAPTURE_REGION:'+region;return;}
      cands.push({i,d,cls:cls(raw.type),x:xy.x,y:xy.y,score:raw.score,strong:isStrong(raw),dims:candidateDims(raw),
        yaw:o_yaw(e.objects[i]),box:M.box(e.objects[i])});});
    const used=new Set(),updated=new Map();
    const take=(t,c,how)=>{used.add(c);updated.set(t,{c,how});};
    const est=[...this.tracks.values()].filter(t=>t.state==='established'),tent=[...this.tracks.values()].filter(t=>t.state==='tentative');
    const strong=cands.filter(c=>c.strong);
    // Pass A: established tracks <- strong; B: remaining established <- weak (score dips
    // sustain identity, never create it); C: tentative <- remaining strong.
    this.assign(est,strong,ms,p.gate,(t,c)=>take(t,c,'ESTABLISHED_UPDATE'));
    this.assign(est.filter(t=>!updated.has(t)),cands.filter(c=>!c.strong),ms,p.weakGate,(t,c)=>take(t,c,'ESTABLISHED_WEAK_UPDATE'));
    this.assign(tent,strong.filter(c=>!used.has(c)),ms,p.gate,(t,c)=>take(t,c,'TENTATIVE_UPDATE'));
    for(const [t,{c,how}] of updated){const alive=this.update(t,c,e,c.strong);c.d.disposition=alive?how:'TRACK_ENDED_WEAK_ONLY';c.d.track=t;}
    for(const t of this.tracks.values())if(!updated.has(t))this.miss(t,ms);
    // Unassigned strong candidates: footprint exclusion, then new tentative tracks.
    const footprintOwner=c=>{for(const t of this.tracks.values()){if(t.cls!==c.cls||ms-t.lastT>=p.expiryMs)continue;
      const centre=this.predictAt(t,ms),yaw=t.heading&&finite(t.heading.yaw)?t.heading.yaw:null;
      if(insideFootprint(c,centre,t.dims,yaw)||insideFootprint(centre,c,c.dims,c.yaw))return t;}return null;};
    for(const c of strong.filter(c=>!used.has(c)).sort((a,b)=>b.score-a.score||a.d.raw.id.localeCompare(b.d.raw.id))){
      const owner=footprintOwner(c);
      if(owner){c.d.disposition='OVERLAPS_TRACK_FOOTPRINT';c.d.track=owner;owner.absorbed.push(c.d.raw.id);this.stats.absorbed++;continue;}
      const t=this.create(c,e);c.d.disposition='NEW_TENTATIVE';c.d.track=t;
    }
    this.markConflicts(ms);
    // Weak leftovers: explained by a track footprint, else unresolved hypotheses.
    this.decayField(ta);
    for(const c of cands.filter(c=>!c.strong&&!used.has(c))){
      const owner=footprintOwner(c);
      if(owner){c.d.disposition='WEAK_INSIDE_TRACK_FOOTPRINT';c.d.track=owner;owner.absorbed.push(c.d.raw.id);this.stats.absorbed++;continue;}
      c.d.disposition='LOW_CONFIDENCE_HYPOTHESIS';this.stats.hypotheses++;
      const k=Math.floor(c.x/p.fieldCellM)+','+Math.floor(c.y/p.fieldCellM),cell=this.field.get(k)||{w:0,t:ta,n:0,ids:[]};
      cell.w=Math.min(4,cell.w+1);cell.t=ta;cell.n++;cell.ids=[...cell.ids.slice(-7),c.d.raw.id];cell.cls=c.cls;this.field.set(k,cell);
    }
    while(this.field.size>p.fieldMaxCells)this.field.delete(this.field.keys().next().value);
    while(this.tracks.size>p.maxTracks){const [id,t]=this.tracks.entries().next().value;this.end(t,'CAPACITY',ms);this.tracks.delete(id);}
    this.lastDispositions=dispositions;return reason||'ACCEPTED';
  }
  // Two tracks whose expanded footprints contain each other's centre cannot both be
  // physical objects. They are shown together as one unresolved hypothesis group;
  // a tentative track in conflict cannot be confirmed. Nothing is merged or averaged.
  conflictsEstablished(t,ms){const p=this.predictAt(t,ms);for(const o of this.tracks.values()){if(o===t||o.cls!==t.cls||o.state!=='established'||ms-o.lastT>=this.policy.expiryMs)continue;
      const q=this.predictAt(o,ms),yo=o.heading&&finite(o.heading.yaw)?o.heading.yaw:null,yt=t.heading&&finite(t.heading.yaw)?t.heading.yaw:null;
      if(insideFootprint(p,q,o.dims,yo)||insideFootprint(q,p,t.dims,yt))return true;}return false;}
  markConflicts(ms){const list=[...this.tracks.values()].filter(t=>ms-t.lastT<this.policy.expiryMs);
    for(const t of list)t.conflict=null;let g=0;
    for(let i=0;i<list.length;i++)for(let j=i+1;j<list.length;j++){const a=list[i],b=list[j];if(a.cls!==b.cls)continue;
      const pa=this.predictAt(a,ms),pb=this.predictAt(b,ms),ya=a.heading&&finite(a.heading.yaw)?a.heading.yaw:null,yb=b.heading&&finite(b.heading.yaw)?b.heading.yaw:null;
      if(!insideFootprint(pb,pa,a.dims,ya)&&!insideFootprint(pa,pb,b.dims,yb))continue;
      const id=a.conflict||b.conflict||('conflict_'+(++g));a.conflict=b.conflict=id;}
  }
  decayField(ta){for(const [k,c] of this.field){c.w*=Math.exp(-(ta-c.t)/this.policy.fieldTauMs);c.t=ta;if(c.w<.03)this.field.delete(k);}}
  create(c,e){const t={id:'tentative_'+(++this.nextTentative),displayId:null,cls:c.cls,state:'tentative',
      bornT:e.capture_ms,bornArrival:e.arrival_ms,confirmedT:null,hist:[],hits:[],weakRun:0,misses:0,lastT:e.capture_ms,
      lastArrival:e.arrival_ms,motion:'init',anchor:{x:c.x,y:c.y},est:{x:c.x,y:c.y,vx:0,vy:0,t:e.capture_ms},speed:0,stillCount:0,
      heading:null,dims:c.dims,box:c.box,bridge:null,entry:null,event:null,absorbed:[],updates:0};
    this.tracks.set(t.id,t);this.stats.tentativeCreated++;this.update(t,c,e,true,true);return t;}
  update(t,c,e,strong,fresh=false){
    const p=this.policy,before=fresh?null:this.display(t,e.arrival_ms);
    t.hist.push({t:e.capture_ms,x:c.x,y:c.y,score:c.score,strong,yaw:c.yaw,l:c.dims.l,w:c.dims.w,sx:e.pose.sensor.x,sy:e.pose.sensor.y});
    if(t.hist.length>p.historyMax)t.hist.shift();
    if(!fresh)t.hits.push(strong);if(t.hits.length>p.confirmWindow)t.hits.shift();
    if(fresh)t.hits=[true];
    t.weakRun=strong?0:t.weakRun+1;t.misses=0;t.lastT=e.capture_ms;t.lastArrival=e.arrival_ms;t.entry=c.d.entry;t.event=e;t.box=c.box||t.box;t.absorbed=[];t.updates++;
    const sized=t.hist.filter(h=>h.strong);t.dims={l:median((sized.length?sized:t.hist).map(h=>h.l)),w:median((sized.length?sized:t.hist).map(h=>h.w))};
    this.fitMotion(t);this.updateHeading(t,e.arrival_ms);
    if(t.state==='tentative'&&t.hits.filter(Boolean).length>=p.confirmHits&&!this.conflictsEstablished(t,e.capture_ms)){t.state='established';t.displayId='T'+(++this.nextEstablished);
      t.confirmedT=e.capture_ms;t.confirmedArrival=e.arrival_ms;this.stats.established++;
      this.stats.confirmCaptureMs.push(e.capture_ms-t.bornT);this.stats.confirmArrivalMs.push(e.arrival_ms-t.bornArrival);}
    if(t.state==='established'&&t.weakRun>=p.weakOnlyMaxFrames){this.end(t,'WEAK_ONLY',e.capture_ms);this.tracks.delete(t.id);return false;}
    if(before){const after=this.display({...t,bridge:null},e.arrival_ms),ox=before.x-after.x,oy=before.y-after.y,m=Math.hypot(ox,oy);
      if(m>p.snapM){t.bridge=null;this.stats.snaps++;}else{t.bridge={x:ox,y:oy,t0:e.arrival_ms};this.stats.bridges.push(m);if(this.stats.bridges.length>8192)this.stats.bridges.shift();}}
    return true;
  }
  miss(t,ms){t.misses++;if(t.state==='tentative'){t.hits.push(false);if(t.hits.length>this.policy.confirmWindow)t.hits.shift();
      if(t.misses>=this.policy.tentativeMaxMisses){this.end(t,'TENTATIVE_DROPPED',ms);this.tracks.delete(t.id);}}}
  end(t,why,ms){if(t.state==='established'){this.stats.establishedLifetimeMs.push(t.lastT-t.confirmedT);
      if(why==='WEAK_ONLY')this.stats.endedWeakOnly++;else this.stats.endedExpired++;}
    else this.stats.tentativeDropped++;}
  // ---- view --------------------------------------------------------------
  view(now,options={}){
    const p=this.policy,mode=options.mode||'current',radius=options.radius||45,selected=this.latest;
    const pReason=selected?R.poseReason(selected):'AWAITING_POSE';
    const ego=!pReason?this.ego.sample(this.source,now):null,sensor=ego?R.sensorAt(selected.pose,ego):null;
    const tracks=[];
    for(const t of this.tracks.values()){
      const age=now-t.lastT,pose=this.display(t,now),yaw=this.displayYaw(t,now);
      let reason=null;
      if(pReason)reason=pReason;else if(age<0)reason='FUTURE_TIMESTAMP';else if(age>=p.expiryMs)reason='EXPIRED';
      else if(t.state==='tentative'&&age>p.freshMs)reason='TENTATIVE_NOT_CURRENT';
      if(!reason&&sensor)reason=R.region(pose,sensor,selected.pose.fov_deg,selected.pose.range_m);
      if(!reason&&ego){const a=ego.yaw*Math.PI/180;if((pose.x-ego.x)*Math.cos(a)+(pose.y-ego.y)*Math.sin(a)<=0)reason='BEHIND_DISPLAY_EGO';}
      if(!reason&&ego&&Math.hypot(pose.x-ego.x,pose.y-ego.y)>radius)reason='OUTSIDE_DISPLAY_RADIUS';
      const vis=t.entry?R.visibility(t.entry,t.event):{state:'unknown',reason:'NO_QUALIFIED_VISIBILITY_EVIDENCE'};
      if(!reason&&mode==='visible'&&vis.state!=='visible')reason=vis.state==='occluded'?'OCCLUDED':'VISIBILITY_UNKNOWN';
      if(!reason&&mode==='established'&&t.state!=='established')reason='TENTATIVE_HIDDEN_BY_VIEW';
      const missed=age>p.freshMs,fade=clamp((p.expiryMs-age)/(p.expiryMs-p.fadeStartMs),0,1);
      const fadeIn=t.state==='established'?clamp((now-t.confirmedArrival)/200,0,1):1;
      tracks.push({key:t.id,track:t,id:t.displayId||t.id,cls:t.cls,tier:t.state,missed,motion:t.motion,speed:t.speed,
        pose,yaw,headingKnown:!!(t.heading&&t.heading.known),directionYaw:t.heading?.directionYaw??null,headingSource:t.heading?t.heading.source:'unknown',dims:t.dims,ageMs:age,
        alpha:fade*(t.state==='established'?.35+.65*fadeIn:1),reason,shown:!reason,visibility:vis,entry:t.entry,event:t.event,
        weakRun:t.weakRun,absorbed:t.absorbed.slice(),hist:t.hist,conflict:t.conflict||null,motionUncertainty:t.motionUncertainty||null});
    }
    const field=[];
    for(const [k,c] of this.field){const w=c.w*Math.exp(-(now-c.t)/p.fieldTauMs);if(w<p.fieldMinWeight)continue;
      const [ix,iy]=k.split(',').map(Number),x=(ix+.5)*p.fieldCellM,y=(iy+.5)*p.fieldCellM;
      let reason=null;if(ego&&Math.hypot(x-ego.x,y-ego.y)>radius+p.fieldCellM)reason='OUTSIDE_DISPLAY_RADIUS';
      if(mode==='visible'||mode==='established')reason='HYPOTHESES_HIDDEN_BY_VIEW';
      field.push({key:k,x,y,w,n:c.n,ids:c.ids,cls:c.cls,reason,shown:!reason});}
    const d=this.lastDispositions,count=f=>d.filter(f).length;
    const counts={raw:d.length,strong:count(x=>x.raw&&isStrong(x.raw)),weak:count(x=>x.raw&&!isStrong(x.raw)),
      established:tracks.filter(r=>r.shown&&r.tier==='established'&&!r.missed).length,
      missed:tracks.filter(r=>r.shown&&r.tier==='established'&&r.missed).length,
      tentative:tracks.filter(r=>r.shown&&r.tier==='tentative').length,
      hypotheses:count(x=>x.disposition==='LOW_CONFIDENCE_HYPOTHESIS'),
      absorbed:count(x=>x.disposition==='OVERLAPS_TRACK_FOOTPRINT'||x.disposition==='WEAK_INSIDE_TRACK_FOOTPRINT'),
      conflicted:tracks.filter(r=>r.shown&&r.conflict).length,
      outside:count(x=>String(x.disposition).startsWith('OUTSIDE_CAPTURE_REGION'))};
    return {ego,sensor,selected,poseReason:pReason,tracks,visible:tracks.filter(r=>r.shown),field,dispositions:d,counts,
      clockMode:selected&&sameHostWall(selected)?'capture':'arrival',
      egoStale:!selected||now-selected.arrival_ms>=p.expiryMs};
  }
}
// Recorded replay and live view both stamp capture and arrival on the same host wall clock.
function sameHostWall(e){return e.age_mapping==='RECORDED_SAME_HOST_WALL'||e.age_mapping==='LIVE_SAME_HOST_WALL';}
function o_yaw(entry){return entry&&entry.heading_valid&&finite(entry.heading_yaw_deg)?entry.heading_yaw_deg:null;}
const api={POLICY,TrackEngine,isStrong,insideFootprint,circularMean,axisMean,axisDelta,candidateDims};
root.ScenesenseReplayTracks=api;if(node)module.exports=api;
})(typeof globalThis!=='undefined'?globalThis:this);
