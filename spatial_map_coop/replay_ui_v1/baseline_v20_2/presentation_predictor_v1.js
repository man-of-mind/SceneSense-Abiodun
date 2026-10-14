// Presentation-only display math for the cooperative spatial-map viewer.
//
// SCOPE CONTRACT
// --------------
// Everything in this file is PRESENTATION. It exists so that a 10 Hz measured
// map reads smoothly on a 60 Hz display. It must never:
//   * modify a measured world coordinate that is reported anywhere,
//   * influence association, track state, counters, or retained evidence,
//   * mutate the snapshot objects handed to it, or
//   * invent a detection that the map service did not report.
// It returns new plain objects; callers draw those and discard them.
//
// The file is served inline by spatial_map_server_moving_ego.py's viewer route
// and is also loadable under node for the offline tests, so the tested code and
// the served code are byte-identical.

(function (root) {
  'use strict';

  // ---- registered presentation constants -----------------------------------
  // Bounded causal extrapolation horizon. Past this the display HOLDS instead
  // of drifting, so a lost UE never slides an object across the map.
  var PREDICTION_HORIZON_MS = 200;
  // Residual decay used to absorb the correction when a measurement arrives.
  var CORRECTION_MS = 120;
  // Two samples further apart than this do not define a usable velocity.
  var MAX_SAMPLE_GAP_MS = 500;
  // A measured clock that jumps backwards by more than this is a restart.
  var BACKWARD_RESET_MS = 1000;
  // Conservative class speed caps (m/s) for display extrapolation only. These
  // are display guards, not measured or calibrated kinematic bounds.
  var MAX_DISPLAY_SPEED_MPS = { Vehicle: 25.0, Pedestrian: 4.0, Cyclist: 12.0 };
  var DEFAULT_MAX_DISPLAY_SPEED_MPS = 15.0;
  var MAX_DISPLAY_YAW_RATE_DPS = 60.0;
  // Lane-guided display yaw acceptance radius and index cell size. The cell
  // size must be >= the acceptance radius for the 3x3 neighbourhood query to
  // be exact (see RoadHeadingIndex).
  var ROAD_YAW_MAX_DISTANCE_M = 8.0;
  // Cell size equals the acceptance radius: with cell >= radius the 3x3
  // neighbourhood is provably exact, and the smallest legal cell keeps the
  // candidate count per query minimal.
  var ROAD_INDEX_CELL_M = 8.0;

  var CLASS_ALIASES = {
    vehicle: 'Vehicle',
    movingvehicle: 'Vehicle',
    parkedvehicle: 'Vehicle',
    car: 'Vehicle',
    person: 'Pedestrian',
    pedestrian: 'Pedestrian',
    walker: 'Pedestrian',
    cyclist: 'Cyclist',
    bicycle: 'Cyclist'
  };

  function canonicalDisplayClass(value) {
    var lowered = String(value === null || value === undefined ? '' : value)
      .trim().toLowerCase().replace(/_/g, '');
    return CLASS_ALIASES[lowered] || null;
  }

  function maxDisplaySpeed(className) {
    var canonical = canonicalDisplayClass(className);
    if (canonical !== null && Object.prototype.hasOwnProperty.call(MAX_DISPLAY_SPEED_MPS, canonical)) {
      return MAX_DISPLAY_SPEED_MPS[canonical];
    }
    return DEFAULT_MAX_DISPLAY_SPEED_MPS;
  }

  // Smoothstep; only used to decay the correction residual.
  function ease(t) { return t * t * (3.0 - 2.0 * t); }

  function angleDeltaDeg(from, to) {
    return ((to - from + 180.0) % 360.0 + 360.0) % 360.0 - 180.0;
  }

  function isFiniteNumber(value) {
    return typeof value === 'number' && isFinite(value);
  }

  function finiteOr(value, fallback) {
    var numeric = Number(value);
    return isFiniteNumber(numeric) ? numeric : fallback;
  }

  // Absent means absent. Number(null) is 0 and Number('') is 0, so a missing
  // measured timestamp must be detected BEFORE any numeric coercion or it turns
  // into a real time of zero and the predictor starts inventing velocities.
  function optionalMs(value) {
    if (value === null || value === undefined || value === '') { return null; }
    var numeric = Number(value);
    return isFiniteNumber(numeric) ? numeric : null;
  }

  // Clamp a velocity vector's MAGNITUDE, preserving its direction. Matches
  // render_action50_before_after_v2._clamped_velocity.
  function clampVelocity(vx, vy, maximum) {
    var speed = Math.sqrt(vx * vx + vy * vy);
    if (!isFiniteNumber(speed) || speed <= 0.0) { return [0.0, 0.0]; }
    if (speed <= maximum) { return [vx, vy]; }
    var gain = maximum / speed;
    return [vx * gain, vy * gain];
  }

  /**
   * Causal constant-velocity display predictor.
   *
   * Velocity is derived ONLY from the last two accepted measured samples, using
   * the measured capture clock so the magnitude is in real m/s. The
   * By default the horizon counts local elapsed time since arrival. A validated
   * adapter may additionally supply ageAtArrivalMs, computed within a declared
   * clock domain. The replay UI supplies recorded installation minus capture;
   * it never subtracts browser time from CARLA time. Optional velocity-consistency
   * and correction-distance guards are disabled for existing callers.
   *
   * No future measurement is ever consulted: observe() only ever sees samples
   * that have already arrived, and sample() is a pure function of the state
   * that observe() left behind.
   */
  function DisplayPredictor(options) {
    var opts = options || {};
    this.horizonMs = finiteOr(opts.horizonMs, PREDICTION_HORIZON_MS);
    this.correctionMs = finiteOr(opts.correctionMs, CORRECTION_MS);
    this.maxSampleGapMs = finiteOr(opts.maxSampleGapMs, MAX_SAMPLE_GAP_MS);
    this.backwardResetMs = finiteOr(opts.backwardResetMs, BACKWARD_RESET_MS);
    this.stationarySpeedMps = finiteOr(opts.stationarySpeedMps, 0);
    this.teleportDistanceM = optionalMs(opts.teleportDistanceM);
    this.maxCorrectionM = optionalMs(opts.maxCorrectionM);
    this.velocityConsistencyMps = optionalMs(opts.velocityConsistencyMps);
    this.states = Object.create(null);
  }

  DisplayPredictor.prototype._reset = function (key, sample, nowMs) {
    var state = {
      className: sample.className === undefined ? null : sample.className,
      measuredMs: optionalMs(sample.measuredMs),
      sampleId: sample.sampleId === undefined ? null : sample.sampleId,
      x: sample.x, y: sample.y, yaw: sample.yaw,
      vx: 0.0, vy: 0.0, yawRateDps: 0.0,
      rawVx: null, rawVy: null,
      arrivalMs: nowMs,
      // Only a validated adapter may supply a same-clock measurement age.
      // Absent means arrival-relative animation, never browser minus CARLA.
      ageAtArrivalMs: optionalMs(sample.ageAtArrivalMs),
      errX: 0.0, errY: 0.0, errYaw: 0.0, errStartMs: nowMs,
      acceptedCount: 1, rejectedCount: 0
    };
    this.states[key] = state;
    return state;
  };

  /**
   * Offer one already-received measurement.
   *
   * sample = {x, y, yaw, className, measuredMs, sampleId, ageAtArrivalMs?}
   *   measuredMs  measured capture time in ms (may be null/undefined/NaN)
   *   sampleId    opaque per-measurement identity used for change detection
   *               when no measured timestamp exists (e.g. a frame counter)
   *
   * Returns 'CREATED' | 'ACCEPTED' | 'REPEATED' | 'OUT_OF_ORDER' | 'RESET' |
   *         'INVALID'.
   */
  DisplayPredictor.prototype.observe = function (key, sample, nowMs) {
    var x = Number(sample.x), y = Number(sample.y);
    if (!isFiniteNumber(x) || !isFiniteNumber(y)) { return 'INVALID'; }
    var yaw = finiteOr(sample.yaw, 0.0);
    var measuredMs = optionalMs(sample.measuredMs);
    var sampleId = sample.sampleId === undefined ? null : sample.sampleId;
    var normalized = {
      x: x, y: y, yaw: yaw, className: sample.className,
      measuredMs: measuredMs, sampleId: sampleId,
      ageAtArrivalMs: optionalMs(sample.ageAtArrivalMs)
    };
    if (normalized.ageAtArrivalMs !== null && normalized.ageAtArrivalMs < 0) { return 'INVALID'; }

    var state = this.states[key];
    if (!state) { this._reset(key, normalized, nowMs); return 'CREATED'; }
    state.className = sample.className === undefined ? state.className : sample.className;

    // ---- is this actually a new measurement? ------------------------------
    if (measuredMs !== null && state.measuredMs !== null) {
      if (measuredMs === state.measuredMs) { state.rejectedCount += 1; return 'REPEATED'; }
      if (measuredMs < state.measuredMs) {
        if (state.measuredMs - measuredMs >= this.backwardResetMs) {
          // Measured clock restarted: adopt the sample with zero velocity
          // rather than extrapolating across the discontinuity.
          this._reset(key, normalized, nowMs);
          return 'RESET';
        }
        state.rejectedCount += 1;
        return 'OUT_OF_ORDER';
      }
    } else if (sampleId !== null || state.sampleId !== null) {
      // No usable measured clock: fall back to opaque sample identity so a
      // repeated poll of the same measurement is not treated as motion.
      if (sampleId === state.sampleId) { state.rejectedCount += 1; return 'REPEATED'; }
    }

    // ---- velocity from the last two measured samples only -----------------
    if (this.teleportDistanceM !== null &&
        Math.hypot(x - state.x, y - state.y) > this.teleportDistanceM) {
      this._reset(key, normalized, nowMs); return 'RESET';
    }
    var vx = 0.0, vy = 0.0, yawRateDps = 0.0, rawVx = null, rawVy = null;
    if (measuredMs !== null && state.measuredMs !== null) {
      var dtMs = measuredMs - state.measuredMs;
      if (dtMs > 0.0 && dtMs <= this.maxSampleGapMs) {
        var dtS = dtMs / 1000.0;
        var clamped = clampVelocity(
          (x - state.x) / dtS, (y - state.y) / dtS, maxDisplaySpeed(state.className)
        );
        vx = clamped[0];
        vy = clamped[1];
        rawVx = vx; rawVy = vy;
        if (this.velocityConsistencyMps !== null) {
          // Two consistent measured displacement intervals are needed before
          // predicting noisy, untracked detector candidates. No GT identity.
          if (state.rawVx === null || Math.hypot(vx - state.rawVx, vy - state.rawVy) > this.velocityConsistencyMps) {
            vx = 0; vy = 0;
          } else {
            vx = (vx + state.rawVx) / 2; vy = (vy + state.rawVy) / 2;
          }
          // A stationary latest interval stops prediction immediately.
          if (Math.hypot(rawVx, rawVy) < this.stationarySpeedMps) { vx = 0; vy = 0; }
        }
        if (Math.hypot(vx, vy) < this.stationarySpeedMps) { vx = 0; vy = 0; }
        yawRateDps = Math.max(
          -MAX_DISPLAY_YAW_RATE_DPS,
          Math.min(MAX_DISPLAY_YAW_RATE_DPS, angleDeltaDeg(state.yaw, yaw) / dtS)
        );
      }
    }

    // ---- adopt the measurement, keeping the drawn pose continuous ----------
    var before = this.sample(key, nowMs);
    var ageSeconds = Math.min(normalized.ageAtArrivalMs || 0, this.horizonMs) / 1000;
    if (this.maxCorrectionM !== null &&
        Math.hypot(before.x - (x + vx * ageSeconds), before.y - (y + vy * ageSeconds)) > this.maxCorrectionM) {
      // A large residual is not evidence of a traversed path. Let the caller
      // identify a display reset instead of sweeping a glyph across the scene.
      this._reset(key, normalized, nowMs); return 'RESET';
    }
    state.errX = before.x - (x + vx * ageSeconds);
    state.errY = before.y - (y + vy * ageSeconds);
    state.errYaw = angleDeltaDeg(yaw + yawRateDps * ageSeconds, before.yaw);
    state.errStartMs = nowMs;
    state.x = x; state.y = y; state.yaw = yaw;
    state.measuredMs = measuredMs;
    state.sampleId = sampleId;
    state.vx = vx; state.vy = vy; state.yawRateDps = yawRateDps;
    state.rawVx = rawVx; state.rawVy = rawVy;
    state.arrivalMs = nowMs;
    state.ageAtArrivalMs = normalized.ageAtArrivalMs;
    state.acceptedCount += 1;
    return 'ACCEPTED';
  };

  /**
   * Pose to draw at local time nowMs. Pure: never mutates predictor state.
   *
   *   t        = clamp(nowMs - arrivalMs + (ageAtArrivalMs || 0), 0, horizonMs) / 1000
   *   x_disp   = x_meas + vx * t + errX * (1 - ease(u))
   *   y_disp   = y_meas + vy * t + errY * (1 - ease(u))
   *   yaw_disp = yaw_meas + yawRate * t + errYaw * (1 - ease(u))
   *   u        = clamp((nowMs - errStartMs) / correctionMs, 0, 1)
   */
  DisplayPredictor.prototype.sample = function (key, nowMs) {
    var state = this.states[key];
    if (!state) { return null; }
    var elapsedMs = nowMs - state.arrivalMs + (state.ageAtArrivalMs || 0);
    var boundedMs = Math.min(Math.max(0.0, elapsedMs), this.horizonMs);
    var t = boundedMs / 1000.0;
    var weight = 0.0;
    if (this.correctionMs > 0.0) {
      var u = Math.min(1.0, Math.max(0.0, (nowMs - state.errStartMs) / this.correctionMs));
      weight = 1.0 - ease(u);
    }
    return {
      x: state.x + state.vx * t + state.errX * weight,
      y: state.y + state.vy * t + state.errY * weight,
      yaw: state.yaw + state.yawRateDps * t + state.errYaw * weight,
      vx: state.vx,
      vy: state.vy,
      predictionMs: boundedMs,
      ageMs: state.ageAtArrivalMs === null ? null : elapsedMs,
      clockMode: state.ageAtArrivalMs === null ? 'ARRIVAL_RELATIVE' : 'MAPPED_CAPTURE',
      // true once local time has passed the horizon: the pose is held, not
      // extrapolated any further.
      held: elapsedMs > this.horizonMs
    };
  };

  /** Drop predictor state for keys absent from the current inventory. */
  DisplayPredictor.prototype.retain = function (keys) {
    var keep = Object.create(null), index;
    for (index = 0; index < keys.length; index += 1) { keep[keys[index]] = true; }
    var present = Object.keys(this.states);
    for (index = 0; index < present.length; index += 1) {
      if (!keep[present[index]]) { delete this.states[present[index]]; }
    }
  };

  DisplayPredictor.prototype.size = function () { return Object.keys(this.states).length; };

  /**
   * Uniform-grid index over road segments for lane-guided DISPLAY YAW only.
   *
   * Replaces a full scan of every Town10HD segment per object per animation
   * frame. Exactness: if the closest point of some segment S to the query q is
   * within ROAD_YAW_MAX_DISTANCE_M, that point lies inside S's bounding box and
   * within one cell of q, so with cellSizeM >= maxDistanceM the 3x3
   * neighbourhood around q necessarily contains S. Candidates are visited in
   * ascending flattened-segment order with a strict `<` comparison, which
   * reproduces the first-wins tie-breaking of the original linear scan, so the
   * returned heading is identical to the unindexed result.
   */
  function RoadHeadingIndex(roads, options) {
    var opts = options || {};
    this.maxDistanceM = finiteOr(opts.maxDistanceM, ROAD_YAW_MAX_DISTANCE_M);
    this.cellSizeM = Math.max(finiteOr(opts.cellSizeM, ROAD_INDEX_CELL_M), this.maxDistanceM);
    this.ax = []; this.ay = []; this.dx = []; this.dy = []; this.len2 = [];
    this.cells = Object.create(null);

    var list = roads || [];
    for (var r = 0; r < list.length; r += 1) {
      var polyline = list[r] || [];
      for (var i = 0; i + 1 < polyline.length; i += 1) {
        var ax = Number(polyline[i][0]), ay = Number(polyline[i][1]);
        var dx = Number(polyline[i + 1][0]) - ax, dy = Number(polyline[i + 1][1]) - ay;
        var len2 = dx * dx + dy * dy;
        // Identical degenerate-segment rejection to the original scan.
        if (!(len2 >= 1e-9)) { continue; }
        if (!isFiniteNumber(ax) || !isFiniteNumber(ay) || !isFiniteNumber(len2)) { continue; }
        var index = this.ax.length;
        this.ax.push(ax); this.ay.push(ay);
        this.dx.push(dx); this.dy.push(dy); this.len2.push(len2);
        this._register(index, ax, ay, ax + dx, ay + dy);
      }
    }
    this.segmentCount = this.ax.length;
    // Freeze into typed arrays: the hot loop then reads contiguous memory.
    this.ax = Float64Array.from(this.ax);
    this.ay = Float64Array.from(this.ay);
    this.dx = Float64Array.from(this.dx);
    this.dy = Float64Array.from(this.dy);
    this.len2 = Float64Array.from(this.len2);
    var keys = Object.keys(this.cells);
    for (var c = 0; c < keys.length; c += 1) {
      this.cells[keys[c]] = Int32Array.from(this.cells[keys[c]]);
    }
  }

  RoadHeadingIndex.prototype._cellOf = function (value) {
    return Math.floor(value / this.cellSizeM);
  };

  RoadHeadingIndex.prototype._register = function (index, x0, y0, x1, y1) {
    var ix0 = this._cellOf(Math.min(x0, x1)), ix1 = this._cellOf(Math.max(x0, x1));
    var iy0 = this._cellOf(Math.min(y0, y1)), iy1 = this._cellOf(Math.max(y0, y1));
    for (var ix = ix0; ix <= ix1; ix += 1) {
      for (var iy = iy0; iy <= iy1; iy += 1) {
        var key = ix + ',' + iy;
        var bucket = this.cells[key];
        if (bucket === undefined) { this.cells[key] = [index]; } else { bucket.push(index); }
      }
    }
  };

  /** Nearest road tangent in radians, or null when nothing is close enough.
   *
   * Allocation-free: the nine candidate buckets are walked in place. Ties are
   * broken toward the LOWEST flattened segment index, which reproduces exactly
   * what the original ascending-order scan with a strict `<` produced, so no
   * sort or dedup pass is needed and a segment registered in several cells can
   * be visited twice harmlessly.
   */
  RoadHeadingIndex.prototype.headingRad = function (x, y) {
    if (this.segmentCount === 0 || !isFiniteNumber(x) || !isFiniteNumber(y)) { return null; }
    var cx = this._cellOf(x), cy = this._cellOf(y);
    var ax = this.ax, ay = this.ay, dx = this.dx, dy = this.dy, len2 = this.len2;
    var bestD2 = Infinity, bestIndex = -1;
    for (var ix = cx - 1; ix <= cx + 1; ix += 1) {
      for (var iy = cy - 1; iy <= cy + 1; iy += 1) {
        var bucket = this.cells[ix + ',' + iy];
        if (bucket === undefined) { continue; }
        for (var k = 0; k < bucket.length; k += 1) {
          var index = bucket[k];
          var sx = ax[index], sy = ay[index], vx = dx[index], vy = dy[index];
          var t = ((x - sx) * vx + (y - sy) * vy) / len2[index];
          if (t < 0.0) { t = 0.0; } else if (t > 1.0) { t = 1.0; }
          var ox = x - (sx + t * vx), oy = y - (sy + t * vy);
          var d2 = ox * ox + oy * oy;
          if (d2 < bestD2 || (d2 === bestD2 && index < bestIndex)) {
            bestD2 = d2;
            bestIndex = index;
          }
        }
      }
    }
    if (bestIndex < 0 || bestD2 > this.maxDistanceM * this.maxDistanceM) { return null; }
    return Math.atan2(dy[bestIndex], dx[bestIndex]);
  };

  RoadHeadingIndex.prototype.stats = function () {
    var keys = Object.keys(this.cells), total = 0;
    for (var i = 0; i < keys.length; i += 1) { total += this.cells[keys[i]].length; }
    return {
      segmentCount: this.segmentCount,
      cellCount: keys.length,
      cellSizeM: this.cellSizeM,
      maxDistanceM: this.maxDistanceM,
      registrations: total
    };
  };

  root.ScenesenseMapPresentation = {
    PREDICTION_HORIZON_MS: PREDICTION_HORIZON_MS,
    CORRECTION_MS: CORRECTION_MS,
    MAX_SAMPLE_GAP_MS: MAX_SAMPLE_GAP_MS,
    BACKWARD_RESET_MS: BACKWARD_RESET_MS,
    MAX_DISPLAY_SPEED_MPS: MAX_DISPLAY_SPEED_MPS,
    DEFAULT_MAX_DISPLAY_SPEED_MPS: DEFAULT_MAX_DISPLAY_SPEED_MPS,
    MAX_DISPLAY_YAW_RATE_DPS: MAX_DISPLAY_YAW_RATE_DPS,
    ROAD_YAW_MAX_DISTANCE_M: ROAD_YAW_MAX_DISTANCE_M,
    ROAD_INDEX_CELL_M: ROAD_INDEX_CELL_M,
    DisplayPredictor: DisplayPredictor,
    RoadHeadingIndex: RoadHeadingIndex,
    canonicalDisplayClass: canonicalDisplayClass,
    maxDisplaySpeed: maxDisplaySpeed,
    clampVelocity: clampVelocity,
    angleDeltaDeg: angleDeltaDeg,
    ease: ease
  };
}(typeof globalThis !== 'undefined' ? globalThis : this));

if (typeof module !== 'undefined' && module.exports) {
  module.exports = (typeof globalThis !== 'undefined' ? globalThis : this).ScenesenseMapPresentation;
}
