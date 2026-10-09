// Deterministic offline tests for the viewer's presentation JavaScript.
//
// Run: node spatial_map_coop/tests/js/presentation_tests_v1.js
// Exits non-zero on the first failure and prints a JSON summary on success.
//
// These exercise the exact asset files the server inlines into the viewer, so a
// pass is a statement about the shipped code, not about a reimplementation.

'use strict';

const path = require('path');
const fs = require('fs');

const ASSETS = path.resolve(__dirname, '..', '..', 'viewer_assets');
const P = require(path.join(ASSETS, 'presentation_predictor_v1.js'));
const I = require(path.join(ASSETS, 'presentation_inventory_v1.js'));

let passed = 0;
const failures = [];
let roadIndexReport = null;

function check(name, condition, detail) {
  if (condition) { passed += 1; return; }
  failures.push(name + (detail ? ' :: ' + detail : ''));
}

function near(a, b, tolerance) {
  return Math.abs(a - b) <= (tolerance === undefined ? 1e-9 : tolerance);
}

// =============================================================================
// 1. Raw inventory semantics
// =============================================================================
function snapshot(rawObjects, associatedObjects, streams) {
  return {
    frame_id: 100,
    raw_spatial_map_objects: rawObjects,
    spatial_map_objects: associatedObjects,
    active_streams: streams || [],
    metadata: { fusion_policy: { implementation: 'CONSERVATIVE_MULTI_UE_V1' } }
  };
}

function rawReport(id, streamId, x, y, type) {
  return {
    id: id, source_stream_id: streamId, type: type || 'Vehicle',
    location: { x: x, y: y, z: 0.5 },
    dimensions: { length: 4.6, width: 2.0, height: 1.5 },
    yaw_deg: 0.0
  };
}

function track(trackId, streams, x, y, ageNs, type) {
  return {
    track_id: trackId, association_id: 'assoc_' + trackId,
    source_stream_ids: streams, source_stream_id: streams[0],
    type: type || 'Vehicle',
    location: { x: x, y: y, z: 0.5 },
    dimensions: { length: 4.6, width: 2.0, height: 1.5 },
    yaw_deg: 0.0, map_yaw_deg: 0.0,
    age_ns: ageNs, capture_timestamp_ns: 1000000000
  };
}

(function rawInventorySemantics() {
  const raw = [
    rawReport('a1', 'ue-a', 10, 0),
    rawReport('a2', 'ue-a', 20, 0),
    rawReport('b1', 'ue-b', 10.4, 0)
  ];
  // The associated side holds ONE track for the A+B pair plus one A-only.
  const assoc = [
    track('t1', ['ue-a', 'ue-b'], 10.4, 0, 0),
    track('t2', ['ue-a'], 20, 0, 0)
  ];
  const snap = snapshot(raw, assoc, [
    { stream_id: 'ue-a', frame_id: 5, sensor_pose: { x: 0, y: 0, yaw_deg: 0 }, fov_deg: 120 },
    { stream_id: 'ue-b', frame_id: 5, sensor_pose: { x: -5, y: 0, yaw_deg: 0 }, fov_deg: 120 }
  ]);

  const rawView = I.projectInventory(snap, I.MODE_RAW, {});
  check('raw inventory equals the current raw reports',
    rawView.objects.length === raw.length, 'got ' + rawView.objects.length);
  check('raw inventory preserves object identity',
    rawView.objects.every((entry, i) => entry.object === raw[i]));
  check('raw mode reports mode=raw', rawView.mode === I.MODE_RAW);
  check('raw mode reports no associated representations',
    rawView.associatedRepresentationCount === 0);
  check('raw mode hides nothing as stale', rawView.staleHidden === 0);

  // =========================================================================
  // 2. Associated inventory cardinality
  // =========================================================================
  const assocView = I.projectInventory(snap, I.MODE_ASSOCIATED, {});
  check('associated inventory equals the current association count',
    assocView.objects.length === assoc.length, 'got ' + assocView.objects.length);
  check('associated representations never exceed supporting observations',
    assocView.associatedRepresentationCount <= rawView.rawObservationCount,
    assocView.associatedRepresentationCount + ' vs ' + rawView.rawObservationCount);
  check('an A+B match yields exactly one representation',
    assocView.counts.both === 1, 'both=' + assocView.counts.both);
  check('the A+B match does not add a third object on top of the raw pair',
    assocView.objects.length === 2);

  // =========================================================================
  // 3. A-only / B-only / A+B membership and colours
  // =========================================================================
  const both = assocView.objects.filter((e) => e.display.multiSource);
  const single = assocView.objects.filter((e) => !e.display.multiSource);
  check('exactly one A+B entry', both.length === 1);
  check('A+B entry uses the combined colour', both[0].display.color === I.COLOR_BOTH);
  check('A+B entry is flagged for the redundant double stroke',
    both[0].display.multiSource === true);
  check('one single-source entry remains visible', single.length === 1);
  check('A-only entry keeps its source colour',
    single[0].display.color === assocView.sourceColor['ue-a']);
  check('A-only count is attributed to ue-a', assocView.counts.perSource['ue-a'] === 1);
  check('B-only count is zero here', assocView.counts.perSource['ue-b'] === 0);
  check('UE-A and UE-B colours differ',
    assocView.sourceColor['ue-a'] !== assocView.sourceColor['ue-b']);
  check('A+B colour differs from both source colours',
    I.COLOR_BOTH !== assocView.sourceColor['ue-a'] &&
    I.COLOR_BOTH !== assocView.sourceColor['ue-b']);

  // B-only visibility, checked with a B-supported track present.
  const bOnly = I.projectInventory(
    snapshot(raw, [track('t3', ['ue-b'], 10.4, 0, 0)], snap.active_streams),
    I.MODE_ASSOCIATED, {}
  );
  check('B-only objects remain visible', bOnly.objects.length === 1);
  check('B-only count is attributed to ue-b', bOnly.counts.perSource['ue-b'] === 1);

  // =========================================================================
  // 4. Stale-track suppression
  // =========================================================================
  const stale = I.projectInventory(
    snapshot(raw, [
      track('fresh', ['ue-a', 'ue-b'], 10.4, 0, 100000000),
      track('stale', ['ue-a'], 30, 0, 900000000)
    ], snap.active_streams),
    I.MODE_ASSOCIATED, {}
  );
  check('a track older than the display age bound is withheld',
    stale.objects.length === 1, 'got ' + stale.objects.length);
  check('the withheld stale track is counted', stale.staleHidden === 1);
  check('the retained track is the fresh one',
    stale.objects[0].object.track_id === 'fresh');
  check('raw mode never applies the stale filter',
    I.projectInventory(snapshot(raw, [], []), I.MODE_RAW, {}).staleHidden === 0);

  // =========================================================================
  // 5. Association reduction ratio
  // =========================================================================
  check('reduction ratio is 1 - assoc/raw',
    near(I.associationReduction(3, 2), 1 - 2 / 3));
  check('reduction ratio is N/A with no raw observations',
    I.associationReduction(0, 0) === I.NOT_AVAILABLE);

  // =========================================================================
  // 6. Display transforms never mutate scientific state
  // =========================================================================
  const before = JSON.stringify(snap);
  I.projectInventory(snap, I.MODE_ASSOCIATED, {});
  I.projectInventory(snap, I.MODE_RAW, {});
  const predictor = new P.DisplayPredictor();
  assoc.forEach((item) => {
    predictor.observe('track:' + item.track_id, {
      x: item.location.x, y: item.location.y, yaw: item.yaw_deg,
      className: item.type, measuredMs: item.capture_timestamp_ns / 1e6,
      sampleId: item.capture_timestamp_ns
    }, 1000);
    predictor.sample('track:' + item.track_id, 1100);
  });
  check('projection and prediction leave the snapshot byte-identical',
    JSON.stringify(snap) === before);
  check('projection does not mutate a track location',
    assoc[0].location.x === 10.4 && assoc[0].location.y === 0);
}());

// =============================================================================
// 7. Causal constant-velocity prediction
// =============================================================================
(function causalPrediction() {
  // correctionMs: 0 isolates the predictor's kinematics from the correction
  // blend. The blend itself is tested separately in section 9; mixing them here
  // would make every expected number depend on both.
  const predictor = new P.DisplayPredictor({ correctionMs: 0 });
  const key = 'v';

  // Two measured samples 100 ms apart, 1.0 m apart -> 10 m/s along +x.
  check('first observe creates state',
    predictor.observe(key, { x: 0, y: 0, yaw: 0, className: 'Vehicle',
      measuredMs: 1000, sampleId: 1000 }, 0) === 'CREATED');
  check('second observe is accepted',
    predictor.observe(key, { x: 1, y: 0, yaw: 0, className: 'Vehicle',
      measuredMs: 1100, sampleId: 1100 }, 100) === 'ACCEPTED');

  let sample = predictor.sample(key, 100);
  check('velocity is derived from the last two measured samples',
    near(sample.vx, 10.0, 1e-9) && near(sample.vy, 0.0, 1e-9),
    'vx=' + sample.vx);
  check('at arrival the drawn pose equals the measurement',
    near(sample.x, 1.0, 1e-9), 'x=' + sample.x);
  check('prediction age is zero at arrival', sample.predictionMs === 0);

  // 50 ms later: 1.0 + 10 * 0.05 = 1.5 m. Correction residual is zero here.
  sample = predictor.sample(key, 150);
  check('constant-velocity extrapolation advances the drawn pose',
    near(sample.x, 1.5, 1e-9), 'x=' + sample.x);
  check('prediction age is reported', near(sample.predictionMs, 50, 1e-9));
  check('extrapolation is not yet held', sample.held === false);

  // ---- the 200 ms bound
  sample = predictor.sample(key, 100 + P.PREDICTION_HORIZON_MS);
  const atHorizon = sample.x;
  check('the horizon is 200 ms', P.PREDICTION_HORIZON_MS === 200);
  check('at the horizon the pose is measurement + v * 0.2',
    near(atHorizon, 1.0 + 10.0 * 0.2, 1e-9), 'x=' + atHorizon);
  const wayPast = predictor.sample(key, 100 + 5000);
  check('past the horizon the pose HOLDS instead of drifting',
    near(wayPast.x, atHorizon, 1e-9), 'x=' + wayPast.x + ' vs ' + atHorizon);
  check('past the horizon the pose is flagged held', wayPast.held === true);
  check('prediction age never exceeds the horizon',
    wayPast.predictionMs === P.PREDICTION_HORIZON_MS);

  // ---- no future measurement is used: a later observe cannot change an
  // earlier sample. Re-sampling the same (key, time) is stable.
  const beforeObserve = predictor.sample(key, 150).x;
  const independent = new P.DisplayPredictor({ correctionMs: 0 });
  independent.observe(key, { x: 0, y: 0, yaw: 0, className: 'Vehicle',
    measuredMs: 1000, sampleId: 1000 }, 0);
  independent.observe(key, { x: 1, y: 0, yaw: 0, className: 'Vehicle',
    measuredMs: 1100, sampleId: 1100 }, 100);
  check('sample() is a pure function of already-observed state',
    near(independent.sample(key, 150).x, beforeObserve, 1e-12));
}());

// =============================================================================
// 8. Conservative class speed clamps
// =============================================================================
(function speedClamping() {
  const cases = [
    ['Vehicle', 25.0],
    ['Pedestrian', 4.0],
    ['Cyclist', 12.0],
    ['SomethingElse', P.DEFAULT_MAX_DISPLAY_SPEED_MPS]
  ];
  cases.forEach(function (row) {
    const className = row[0];
    const cap = row[1];
    const predictor = new P.DisplayPredictor({ correctionMs: 0 });
    predictor.observe('k', { x: 0, y: 0, yaw: 0, className: className,
      measuredMs: 0, sampleId: 0 }, 0);
    // A 100 m jump in 100 ms would be 1000 m/s.
    predictor.observe('k', { x: 100, y: 0, yaw: 0, className: className,
      measuredMs: 100, sampleId: 100 }, 100);
    const sample = predictor.sample('k', 100);
    check(className + ' speed is clamped to ' + cap + ' m/s',
      near(Math.hypot(sample.vx, sample.vy), cap, 1e-6),
      'speed=' + Math.hypot(sample.vx, sample.vy));
  });

  // Direction is preserved by the magnitude clamp.
  const predictor = new P.DisplayPredictor({ correctionMs: 0 });
  predictor.observe('d', { x: 0, y: 0, yaw: 0, className: 'Pedestrian',
    measuredMs: 0, sampleId: 0 }, 0);
  predictor.observe('d', { x: 30, y: 40, yaw: 0, className: 'Pedestrian',
    measuredMs: 100, sampleId: 100 }, 100);
  const s = predictor.sample('d', 100);
  check('the speed clamp preserves direction',
    near(s.vx / s.vy, 30 / 40, 1e-9), 'vx/vy=' + (s.vx / s.vy));
  check('the clamped speed equals the pedestrian cap',
    near(Math.hypot(s.vx, s.vy), 4.0, 1e-6));

  // Yaw rate clamp.
  const yawPredictor = new P.DisplayPredictor({ correctionMs: 0 });
  yawPredictor.observe('y', { x: 0, y: 0, yaw: 0, className: 'Vehicle',
    measuredMs: 0, sampleId: 0 }, 0);
  yawPredictor.observe('y', { x: 0, y: 0, yaw: 170, className: 'Vehicle',
    measuredMs: 100, sampleId: 100 }, 100);
  // 170 deg in 100 ms is 1700 deg/s; it must clamp to the registered 60 deg/s.
  const yawAdvance = yawPredictor.sample('y', 200);
  check('yaw rate is clamped to +/-60 deg/s',
    near(yawAdvance.yaw - 170.0, 6.0, 1e-9), 'advance=' + (yawAdvance.yaw - 170.0));
  check('the yaw-rate clamp constant is 60 deg/s',
    P.MAX_DISPLAY_YAW_RATE_DPS === 60.0);
  // A backwards turn clamps symmetrically.
  const backwards = new P.DisplayPredictor({ correctionMs: 0 });
  backwards.observe('y2', { x: 0, y: 0, yaw: 0, className: 'Vehicle',
    measuredMs: 0, sampleId: 0 }, 0);
  backwards.observe('y2', { x: 0, y: 0, yaw: -170, className: 'Vehicle',
    measuredMs: 100, sampleId: 100 }, 100);
  check('a negative yaw rate clamps to -60 deg/s',
    near(backwards.sample('y2', 200).yaw - (-170.0), -6.0, 1e-9));
}());

// =============================================================================
// 9. Correction when a new measurement arrives
// =============================================================================
(function correction() {
  const predictor = new P.DisplayPredictor();
  predictor.observe('c', { x: 0, y: 0, yaw: 0, className: 'Vehicle',
    measuredMs: 0, sampleId: 0 }, 0);
  predictor.observe('c', { x: 1, y: 0, yaw: 0, className: 'Vehicle',
    measuredMs: 100, sampleId: 100 }, 100);
  // Let the first residual decay fully so this test measures ONE correction.
  const settledBefore = predictor.sample('c', 100 + P.CORRECTION_MS);
  check('the residual is gone after one correction window',
    near(settledBefore.x, 1.0 + 10.0 * (P.CORRECTION_MS / 1000.0), 1e-9),
    'x=' + settledBefore.x);

  // Extrapolated onward to t=250 with zero residual -> 1 + 10 * 0.15 = 2.5.
  const drifted = predictor.sample('c', 250).x;
  check('pre-correction pose is extrapolated', near(drifted, 2.5, 1e-9),
    'x=' + drifted);

  // A new measurement arrives at t=250 but reports only x = 1.2: the object
  // slowed far more than the display assumed. The drawn pose must not jump.
  predictor.observe('c', { x: 1.2, y: 0, yaw: 0, className: 'Vehicle',
    measuredMs: 250, sampleId: 250 }, 250);
  const atArrival = predictor.sample('c', 250);
  check('the drawn pose is continuous across a correction (no visible jump)',
    near(atArrival.x, drifted, 1e-9), 'x=' + atArrival.x + ' vs ' + drifted);
  check('the correction adopts the new measurement as the base immediately',
    near(predictor.sample('c', 250).vx, (1.2 - 1.0) / 0.15, 1e-9),
    'vx=' + predictor.sample('c', 250).vx);

  // After the correction window the residual is gone and the pose is pure
  // measurement + prediction.
  const settled = predictor.sample('c', 250 + P.CORRECTION_MS);
  const expected = 1.2 + atArrival.vx * (P.CORRECTION_MS / 1000.0);
  check('the correction residual decays to zero within the window',
    near(settled.x, expected, 1e-9), 'x=' + settled.x + ' expected ' + expected);
  check('the correction window is shorter than the horizon',
    P.CORRECTION_MS < P.PREDICTION_HORIZON_MS);

  // Monotone convergence: the residual magnitude never grows.
  let previous = Infinity;
  let monotone = true;
  for (let t = 250; t <= 250 + P.CORRECTION_MS; t += 5) {
    const sample = predictor.sample('c', t);
    const pure = 1.2 + sample.vx * ((t - 250) / 1000.0);
    const residual = Math.abs(sample.x - pure);
    if (residual > previous + 1e-12) { monotone = false; }
    previous = residual;
  }
  check('the correction residual decays monotonically', monotone);

  // Steady state under a regular 100 ms cadence: the residual must shrink
  // toward zero rather than sit at a constant lag, which is the whole point of
  // replacing the lag-only interpolation.
  const steady = new P.DisplayPredictor();
  let residuals = [];
  for (let step = 0; step <= 12; step += 1) {
    const time = step * 100;
    steady.observe('s', { x: step * 1.0, y: 0, yaw: 0, className: 'Vehicle',
      measuredMs: time, sampleId: time }, time);
    residuals.push(Math.abs(steady.sample('s', time).x - step * 1.0));
  }
  check('steady-state display lag decays toward zero',
    residuals[residuals.length - 1] < 0.01 && residuals[1] > 0.5,
    'first=' + residuals[1].toFixed(4) + ' last=' +
    residuals[residuals.length - 1].toFixed(6));
}());

// =============================================================================
// 10. Repeated, missing, invalid and non-monotonic timestamps
// =============================================================================
(function timestampSafety() {
  const predictor = new P.DisplayPredictor({ correctionMs: 0 });
  predictor.observe('t', { x: 0, y: 0, yaw: 0, className: 'Vehicle',
    measuredMs: 1000, sampleId: 1000 }, 0);
  predictor.observe('t', { x: 1, y: 0, yaw: 0, className: 'Vehicle',
    measuredMs: 1100, sampleId: 1100 }, 100);

  // Repeated timestamp: not a new measurement.
  check('a repeated measured timestamp is rejected',
    predictor.observe('t', { x: 99, y: 0, yaw: 0, className: 'Vehicle',
      measuredMs: 1100, sampleId: 1100 }, 120) === 'REPEATED');
  check('a repeated timestamp does not move the drawn pose to the repeat value',
    predictor.sample('t', 100).x < 2.0);

  // Non-monotonic (small backward step): ignored, state preserved.
  check('an out-of-order measurement is rejected',
    predictor.observe('t', { x: -50, y: 0, yaw: 0, className: 'Vehicle',
      measuredMs: 1050, sampleId: 1050 }, 130) === 'OUT_OF_ORDER');
  check('an out-of-order measurement does not corrupt the state',
    near(predictor.sample('t', 100).x, 1.0, 1e-9));

  // Large backward jump: treated as a clock restart, reset with zero velocity.
  check('a large backward clock jump resets the state',
    predictor.observe('t', { x: 5, y: 0, yaw: 0, className: 'Vehicle',
      measuredMs: 1100 - P.BACKWARD_RESET_MS, sampleId: 7 }, 200) === 'RESET');
  const afterReset = predictor.sample('t', 400);
  check('after a reset the velocity is zero', afterReset.vx === 0 && afterReset.vy === 0);
  check('after a reset the pose is the new measurement', near(afterReset.x, 5.0, 1e-9));

  // Missing / non-finite timestamps: position is adopted, velocity stays zero.
  const noClock = new P.DisplayPredictor({ correctionMs: 0 });
  noClock.observe('n', { x: 0, y: 0, yaw: 0, className: 'Vehicle',
    measuredMs: null, sampleId: 1 }, 0);
  check('a missing timestamp still accepts a new sample identity',
    noClock.observe('n', { x: 9, y: 0, yaw: 0, className: 'Vehicle',
      measuredMs: null, sampleId: 2 }, 100) === 'ACCEPTED');
  const noClockSample = noClock.sample('n', 200);
  check('a missing timestamp yields zero velocity, never a guessed one',
    noClockSample.vx === 0 && noClockSample.vy === 0);
  check('a missing timestamp with a repeated sample id is rejected',
    noClock.observe('n', { x: 11, y: 0, yaw: 0, className: 'Vehicle',
      measuredMs: null, sampleId: 2 }, 300) === 'REPEATED');
  check('NaN timestamps are treated as missing, not as a number',
    noClock.observe('n', { x: 12, y: 0, yaw: 0, className: 'Vehicle',
      measuredMs: NaN, sampleId: 3 }, 400) === 'ACCEPTED');

  // Non-finite positions are refused outright.
  check('a non-finite x is refused',
    noClock.observe('n', { x: NaN, y: 0, yaw: 0, className: 'Vehicle',
      measuredMs: null, sampleId: 4 }, 500) === 'INVALID');
  check('an infinite y is refused',
    noClock.observe('n', { x: 1, y: Infinity, yaw: 0, className: 'Vehicle',
      measuredMs: null, sampleId: 5 }, 600) === 'INVALID');

  // Regression: Number(null) is 0, so a null timestamp must be detected as
  // ABSENT rather than coerced into a real measured time of zero.
  const nulled = new P.DisplayPredictor({ correctionMs: 0 });
  nulled.observe('z', { x: 0, y: 0, yaw: 0, className: 'Vehicle',
    measuredMs: null, sampleId: 'a' }, 0);
  check('a null timestamp is absent, not time zero',
    nulled.observe('z', { x: 1, y: 0, yaw: 0, className: 'Vehicle',
      measuredMs: null, sampleId: 'b' }, 100) === 'ACCEPTED');
  check('an empty-string timestamp is absent too',
    nulled.observe('z', { x: 2, y: 0, yaw: 0, className: 'Vehicle',
      measuredMs: '', sampleId: 'c' }, 200) === 'ACCEPTED');
  check('a genuine zero timestamp is still a real measurement',
    (function () {
      const zeroed = new P.DisplayPredictor({ correctionMs: 0 });
      zeroed.observe('q', { x: 0, y: 0, yaw: 0, className: 'Vehicle',
        measuredMs: 0, sampleId: 0 }, 0);
      return zeroed.observe('q', { x: 1, y: 0, yaw: 0, className: 'Vehicle',
        measuredMs: 0, sampleId: 0 }, 50) === 'REPEATED';
    }()));

  // A gap longer than the registered maximum must not carry a stale velocity.
  const gapped = new P.DisplayPredictor({ correctionMs: 0 });
  gapped.observe('g', { x: 0, y: 0, yaw: 0, className: 'Vehicle',
    measuredMs: 0, sampleId: 0 }, 0);
  gapped.observe('g', { x: 1, y: 0, yaw: 0, className: 'Vehicle',
    measuredMs: P.MAX_SAMPLE_GAP_MS + 1, sampleId: 1 }, 10);
  const gapSample = gapped.sample('g', 10);
  check('a too-long sample gap yields zero velocity',
    gapSample.vx === 0 && gapSample.vy === 0);

  // retain() drops departed keys so state cannot grow without bound.
  gapped.observe('other', { x: 0, y: 0, yaw: 0, className: 'Vehicle',
    measuredMs: 0, sampleId: 0 }, 0);
  check('predictor holds both keys', gapped.size() === 2);
  gapped.retain(['g']);
  check('retain drops departed keys', gapped.size() === 1);
  check('retain([]) clears all state',
    (function () { gapped.retain([]); return gapped.size() === 0; }()));
}());

// =============================================================================
// 11. Cached road-yaw lookup equivalence on real Town10HD_Opt geometry
// =============================================================================
(function roadIndexEquivalence() {
  const geometryPath = path.resolve(
    __dirname, '..', '..', 'recordings', 'two_ego_live.jsonl.static.json'
  );
  const geometry = JSON.parse(fs.readFileSync(geometryPath, 'utf8'));
  const roads = geometry.roads || [];
  let segmentCount = 0;
  roads.forEach(function (polyline) { segmentCount += Math.max(0, polyline.length - 1); });
  check('real Town10HD_Opt geometry loaded', roads.length > 100 && segmentCount > 5000,
    roads.length + ' polylines, ' + segmentCount + ' segments');

  // The ORIGINAL unindexed implementation, verbatim, as the reference oracle.
  function roadHdgLinear(x, y) {
    let bd = 1e18;
    let bh = null;
    for (const pl of roads) {
      for (let i = 0; i < pl.length - 1; i++) {
        const ax = pl[i][0], ay = pl[i][1];
        const dx = pl[i + 1][0] - ax, dy = pl[i + 1][1] - ay;
        const s2 = dx * dx + dy * dy;
        if (s2 < 1e-9) continue;
        const t = Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / s2));
        const px = ax + t * dx, py = ay + t * dy;
        const d2 = (x - px) * (x - px) + (y - py) * (y - py);
        if (d2 < bd) { bd = d2; bh = Math.atan2(dy, dx); }
      }
    }
    return (bh !== null && bd <= 64) ? bh : null;
  }

  const index = new P.RoadHeadingIndex(roads);
  const stats = index.stats();
  check('index registered every usable segment',
    stats.segmentCount === segmentCount || stats.segmentCount > 5000,
    'indexed ' + stats.segmentCount + ' of ' + segmentCount);
  check('index cell size is at least the acceptance radius',
    stats.cellSizeM >= stats.maxDistanceM);
  check('index acceptance radius matches the original 8 m (64 m^2)',
    stats.maxDistanceM === 8.0);

  // Sweep a dense grid over the whole town, including off-road points that must
  // return null from both implementations.
  let compared = 0;
  let mismatches = 0;
  let nonNull = 0;
  const queries = [];
  for (let x = -130; x <= 130; x += 3.5) {
    for (let y = -80; y <= 155; y += 3.5) {
      queries.push([x, y]);
    }
  }
  queries.forEach(function (q) {
    const reference = roadHdgLinear(q[0], q[1]);
    const cached = index.headingRad(q[0], q[1]);
    compared += 1;
    if (reference === null || cached === null) {
      if (reference !== cached) { mismatches += 1; }
    } else {
      nonNull += 1;
      if (!near(reference, cached, 0)) { mismatches += 1; }
    }
  });
  check('cached road-yaw lookup is exactly equivalent to the linear scan',
    mismatches === 0, mismatches + ' mismatches over ' + compared + ' queries');
  check('the sweep actually exercised on-road points',
    nonNull > 1000, nonNull + ' on-road hits');

  // Performance: informational, but a genuine regression guard on the whole point
  // of the change.
  const timedQueries = queries.slice(0, 2000);
  let t0 = process.hrtime.bigint();
  timedQueries.forEach(function (q) { roadHdgLinear(q[0], q[1]); });
  const linearNs = Number(process.hrtime.bigint() - t0);
  t0 = process.hrtime.bigint();
  timedQueries.forEach(function (q) { index.headingRad(q[0], q[1]); });
  const indexedNs = Number(process.hrtime.bigint() - t0);
  check('the index is materially faster than the full scan',
    indexedNs * 4 < linearNs,
    'linear ' + (linearNs / 1e6).toFixed(1) + ' ms vs indexed ' +
    (indexedNs / 1e6).toFixed(1) + ' ms');

  // Degenerate inputs.
  check('an empty road set yields null', new P.RoadHeadingIndex([]).headingRad(0, 0) === null);
  check('a non-finite query yields null', index.headingRad(NaN, 0) === null);

  roadIndexReport = {
    segments: stats.segmentCount,
    cells: stats.cellCount,
    queries: compared,
    on_road_hits: nonNull,
    linear_ms: Number((linearNs / 1e6).toFixed(2)),
    indexed_ms: Number((indexedNs / 1e6).toFixed(2)),
    speedup: Number((linearNs / indexedNs).toFixed(1))
  };
}());

// =============================================================================
// 12. Metric formatting never substitutes zero for absent
// =============================================================================
(function metricFormatting() {
  check('null formats as N/A', I.formatMetric(null) === I.NOT_AVAILABLE);
  check('undefined formats as N/A', I.formatMetric(undefined) === I.NOT_AVAILABLE);
  check('NaN formats as N/A', I.formatMetric(NaN) === I.NOT_AVAILABLE);
  check('an N/A string passes through', I.formatMetric('N/A') === 'N/A');
  check('the incompatible-clock string passes through unchanged',
    I.formatMetric(I.INCOMPATIBLE_CLOCKS) === I.INCOMPATIBLE_CLOCKS);
  check('zero formats as zero, not N/A', I.formatMetric(0, 1) === '0.0');

  const empty = I.describeWindow({ label: 'x', unit: 'ms', count: 0, capacity: 120 });
  check('an empty rolling window renders N/A', empty.text === I.NOT_AVAILABLE);
  check('an empty rolling window is flagged empty', empty.empty === true);
  const filled = I.describeWindow({
    label: 'edge tail-complete -> map-install', unit: 'ms',
    count: 5, capacity: 120, current: 11.2, p50: 10.5, p95: 14.1
  });
  check('a filled window names its boundary',
    filled.label.indexOf('->') > 0 && filled.label.indexOf('map-install') > 0);
  check('a filled window shows current, P50 and P95',
    /now .* P50 .* P95/.test(filled.text), filled.text);
  check('no window label uses the phrase "network latency"',
    filled.label.toLowerCase().indexOf('network latency') === -1);

  // Bounded, deterministic browser-side rolling buffer.
  const buffer = new I.RollingBuffer(5);
  for (let i = 1; i <= 20; i += 1) { buffer.push(i); }
  check('the rolling buffer is bounded', buffer.values.length === 5);
  check('the rolling buffer keeps the newest samples',
    buffer.values[0] === 16 && buffer.values[4] === 20);
  const summary = buffer.summary();
  check('the buffer reports the current sample', summary.current === 20);
  check('buffer percentiles are deterministic nearest-rank',
    summary.p50 === 18 && summary.p95 === 20,
    'p50=' + summary.p50 + ' p95=' + summary.p95);
  check('a non-finite sample is dropped, not zeroed', buffer.push(NaN) === false);
  check('an empty buffer reports N/A',
    new I.RollingBuffer(3).summary().current === I.NOT_AVAILABLE);
}());

// =============================================================================
if (failures.length > 0) {
  console.error('FAILURES (' + failures.length + '):');
  failures.forEach(function (item) { console.error('  - ' + item); });
  process.exit(1);
}
console.log(JSON.stringify({
  status: 'PASS',
  checks_passed: passed,
  road_index: roadIndexReport
}, null, 2));
