// Cooperative spatial-map viewer application (presentation layer).
//
// Responsibilities, and nothing else:
//   * poll the map service's snapshot and metrics endpoints;
//   * project the snapshot with ScenesenseMapInventory (pure);
//   * advance the causal display predictor with ScenesenseMapPresentation (pure);
//   * draw, and render an honest metrics panel.
//
// It never posts, never mutates a snapshot object, and never computes a
// scientific quantity. The keyboard changes only what is drawn.

(function () {
  'use strict';

  var P = window.ScenesenseMapPresentation;
  var I = window.ScenesenseMapInventory;

  // ---- cadences ------------------------------------------------------------
  var SNAPSHOT_POLL_MS = 50;      // 20 Hz snapshot polling
  var METRICS_POLL_MS = 500;      // 2 Hz metrics/chart refresh, per the contract
  var STATIC_RETRY_MS = 2000;
  var FPS_WINDOW = 60;
  var SPARK_CAPACITY = 120;       // bounded rolling window for the browser plots

  var THEME = {
    background: '#0A0D12',
    stageEdge: '#1B222C',
    building: '#181D25',
    buildingEdge: '#242B35',
    road: '#39424E',
    roadCasing: '#22293280',
    anchor: '#6B7280',
    roi: 'rgba(120,170,225,0.28)',
    routeOverlay: 'rgba(240,228,66,0.22)',
    text: '#D7DEE7',
    textDim: '#8E99A6',
    egoOutline: '#0A0D12'
  };

  var cv = document.getElementById('map');
  var ctx = cv.getContext('2d');
  var badgeEl = document.getElementById('badge');
  var egoLineEl = document.getElementById('egoline');
  var noticeEl = document.getElementById('notice');
  var legendEl = document.getElementById('legend');
  var metricsEl = document.getElementById('metrics');
  var controlsEl = document.getElementById('controls');
  var stageEl = document.getElementById('stage');

  var DPR = Math.max(1, window.devicePixelRatio || 1);

  var statics = { roads: [], buildings: [], route_overlay: null };
  var roadIndex = null;
  var snap = null;
  var serverMetrics = null;
  var viewMode = I.MODE_ASSOCIATED;
  var showMetrics = true;
  var showLegend = true;
  var showControls = true;
  var showRoute = true;

  // One predictor per kind of thing. Keys never cross between them.
  var trackPredictor = new P.DisplayPredictor();
  var streamPredictor = new P.DisplayPredictor();

  var pollRtt = new I.RollingBuffer(SPARK_CAPACITY);
  var frameIntervals = new I.RollingBuffer(FPS_WINDOW);
  var lastFrameMs = 0;
  var lastEgo = null;          // last drawn camera pose, for hold-through-gap

  // ---- layout --------------------------------------------------------------
  function resize() {
    var w = stageEl.clientWidth;
    var h = stageEl.clientHeight;
    cv.width = Math.max(1, Math.round(w * DPR));
    cv.height = Math.max(1, Math.round(h * DPR));
    cv.style.width = w + 'px';
    cv.style.height = h + 'px';
  }
  window.addEventListener('resize', resize);

  // ---- view mode and chrome ------------------------------------------------
  function setViewMode(mode) {
    viewMode = mode === I.MODE_RAW ? I.MODE_RAW : I.MODE_ASSOCIATED;
    document.getElementById('rawBtn').classList.toggle('active', viewMode === I.MODE_RAW);
    document.getElementById('assocBtn')
      .classList.toggle('active', viewMode === I.MODE_ASSOCIATED);
  }

  function applyChrome() {
    legendEl.classList.toggle('hidden', !showLegend);
    metricsEl.classList.toggle('hidden', !showMetrics);
    controlsEl.classList.toggle('hidden', !showControls);
  }

  document.getElementById('rawBtn').onclick = function () { setViewMode(I.MODE_RAW); };
  document.getElementById('assocBtn').onclick = function () { setViewMode(I.MODE_ASSOCIATED); };
  document.getElementById('metricsBtn').onclick = function () {
    showMetrics = !showMetrics; applyChrome();
  };
  document.getElementById('presentBtn').onclick = function () {
    showMetrics = false; showControls = false; applyChrome();
  };

  window.addEventListener('keydown', function (event) {
    var key = String(event.key || '').toLowerCase();
    if (key === 'r' || key === '1') { setViewMode(I.MODE_RAW); }
    if (key === 'f' || key === '2') { setViewMode(I.MODE_ASSOCIATED); }
    if (key === ' ') {
      event.preventDefault();
      setViewMode(viewMode === I.MODE_RAW ? I.MODE_ASSOCIATED : I.MODE_RAW);
    }
    if (key === 'm') { showMetrics = !showMetrics; applyChrome(); }
    if (key === 'l') { showLegend = !showLegend; applyChrome(); }
    if (key === 'c') { showControls = !showControls; applyChrome(); }
    if (key === 'b') { showRoute = !showRoute; }
    if (key === 'p') {
      // Presentation mode: strip everything that is not the map and its legend.
      var bare = showMetrics || showControls;
      showMetrics = !bare; showControls = !bare; applyChrome();
    }
  });

  // ---- data ----------------------------------------------------------------
  function loadStatic() {
    fetch('/api/spatial_map/static_geometry').then(function (response) {
      return response.json();
    }).then(function (document_) {
      statics = document_ || statics;
      // Built once. Never rebuilt on an animation frame.
      roadIndex = new P.RoadHeadingIndex(statics.roads || []);
    }).catch(function () {
      window.setTimeout(loadStatic, STATIC_RETRY_MS);
    });
  }

  function poll() {
    var issued = performance.now();
    fetch('/api/spatial_map/latest').then(function (response) {
      return response.json();
    }).then(function (document_) {
      snap = document_;
      pollRtt.push(performance.now() - issued);
    }).catch(function () {
      /* keep the previous snapshot and hold */
    }).then(function () {
      window.setTimeout(poll, SNAPSHOT_POLL_MS);
    });
  }

  function pollMetrics() {
    fetch('/api/spatial_map/metrics').then(function (response) {
      return response.json();
    }).then(function (document_) {
      serverMetrics = document_;
    }).catch(function () {
      /* leave the last metrics document in place */
    }).then(function () {
      window.setTimeout(pollMetrics, METRICS_POLL_MS);
    });
  }

  // ---- ego / camera --------------------------------------------------------
  function followTarget() {
    var focus = snap && snap.metadata && snap.metadata.focus_view;
    if (!focus || !focus.ego_pose) { return null; }
    var streams = (snap.active_streams || []);
    var followed = null;
    for (var i = 0; i < streams.length; i += 1) {
      if (streams[i].stream_id === focus.follow_stream_id) { followed = streams[i]; break; }
    }
    return {
      x: Number(focus.ego_pose.x),
      y: Number(focus.ego_pose.y),
      yaw: Number(focus.ego_pose.yaw_deg),
      radius: Number(focus.radius_m || 40) + Number(focus.padding_m || 0),
      streamId: focus.follow_stream_id,
      measuredMs: followed ? captureMs(followed) : null,
      sampleId: followed ? followed.frame_id : (snap ? snap.frame_id : null)
    };
  }

  // capture_timestamp_ns is the measured capture clock the service exposes for
  // this stream. Absent (legacy packets) -> null, which makes the predictor fall
  // back to position-stepping without a velocity.
  function captureMs(stream) {
    var value = Number(stream && stream.capture_timestamp_ns);
    return (typeof value === 'number' && isFinite(value)) ? value / 1e6 : null;
  }

  // ---- drawing -------------------------------------------------------------
  function drawBackground(project, W, H) {
    ctx.fillStyle = THEME.background;
    ctx.fillRect(0, 0, W, H);

    var i, j, points, screen;
    ctx.fillStyle = THEME.building;
    ctx.strokeStyle = THEME.buildingEdge;
    ctx.lineWidth = 1 * DPR;
    var buildings = statics.buildings || [];
    for (i = 0; i < buildings.length; i += 1) {
      points = buildings[i];
      ctx.beginPath();
      for (j = 0; j < points.length; j += 1) {
        screen = project(points[j][0], points[j][1]);
        if (j === 0) { ctx.moveTo(screen[0], screen[1]); } else { ctx.lineTo(screen[0], screen[1]); }
      }
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    }

    // Roads get a wide dark casing plus a lighter core: reads as geometry rather
    // than as a wireframe on a recording.
    var roads = statics.roads || [];
    var pass;
    for (pass = 0; pass < 2; pass += 1) {
      ctx.strokeStyle = pass === 0 ? THEME.roadCasing : THEME.road;
      ctx.lineWidth = (pass === 0 ? 7 : 3.2) * DPR;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      for (i = 0; i < roads.length; i += 1) {
        points = roads[i];
        ctx.beginPath();
        for (j = 0; j < points.length; j += 1) {
          screen = project(points[j][0], points[j][1]);
          if (j === 0) { ctx.moveTo(screen[0], screen[1]); } else { ctx.lineTo(screen[0], screen[1]); }
        }
        ctx.stroke();
      }
    }

    // Optional canonical Route-B overlay: only when the service published a
    // hash-verified route contract.
    var overlay = statics.route_overlay;
    if (showRoute && overlay && overlay.points && overlay.points.length > 1) {
      ctx.save();
      ctx.strokeStyle = THEME.routeOverlay;
      ctx.lineWidth = 10 * DPR;
      ctx.lineCap = 'round';
      ctx.setLineDash([18 * DPR, 14 * DPR]);
      ctx.beginPath();
      for (j = 0; j < overlay.points.length; j += 1) {
        screen = project(overlay.points[j][0], overlay.points[j][1]);
        if (j === 0) { ctx.moveTo(screen[0], screen[1]); } else { ctx.lineTo(screen[0], screen[1]); }
      }
      ctx.stroke();
      ctx.restore();
    }

    ctx.fillStyle = THEME.anchor;
    var anchors = (snap && snap.traffic_light_anchors) || [];
    for (i = 0; i < anchors.length; i += 1) {
      var location = anchors[i].location || {};
      screen = project(location.x, location.y);
      ctx.beginPath();
      ctx.moveTo(screen[0], screen[1] - 4.5 * DPR);
      ctx.lineTo(screen[0] - 4.5 * DPR, screen[1] + 3.5 * DPR);
      ctx.lineTo(screen[0] + 4.5 * DPR, screen[1] + 3.5 * DPR);
      ctx.closePath();
      ctx.fill();
    }
  }

  function drawFovWedge(project, scale, pose, color, radiusM) {
    var screen = project(pose.x, pose.y);
    var yaw = pose.yaw * Math.PI / 180.0;
    var half = pose.fovDeg * Math.PI / 360.0;
    var radius = radiusM * scale;
    ctx.save();
    ctx.translate(screen[0], screen[1]);
    ctx.rotate(yaw);
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.arc(0, 0, radius, -half, half);
    ctx.closePath();
    ctx.globalAlpha = 0.10;
    ctx.fillStyle = color;
    ctx.fill();
    ctx.globalAlpha = 0.42;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.2 * DPR;
    ctx.stroke();
    ctx.restore();
  }

  function drawFootprint(project, scale, entry, pose) {
    var display = entry.display;
    var screen = project(pose.x, pose.y);
    var lengthPx = Math.max(4 * DPR, display.lengthM * scale);
    var widthPx = Math.max(4 * DPR, display.widthM * scale);

    // Vehicles take the cached road tangent for DISPLAY YAW only. Measured XY is
    // never lane-snapped.
    var yawDeg = pose.yaw;
    if (!display.isPedestrian && roadIndex) {
      var heading = roadIndex.headingRad(pose.x, pose.y);
      if (heading !== null) { yawDeg = heading * 180.0 / Math.PI; }
    }

    // Presentation-only prediction leader: a faint line back to the measured
    // point. It is a single thin line, never a second footprint or marker, so a
    // predicted object can never read as an additional detection.
    if (display.predictionMs > 0 && pose.measuredX !== undefined) {
      var from = project(pose.measuredX, pose.measuredY);
      ctx.save();
      ctx.globalAlpha = 0.34;
      ctx.strokeStyle = display.color;
      ctx.lineWidth = 1.1 * DPR;
      ctx.beginPath();
      ctx.moveTo(from[0], from[1]);
      ctx.lineTo(screen[0], screen[1]);
      ctx.stroke();
      ctx.restore();
    }

    ctx.save();
    ctx.translate(screen[0], screen[1]);
    ctx.rotate(yawDeg * Math.PI / 180.0);
    ctx.strokeStyle = display.color;
    ctx.lineWidth = 2 * DPR;
    // Held past the prediction horizon -> dashed, so "no longer tracking a
    // measurement" is visible without reading the panel.
    ctx.setLineDash(display.held ? [5 * DPR, 4 * DPR] : []);
    if (display.isPedestrian) {
      ctx.globalAlpha = 0.30;
      ctx.fillStyle = display.color;
      ctx.fillRect(-lengthPx / 2, -widthPx / 2, lengthPx, widthPx);
      ctx.globalAlpha = 1.0;
    }
    ctx.strokeRect(-lengthPx / 2, -widthPx / 2, lengthPx, widthPx);
    // A+B association: redundant double stroke so the combined state is not
    // signalled by hue alone.
    if (display.multiSource) {
      ctx.globalAlpha = 0.85;
      ctx.lineWidth = 1 * DPR;
      ctx.strokeRect(
        -lengthPx / 2 - 3 * DPR, -widthPx / 2 - 3 * DPR,
        lengthPx + 6 * DPR, widthPx + 6 * DPR
      );
      ctx.globalAlpha = 1.0;
    }
    ctx.setLineDash([]);
    ctx.restore();

    ctx.fillStyle = display.color;
    ctx.beginPath();
    ctx.arc(screen[0], screen[1], 2.4 * DPR, 0, Math.PI * 2);
    ctx.fill();
  }

  function drawEgoMarker(project, pose, color, label) {
    var screen = project(pose.x, pose.y);
    ctx.save();
    ctx.translate(screen[0], screen[1]);
    ctx.rotate(pose.yaw * Math.PI / 180.0);
    ctx.fillStyle = color;
    ctx.strokeStyle = THEME.egoOutline;
    ctx.lineWidth = 2 * DPR;
    ctx.beginPath();
    ctx.moveTo(13 * DPR, 0);
    ctx.lineTo(-8 * DPR, -7.5 * DPR);
    ctx.lineTo(-5 * DPR, 0);
    ctx.lineTo(-8 * DPR, 7.5 * DPR);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    ctx.restore();
    ctx.font = '600 ' + (12 * DPR) + 'px ui-sans-serif, system-ui, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillStyle = THEME.background;
    ctx.fillText(label, screen[0] + 13 * DPR + 1, screen[1] - 11 * DPR + 1);
    ctx.fillStyle = color;
    ctx.fillText(label, screen[0] + 13 * DPR, screen[1] - 11 * DPR);
  }

  // ---- panels --------------------------------------------------------------
  function row(label, value, tone) {
    return '<div class="row"><span class="k">' + label + '</span>' +
      '<span class="v' + (tone ? ' ' + tone : '') + '">' + value + '</span></div>';
  }

  function swatch(color, doubled) {
    return '<span class="sw' + (doubled ? ' dbl' : '') +
      '" style="background:' + color + ';border-color:' + color + '"></span>';
  }

  function renderLegend(view) {
    var html = '';
    if (view.associated) {
      var sources = view.sources;
      for (var i = 0; i < sources.length; i += 1) {
        html += '<div class="row"><span class="k">' +
          swatch(view.sourceColor[sources[i]]) + ' ' + sources[i] + ' only</span>' +
          '<span class="v">' + (view.counts.perSource[sources[i]] || 0) + '</span></div>';
      }
      html += '<div class="row"><span class="k">' + swatch(I.COLOR_BOTH, true) +
        ' A + B associated</span><span class="v">' + view.counts.both + '</span></div>';
    } else {
      for (var j = 0; j < view.sources.length; j += 1) {
        html += '<div class="row"><span class="k">' +
          swatch(view.sourceColor[view.sources[j]]) + ' ' + view.sources[j] +
          ' raw reports</span><span class="v">' +
          (view.counts.perSource[view.sources[j]] || 0) + '</span></div>';
      }
    }
    html += '<div class="sep"></div>';
    html += '<div class="note"><span class="wedge"></span> measured sensor FoV</div>';
    html += '<div class="note"><span class="lead"></span> presentation-only ' +
      'predicted motion (≤' + P.PREDICTION_HORIZON_MS + ' ms, causal)</div>';
    html += '<div class="note"><span class="dash"></span> held past the ' +
      'prediction horizon</div>';
    if (statics.route_overlay && statics.route_overlay.points) {
      html += '<div class="note"><span class="route"></span> canonical Route-B ' +
        'reference path (display overlay)</div>';
    }
    legendEl.innerHTML = html;
  }

  function sparkline(values, width, height) {
    if (!values || values.length < 2) { return ''; }
    var minimum = Math.min.apply(null, values);
    var maximum = Math.max.apply(null, values);
    var span = (maximum - minimum) || 1;
    var points = [];
    for (var i = 0; i < values.length; i += 1) {
      var x = (i / (values.length - 1)) * width;
      var y = height - ((values[i] - minimum) / span) * height;
      points.push(x.toFixed(1) + ',' + y.toFixed(1));
    }
    return '<svg class="spark" viewBox="0 0 ' + width + ' ' + height +
      '" preserveAspectRatio="none"><polyline points="' + points.join(' ') +
      '"/></svg>';
  }

  function renderMetrics(view) {
    var html = '';
    var na = I.NOT_AVAILABLE;

    // --- always-valid viewer/service metrics
    html += '<div class="sec">stream health</div>';
    var streams = (serverMetrics && serverMetrics.streams) || {};
    var names = Object.keys(streams).sort();
    if (names.length === 0) {
      html += row('streams', na, 'dim');
    }
    for (var i = 0; i < names.length; i += 1) {
      var stream = streams[names[i]];
      html += row(names[i] + ' rate', I.formatMetric(stream.update_rate_hz, 2, 'Hz'));
      html += row(names[i] + ' last frame', I.formatMetric(stream.last_accepted_frame_id, 0));
      html += row(names[i] + ' since rx', I.formatMetric(stream.since_last_received_ms, 0, 'ms'));
    }

    html += '<div class="sec">association</div>';
    html += row('raw observations', view.rawObservationCount);
    html += row(
      'associated representations',
      view.associated ? view.associatedRepresentationCount : na,
      view.associated ? '' : 'dim'
    );
    var perSource = view.counts.perSource;
    var sourceNames = view.sources;
    if (view.associated) {
      for (var s = 0; s < sourceNames.length; s += 1) {
        html += row(sourceNames[s] + ' only', perSource[sourceNames[s]] || 0);
      }
      html += row('A + B associated', view.counts.both);
      var reduction = I.associationReduction(
        view.rawObservationCount, view.associatedRepresentationCount
      );
      html += row(
        'reduction 1-assoc/raw',
        typeof reduction === 'string' ? reduction : (reduction * 100).toFixed(1) + ' %'
      );
    } else {
      html += row('mode', 'raw — measurement-stepped', 'dim');
    }
    html += row('stale tracks hidden', view.staleHidden, view.staleHidden ? 'warn' : '');

    html += '<div class="sec">service counters</div>';
    var counters = (serverMetrics && serverMetrics.ingest_counters) || {};
    var counterKeys = ['accepted', 'duplicate_identical', 'conflict', 'rejected', 'stale_streams'];
    for (var c = 0; c < counterKeys.length; c += 1) {
      html += row(
        counterKeys[c].replace(/_/g, ' '),
        Object.prototype.hasOwnProperty.call(counters, counterKeys[c])
          ? counters[counterKeys[c]] : na
      );
    }

    html += '<div class="sec">viewer / service timing</div>';
    var rtt = pollRtt.summary();
    html += row('browser poll RTT', rtt.count
      ? I.formatMetric(rtt.p50, 1, 'ms') + ' P50' : na);
    var fps = frameIntervals.summary();
    html += row('render FPS', fps.count && fps.p50
      ? I.formatMetric(1000.0 / fps.p50, 0, 'fps') : na);
    var serviceWindows = (serverMetrics && serverMetrics.service_windows) || [];
    for (var w = 0; w < serviceWindows.length; w += 1) {
      var described = I.describeWindow(serviceWindows[w]);
      html += row(described.label, described.text, described.empty ? 'dim' : '');
    }

    // --- direct-edge timing: only when the service says the evidence exists
    html += '<div class="sec">direct edge-to-map latency</div>';
    var transport = (serverMetrics && serverMetrics.transport) || {};
    var windows = (serverMetrics && serverMetrics.latency_windows) || [];
    var anyDirect = false;
    for (var d = 0; d < windows.length; d += 1) {
      var summary = windows[d];
      var info = I.describeWindow(summary);
      html += '<div class="lat' + (info.empty ? ' dim' : '') + '">' +
        '<div class="latlabel">' + info.label + '</div>' +
        '<div class="latvalue">' + info.text + '</div>' +
        (info.empty ? '' : '<div class="latmeta">n=' + info.count + '/' +
          info.capacity + '</div>') +
        '</div>';
      if (!info.empty) { anyDirect = true; }
    }
    if (!anyDirect) {
      html += '<div class="note dim">no direct-edge timing evidence on this ' +
        'transport — charts intentionally empty</div>';
    }
    var outcomes = (serverMetrics && serverMetrics.terminal_outcome_counts) || {};
    var outcomeNames = Object.keys(outcomes).sort();
    if (outcomeNames.length === 0) {
      html += row('terminal outcomes', na, 'dim');
    } else {
      for (var o = 0; o < outcomeNames.length; o += 1) {
        html += row(outcomeNames[o], outcomes[outcomeNames[o]]);
      }
    }

    html += '<div class="sec">transport honesty</div>';
    html += row('transport', transport.label || na, 'dim');
    html += row('cross-clock wall AoI', transport.cross_clock_wall_aoi || na, 'warn');
    if (transport.http_post_duration) {
      html += row('HTTP POST', transport.http_post_duration, 'dim');
    }
    if (transport.model_execution) {
      html += row('model execution', transport.model_execution, 'dim');
    }

    html += '<div class="sec">browser poll RTT (last ' + SPARK_CAPACITY + ')</div>';
    html += sparkline(pollRtt.values, 200, 34) ||
      '<div class="note dim">' + na + '</div>';

    metricsEl.innerHTML = html;
  }

  // ---- main loop -----------------------------------------------------------
  function draw() {
    window.requestAnimationFrame(draw);
    var now = performance.now();
    if (lastFrameMs) { frameIntervals.push(now - lastFrameMs); }
    lastFrameMs = now;

    var W = cv.width;
    var H = cv.height;
    ctx.fillStyle = THEME.background;
    ctx.fillRect(0, 0, W, H);

    var target = followTarget();
    if (target) {
      streamPredictor.observe('__follow__', {
        x: target.x, y: target.y, yaw: target.yaw, className: 'Vehicle',
        measuredMs: target.measuredMs, sampleId: target.sampleId
      }, now);
    }
    var camera = streamPredictor.sample('__follow__', now);
    if (!camera) {
      badgeEl.textContent = 'WAITING FOR EGO STREAM';
      badgeEl.className = 'badge waiting';
      noticeEl.textContent = 'no measured ego pose yet';
      return;
    }
    var radius = target ? target.radius : (lastEgo ? lastEgo.radius : 40);
    lastEgo = { radius: radius };

    var scale = Math.min(W, H) / 2 / radius;
    var project = function (wx, wy) {
      return [W / 2 + (wx - camera.x) * scale, H / 2 + (wy - camera.y) * scale];
    };

    drawBackground(project, W, H);

    // ROI box, guarded: focus_view can legitimately be absent.
    var focus = (snap && snap.metadata && snap.metadata.focus_view) || null;
    var bounds = focus && focus.bounds;
    if (bounds) {
      var corner0 = project(bounds.x_min, bounds.y_min);
      var corner1 = project(bounds.x_max, bounds.y_max);
      ctx.strokeStyle = THEME.roi;
      ctx.lineWidth = 1.4 * DPR;
      ctx.strokeRect(
        Math.min(corner0[0], corner1[0]), Math.min(corner0[1], corner1[1]),
        Math.abs(corner1[0] - corner0[0]), Math.abs(corner1[1] - corner0[1])
      );
    }

    var view = I.projectInventory(snap, viewMode, {});

    // --- per-stream measured poses: predicted, drawn as ego markers + FoV
    var streams = (snap && snap.active_streams) || [];
    var streamKeys = [];
    var drawnStreams = [];
    var index;
    for (index = 0; index < streams.length; index += 1) {
      var stream = streams[index];
      var sensor = stream.sensor_pose || {};
      if (sensor.x === null || sensor.x === undefined) { continue; }
      var key = 'stream:' + stream.stream_id;
      streamKeys.push(key);
      streamPredictor.observe(key, {
        x: Number(sensor.x), y: Number(sensor.y), yaw: Number(sensor.yaw_deg || 0),
        className: 'Vehicle', measuredMs: captureMs(stream), sampleId: stream.frame_id
      }, now);
      var pose = streamPredictor.sample(key, now);
      if (!pose) { continue; }
      drawnStreams.push({
        stream: stream,
        color: view.sourceColor[stream.stream_id] || I.COLOR_UE_A,
        pose: { x: pose.x, y: pose.y, yaw: pose.yaw, fovDeg: Number(stream.fov_deg || 120) }
      });
    }
    streamKeys.push('__follow__');
    streamPredictor.retain(streamKeys);

    for (index = 0; index < drawnStreams.length; index += 1) {
      drawFovWedge(
        project, scale, drawnStreams[index].pose, drawnStreams[index].color,
        Math.min(32, radius)
      );
    }

    // --- objects
    if (view.associated) {
      // Associated tracks are predicted. Raw reports are NOT: see below.
      var trackKeys = [];
      for (index = 0; index < view.objects.length; index += 1) {
        var entry = view.objects[index];
        var object = entry.object;
        var identity = object.track_id || object.association_id;
        var location = object.location || {};
        if (!identity) {
          drawFootprint(project, scale, entry, {
            x: Number(location.x), y: Number(location.y),
            yaw: Number(object.map_yaw_deg != null ? object.map_yaw_deg : (object.yaw_deg || 0))
          });
          continue;
        }
        var trackKey = 'track:' + identity;
        trackKeys.push(trackKey);
        var captureNs = Number(object.capture_timestamp_ns);
        trackPredictor.observe(trackKey, {
          x: Number(location.x), y: Number(location.y),
          yaw: Number(object.map_yaw_deg != null ? object.map_yaw_deg : (object.yaw_deg || 0)),
          className: entry.display.className,
          measuredMs: isFinite(captureNs) ? captureNs / 1e6 : null,
          sampleId: isFinite(captureNs) ? captureNs : null
        }, now);
        var predicted = trackPredictor.sample(trackKey, now);
        entry.display.predictionMs = predicted.predictionMs;
        entry.display.held = predicted.held;
        drawFootprint(project, scale, entry, {
          x: predicted.x, y: predicted.y, yaw: predicted.yaw,
          measuredX: Number(location.x), measuredY: Number(location.y)
        });
      }
      trackPredictor.retain(trackKeys);
    } else {
      // RAW MODE IS MEASUREMENT-STEPPED, DELIBERATELY.
      // Smoothing raw reports would blur which measurements were actually
      // received, which is exactly what raw mode exists to show. Objects step
      // when a report arrives. Predictor state for tracks is dropped so a later
      // switch back to associated mode restarts causally.
      trackPredictor.retain([]);
      for (index = 0; index < view.objects.length; index += 1) {
        var rawEntry = view.objects[index];
        var rawLocation = rawEntry.object.location || {};
        drawFootprint(project, scale, rawEntry, {
          x: Number(rawLocation.x), y: Number(rawLocation.y),
          yaw: Number(
            rawEntry.object.map_yaw_deg != null
              ? rawEntry.object.map_yaw_deg
              : (rawEntry.object.yaw_deg || 0)
          )
        });
      }
    }

    for (index = 0; index < drawnStreams.length; index += 1) {
      drawEgoMarker(
        project, drawnStreams[index].pose, drawnStreams[index].color,
        drawnStreams[index].stream.stream_id
      );
    }

    // --- chrome
    var isAssociated = view.associated;
    badgeEl.textContent = isAssociated ? 'ASSOCIATED MAP' : 'RAW UE REPORTS';
    badgeEl.className = 'badge ' + (isAssociated ? 'assoc' : 'raw');
    egoLineEl.innerHTML =
      '<span class="k">ego</span><span class="v">' +
      camera.x.toFixed(1) + ', ' + camera.y.toFixed(1) + '</span>' +
      '<span class="k">yaw</span><span class="v">' + camera.yaw.toFixed(0) + '°</span>' +
      '<span class="k">ROI</span><span class="v">' + radius.toFixed(0) + ' m</span>' +
      '<span class="k">frame</span><span class="v">' +
      ((snap && snap.frame_id != null) ? snap.frame_id : '—') + '</span>' +
      '<span class="k">veh</span><span class="v">' + view.counts.vehicles + '</span>' +
      '<span class="k">ped</span><span class="v">' + view.counts.pedestrians + '</span>';
    var notices = [];
    if (!target) { notices.push('holding last measured ego pose'); }
    if (view.staleHidden) { notices.push(view.staleHidden + ' stale track(s) hidden'); }
    if (!view.conservative) { notices.push('historical fusion policy'); }
    noticeEl.textContent = notices.join(' · ');

    if (showLegend) { renderLegend(view); }
    if (showMetrics && now - (renderMetrics.lastAt || 0) >= METRICS_POLL_MS) {
      // Panel text is rebuilt at 2 Hz while the canvas runs at rAF, so the
      // numbers do not flicker and the DOM is not rewritten every frame.
      renderMetrics.lastAt = now;
      renderMetrics(view);
    }
  }

  resize();
  setViewMode(I.MODE_ASSOCIATED);
  applyChrome();
  loadStatic();
  poll();
  pollMetrics();
  window.requestAnimationFrame(draw);
}());
