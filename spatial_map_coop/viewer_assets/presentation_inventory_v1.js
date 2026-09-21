// Pure inventory projection and metric formatting for the cooperative viewer.
//
// SCOPE CONTRACT
// --------------
// This module decides *what is drawn*, never *what is true*. It is a read-only
// projection of a snapshot the map service already computed:
//
//   * raw mode returns the current raw reports, unchanged and un-predicted;
//   * associated mode returns at most one representation per current
//     association, and never more representations than the observations
//     supporting them;
//   * historical/retained tracks are excluded by the service's
//     CURRENT_ASSOCIATION_TRACKS_ONLY policy and again here by display age;
//   * no function here invents, merges, splits or moves a detection.
//
// It contains no DOM access and no timing source, so it runs unchanged under
// node for the offline tests.

(function (root) {
  'use strict';

  // Okabe-Ito colour-blind-safe set. UE-A keeps the blue family, UE-B the
  // orange family and A+B the green family, so the existing source semantics
  // are preserved while the exact values become CB-readable as a set. Hue is
  // never the only cue: A+B is additionally drawn with a double stroke.
  var COLOR_UE_A = '#56B4E9';   // sky blue
  var COLOR_UE_B = '#E69F00';   // orange
  var COLOR_BOTH = '#009E73';   // bluish green
  var SOURCE_PALETTE = [COLOR_UE_A, COLOR_UE_B, '#CC79A7', '#0072B2', '#D55E00', '#F0E442'];
  var COLOR_PEDESTRIAN = '#CC79A7';  // reddish purple
  var COLOR_VEHICLE = COLOR_UE_A;
  var COLOR_UNKNOWN = '#DDDDDD';

  // Display age beyond which a current-association track is withheld from the
  // drawn inventory and counted in the HUD instead.
  var MAX_DISPLAY_TRACK_AGE_NS = 750000000;

  var CANONICAL_SIZES = {
    Vehicle: [4.6, 2.0],
    Pedestrian: [0.8, 0.8],
    Cyclist: [1.8, 0.7]
  };

  var NOT_AVAILABLE = 'N/A';
  var INCOMPATIBLE_CLOCKS = 'N/A - incompatible clock domains';

  var MODE_RAW = 'raw';
  var MODE_ASSOCIATED = 'associated';

  var CONSERVATIVE_IMPLEMENTATION = 'CONSERVATIVE_MULTI_UE_V1';

  function isFiniteNumber(value) {
    return typeof value === 'number' && isFinite(value);
  }

  function numberOr(value, fallback) {
    var numeric = Number(value);
    return isFiniteNumber(numeric) ? numeric : fallback;
  }

  /** Supporting source stream IDs for one displayed object. */
  function supportingSources(item) {
    var list = item && item.source_stream_ids;
    if (Object.prototype.toString.call(list) === '[object Array]' && list.length > 0) {
      return list.slice();
    }
    if (item && item.source_stream_id) { return [item.source_stream_id]; }
    return [];
  }

  function canonicalSize(className, dimensions) {
    if (Object.prototype.hasOwnProperty.call(CANONICAL_SIZES, className)) {
      return CANONICAL_SIZES[className].slice();
    }
    var dims = dimensions || {};
    return [numberOr(dims.length, 1.2), numberOr(dims.width, 1.2)];
  }

  /**
   * Project one snapshot into a drawable inventory.
   *
   * Returns { mode, conservative, objects, staleHidden, sources, sourceColor,
   *           counts, rawObservationCount, associatedRepresentationCount }.
   *
   * `objects` entries are the snapshot's own objects plus derived presentation
   * fields under `display`. The snapshot objects themselves are not mutated.
   */
  function projectInventory(snapshot, viewMode, options) {
    var opts = options || {};
    var maxAgeNs = numberOr(opts.maxDisplayTrackAgeNs, MAX_DISPLAY_TRACK_AGE_NS);
    var snap = snapshot || {};
    var metadata = snap.metadata || {};
    var policy = metadata.fusion_policy || {};
    var conservative = policy.implementation === CONSERVATIVE_IMPLEMENTATION;
    var associated = conservative && viewMode === MODE_ASSOCIATED;

    var rawObjects = snap.raw_spatial_map_objects || [];
    var associatedObjects = snap.spatial_map_objects || [];

    var pool = associated ? associatedObjects : rawObjects;
    var kept = [];
    var staleHidden = 0;
    var index;
    for (index = 0; index < pool.length; index += 1) {
      var item = pool[index];
      if (associated && numberOr(item.age_ns, 0) > maxAgeNs) { staleHidden += 1; continue; }
      kept.push(item);
    }

    // Source inventory: every active stream plus every stream that supports a
    // drawn object. Sorted so colour assignment is stable frame to frame.
    var streams = snap.active_streams || [];
    var sourceSet = Object.create(null);
    for (index = 0; index < streams.length; index += 1) {
      if (streams[index] && streams[index].stream_id) {
        sourceSet[streams[index].stream_id] = true;
      }
    }
    for (index = 0; index < kept.length; index += 1) {
      var supports = supportingSources(kept[index]);
      for (var s = 0; s < supports.length; s += 1) {
        if (supports[s]) { sourceSet[supports[s]] = true; }
      }
    }
    var sources = Object.keys(sourceSet).sort();
    var sourceColor = Object.create(null);
    for (index = 0; index < sources.length; index += 1) {
      sourceColor[sources[index]] = SOURCE_PALETTE[index % SOURCE_PALETTE.length];
    }
    var bySource = sources.length > 1;

    var perSource = Object.create(null);
    for (index = 0; index < sources.length; index += 1) { perSource[sources[index]] = 0; }
    var both = 0;
    var vehicles = 0;
    var pedestrians = 0;
    var objects = [];

    for (index = 0; index < kept.length; index += 1) {
      var object = kept[index];
      var className = String(object.type || 'Unknown');
      var isPedestrian = className === 'Pedestrian';
      if (isPedestrian) { pedestrians += 1; } else { vehicles += 1; }
      var support = supportingSources(object);
      var multi = associated && support.length >= 2;
      if (multi) {
        both += 1;
      } else {
        for (var k = 0; k < support.length; k += 1) {
          if (support[k] in perSource) { perSource[support[k]] += 1; }
        }
      }
      var color;
      if (multi) {
        color = COLOR_BOTH;
      } else if (bySource) {
        color = sourceColor[support[0]] || COLOR_UNKNOWN;
      } else {
        color = isPedestrian ? COLOR_PEDESTRIAN : COLOR_VEHICLE;
      }
      var size = canonicalSize(className, object.dimensions);
      objects.push({
        object: object,
        display: {
          className: className,
          isPedestrian: isPedestrian,
          support: support,
          multiSource: multi,
          color: color,
          lengthM: size[0],
          widthM: size[1],
          // Presentation-only fields are namespaced under display.* so a reader
          // of the snapshot can never mistake them for measured state.
          predictionMs: 0,
          held: false
        }
      });
    }

    return {
      mode: associated ? MODE_ASSOCIATED : MODE_RAW,
      conservative: conservative,
      associated: associated,
      objects: objects,
      staleHidden: staleHidden,
      sources: sources,
      sourceColor: sourceColor,
      bySource: bySource,
      counts: {
        perSource: perSource,
        both: both,
        vehicles: vehicles,
        pedestrians: pedestrians
      },
      rawObservationCount: rawObjects.length,
      associatedRepresentationCount: associated ? kept.length : 0,
      drawnCount: objects.length
    };
  }

  /**
   * 1 - associated/raw, or N/A when there are no raw observations.
   * Never returns a negative-looking artefact from a zero denominator.
   */
  function associationReduction(rawObservations, associatedRepresentations) {
    var raw = Number(rawObservations);
    if (!isFiniteNumber(raw) || raw <= 0) { return NOT_AVAILABLE; }
    return 1.0 - (Number(associatedRepresentations) / raw);
  }

  /** Format a metric, passing N/A strings straight through. Never shows 0 for absent. */
  function formatMetric(value, digits, unit) {
    if (value === null || value === undefined) { return NOT_AVAILABLE; }
    if (typeof value === 'string') { return value; }
    if (!isFiniteNumber(Number(value))) { return NOT_AVAILABLE; }
    var places = digits === undefined ? 1 : digits;
    return Number(value).toFixed(places) + (unit ? ' ' + unit : '');
  }

  /** One-line rendering of a rolling-window summary from the server. */
  function describeWindow(summary) {
    if (!summary) { return { label: NOT_AVAILABLE, text: NOT_AVAILABLE, empty: true }; }
    var empty = !summary.count;
    var unit = summary.unit || 'ms';
    return {
      label: String(summary.label || ''),
      empty: empty,
      count: summary.count || 0,
      capacity: summary.capacity || 0,
      text: empty
        ? NOT_AVAILABLE
        : ('now ' + formatMetric(summary.current, 1) +
           ' · P50 ' + formatMetric(summary.p50, 1) +
           ' · P95 ' + formatMetric(summary.p95, 1) + ' ' + unit)
    };
  }

  /** Bounded rolling numeric buffer for the browser-side sparklines. */
  function RollingBuffer(capacity) {
    this.capacity = Math.max(1, capacity | 0);
    this.values = [];
  }

  RollingBuffer.prototype.push = function (value) {
    var numeric = Number(value);
    if (!isFiniteNumber(numeric)) { return false; }
    this.values.push(numeric);
    while (this.values.length > this.capacity) { this.values.shift(); }
    return true;
  };

  RollingBuffer.prototype.summary = function () {
    if (this.values.length === 0) {
      return { count: 0, current: NOT_AVAILABLE, p50: NOT_AVAILABLE, p95: NOT_AVAILABLE };
    }
    var sorted = this.values.slice().sort(function (a, b) { return a - b; });
    var rank = function (fraction) {
      var position = Math.max(1, Math.ceil(fraction * sorted.length));
      return sorted[Math.min(position, sorted.length) - 1];
    };
    return {
      count: this.values.length,
      current: this.values[this.values.length - 1],
      p50: rank(0.50),
      p95: rank(0.95)
    };
  };

  root.ScenesenseMapInventory = {
    MODE_RAW: MODE_RAW,
    MODE_ASSOCIATED: MODE_ASSOCIATED,
    MAX_DISPLAY_TRACK_AGE_NS: MAX_DISPLAY_TRACK_AGE_NS,
    CANONICAL_SIZES: CANONICAL_SIZES,
    COLOR_UE_A: COLOR_UE_A,
    COLOR_UE_B: COLOR_UE_B,
    COLOR_BOTH: COLOR_BOTH,
    COLOR_PEDESTRIAN: COLOR_PEDESTRIAN,
    SOURCE_PALETTE: SOURCE_PALETTE,
    NOT_AVAILABLE: NOT_AVAILABLE,
    INCOMPATIBLE_CLOCKS: INCOMPATIBLE_CLOCKS,
    RollingBuffer: RollingBuffer,
    projectInventory: projectInventory,
    associationReduction: associationReduction,
    supportingSources: supportingSources,
    canonicalSize: canonicalSize,
    formatMetric: formatMetric,
    describeWindow: describeWindow
  };
}(typeof globalThis !== 'undefined' ? globalThis : this));

if (typeof module !== 'undefined' && module.exports) {
  module.exports = (typeof globalThis !== 'undefined' ? globalThis : this).ScenesenseMapInventory;
}
