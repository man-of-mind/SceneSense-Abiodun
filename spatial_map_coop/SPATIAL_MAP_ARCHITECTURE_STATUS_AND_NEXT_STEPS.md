# Cooperative Spatial Map: Architecture Status and Next Steps

Date: 2026-08-25  
Purpose: advisor handoff and implementation roadmap

## Executive summary

The project has reached the end of the **multi-vehicle visualization stage** and
the beginning of the **cooperative reasoning stage**.

The latest demonstrated system can run two moving ego vehicles on the same
route, receive object detections from both, transform those detections into a
common CARLA world frame, and render both egos and their detections on an
ego-following top-down map. The browser deliberately colors detections by
source, so duplicate reports of the same physical object remain visible. This
is the correct “before cooperative fusion” demonstration.

There different prototype implementation:

- the live server contains a greedy class/distance/dimension association,
  score-weighted position averaging, nearest-track matching, and exponential
  smoothing;
- `spatial_map_geometry` contains a second greedy association baseline and a
  two-dimensional FoV-overlap disagreement reasoner;
- `cooperative_fusion` contains mean, weighted-mean, bearing-triangulation, and
  dimension-fusion experiments.

These pieces are not yet one validated moving-ego cooperative pipeline. The
two-color viewer renders the **raw** reports, the live association/fusion is a
heuristic baseline with little multi-source support in the recorded traces,
and the strongest triangulation results came from controlled static scenes
with oracle-assisted association. Real occlusion classification and the
warning feedback loop remain unbuilt.

The next bounded milestone should therefore be:

> Time-align two ego streams, associate observations with a gated Hungarian
> assignment, produce one provenance-preserving track per physical object,
> and compare highest-confidence selection, covariance-aware position fusion,
> and geometry-gated triangulation on the same recorded/ground-truthed scenes.

## Current end-to-end data path

```text
RGB + radar on each ego
        |
        v
split perception inference and object decoder
        |
        | local XYZ + class + score + image center
        v
camera-to-world transform in the decoder/client             IMPLEMENTED
        |
        | world XYZ + ego camera pose/FoV + provenance
        v
UDP multi-stream ingest, freshness filtering, map snapshot  IMPLEMENTED
        |
        +--> raw source-colored moving-map viewer            DEMONSTRATED LIVE
        |
        +--> heuristic clustering/weighted mean/EMA          PRESENT, NOT VALIDATED
        |
        v
time alignment -> assignment -> fused tracks                NEXT MILESTONE
        |
        v
per-ego visibility and occlusion reasoning                  SYNTHETIC BASELINE ONLY
        |
        v
hazard classification and feedback to a blind ego           NOT IMPLEMENTED
```

## Status by architecture component

| Component | Current status | What the code actually does | Main gap |
|---|---|---|---|
| Per-ego perception | Working research pipeline | RGB-radar model produces class heatmaps, local XYZ, dimensions, yaw, scores, segmentation | Precision/localization limitations remain model concerns, but the interface exists |
| Coordinate transform | Implemented live | `decode_objects()` multiplies learned sensor-local XYZ by the live camera-to-world matrix | No explicit pose/calibration uncertainty; GPS/pose error is not propagated |
| Object packet and transport | Implemented live | Sends stream/frame/time, camera pose and matrix, FoV, detections, segmentation summary, and latency over UDP | Needs a stable cooperative observation contract containing bearing and uncertainty |
| Multi-stream state | Implemented | Server keeps the latest packet per stream and expires stale streams | It keeps “latest” states, not a timestamp-aligned multi-stream buffer |
| Moving global map | Demonstrated live | Common Town10HD world frame, static roads/buildings, followed-ego ROI, all ego markers | Primarily visualization, not yet a validated world model |
| Source-colored two-ego view | Demonstrated live | Canvas draws `raw_spatial_map_objects` in a different color per stream | Duplicate boxes intentionally remain |
| Cross-ego association | Prototype only | Live server uses score-ordered greedy class/XY/dimension clustering; geometry scaffold has a similar 3 m greedy baseline | No global one-to-one optimization, uncertainty gate, time compensation, or association metrics |
| Measurement fusion | Prototype only | Live server uses score/radar-weighted arithmetic means; separate module implements mean, information weighting, and bearing triangulation | Triangulation is not connected to the moving map; confidence is not a calibrated covariance |
| Temporal tracking | Rudimentary baseline | Nearest same-class track within 6 m plus EMA smoothing and a stale timeout | No velocity state, Kalman prediction, lifecycle logic, or out-of-sequence handling |
| FoV geometry | Implemented as 2D scaffold | Builds ground-plane wedge polygons, intersections, point-in-polygon tests | Not a true 3D frustum or visibility polygon |
| Occlusion reasoning | Synthetic baseline | “Seen by A, inside B FoV, missing from B” emits a possible-occlusion/miss hypothesis | No ray/occluder test, visibility grid, real GT metrics, or reliable real-scene classification |
| Alerts/feedback | Not implemented | Design notes only | Needs hazard policy, target-ego coordinates, message protocol, and safety evaluation |

## What the latest evidence proves

### Live two-ego map

`autonomous_run/figs/stage2_LIVE_two_ego.png` and
`recordings/two_ego_live.jsonl` prove that two moving egos can contribute to
the same map. The selected live replay frame shows four reports from
`fusion_ego` and seven from `fusion_ego_b`, with different colors and both ego
positions visible.

The trace has 21 snapshots; both streams are simultaneously fresh in 9 of
them. The longer `two_ego_occl.jsonl` trace has 36 snapshots, with both fresh
in 28. This asynchronous availability is a concrete reason to add timestamp
buffering before more sophisticated matching.

The live server's heuristic found a two-source cluster in only 2 of the 21
`two_ego_live` snapshots and 4 of the 36 `two_ego_occl` snapshots. That is
evidence that the baseline exists, not evidence that association is solved.

### Cooperative fusion research

`cooperative_fusion/fusion.py` implements:

- arithmetic mean of world positions;
- score/assumed-variance weighted mean;
- least-squares closest-point triangulation of camera bearing rays;
- view-geometry-weighted dimension fusion.

The controlled Phase-2 experiment reported 1.40 m car XY error from
two-view box-center triangulation versus 2.09 m for its radar reference.
Phase 2b reported a clear pedestrian result of roughly 0.26–0.35 m at useful
baselines. However, these were static, small-object-count experiments with
oracle-assisted association and specially derived bearings. They establish
that the geometry can help; they do not establish that it works in the
moving, asynchronous, multi-object map.

### Occlusion groundwork

The synthetic truck/pedestrian scene passes the FoV-disagreement baseline with
precision and recall of 1.0, but it contains only a known constructed case.
The real-trace figure produces several unverified hypotheses and visibly
illustrates the expected over-flagging. Real occlusion performance is
therefore unknown.

One old note says the live server still needs to expose every stream's pose and
FoV. That part has since been added to `active_streams`. Per-stream sensing
range and a visibility/occluder representation are still missing; the replay
adapter currently supplies a fixed range.

## Relevant files and how to use them

### 1. `spatial_map_coop`: current integration hub

| File | Relevance |
|---|---|
| `spatial_map_server_moving_ego.py` | Main live multi-stream server, heuristic clustering/fusion, track smoothing, APIs, static map, ego-following ROI, raw two-color canvas |
| `README.md` | Reconciled runbook and honest Stage 1–4 boundary |
| `record_trace.py` | Captures `/api/spatial_map/latest` snapshots without coupling later work to CARLA |
| `replay_trace.py` | Re-renders recordings offline and preserves the source-colored “before fusion” view |
| `synthetic_scenes.py` | Deterministic two-ego duplication and truck/pedestrian occlusion scenes |
| `stage3_occlusion.py` | Converts synthetic or recorded snapshots into `LocalSensorMap` inputs and runs the baseline reasoner |
| `recordings/two_ego_live.jsonl` | Primary real two-ego replay evidence |
| `recordings/two_ego_occl.jsonl` | Longer both-ego trace used for real FoV hypothesis rendering |
| `autonomous_run/GALLERY.md` and `figs/` | Visual evidence and explanations, not runtime components |
| `SPATIAL_MAP_PRESENTATION.md` | Existing presentation narrative; useful background, but its “not built” statements should be interpreted as “not integrated and validated” where heuristic code exists |

### 2. Live perception and coordinate conversion outside the folder

| File | Relevance |
|---|---|
| `../carla_split_inference_udp_fusion_object_pole_client_spatial_stream_oai.py` | Live ego client: split inference, camera pose capture, spatial packet creation, and UDP publishing |
| `../pole_lraspp_multimodal_fusion/pole_lraspp_multimodal_fusion/object_targets.py` | Decoder: heatmap peaks, class-aware image NMS, local XYZ regression, and `camera_matrix × local_xyz` world conversion |
| `../real_time_spatial_map_server_fusion_object_v2.py` | Baseline ancestor/reference for the moving-ego server; it should not be the main place for new cooperative research changes |

The client already sends `center_px` and the camera matrix. The moving-map
server currently discards `center_px` while normalizing observations. A clean
next interface should preserve a world bearing ray directly, or preserve the
pixel center plus intrinsics and pose so the server can reconstruct it.

### 3. `spatial_map_geometry`: reusable reasoning scaffold

| File | Relevance |
|---|---|
| `schemas.py` | `SensorPose2D`, `SpatialObject`, `LocalSensorMap`, association and occlusion result types |
| `geometry.py` | 2D FoV wedges, polygon intersection/area, point-in-polygon, and object footprints |
| `association.py` | Simple greedy class-aware XY association baseline |
| `occlusion_reasoner.py` | Conservative overlap/disagreement hypotheses; explicitly not proof of occlusion |
| `demo_two_view_overlap.py` | Small offline association/overlap demonstration |
| `live_visibility_server.py` | Standalone HTTP/visual prototype using the geometry scaffold; useful for experiments, but separate from the moving-ego server |

### 4. `cooperative_fusion`: algorithm research, not the current live map

| File | Relevance |
|---|---|
| `fusion.py` | Reusable mean, variance-weighted, triangulation, and dimension-fusion mathematics |
| `phase2_two_view_fusion.py` | Controlled two-camera car-position experiment; documents bearing construction and oracle association |
| `phase2b_full_fusion.py` | Controlled car/person, dimension, baseline-sweep, and multi-frame experiment |
| `RESULTS_fusion_module.md` | Synthetic and early fusion results |
| `RESULTS_phase2_two_view.md` | CARLA-controlled fusion results and limitations |
| `MORNING_SUMMARY_20260627.md` | Concise handoff for the strongest controlled fusion results |

The `phase0*`, deployment-video, and radar-PPS figure files provide historical
model context but are not the main implementation path for the cooperative map.
Generated PNGs, recordings, and `__pycache__` files are evidence/artifacts, not
architecture modules.

## Matching recommendation: Hungarian first, JPDA later

Use a gated Hungarian assignment as the next real matching layer. It gives a
globally consistent one-to-one assignment between observations from two
streams and is explainable and easy to evaluate. A useful cost should include:

1. hard class compatibility;
2. time-aligned world-XY distance, preferably Mahalanobis distance when
   covariance is available;
3. a maximum physical-speed/time gate;
4. optional weak size/heading evidence;
5. source freshness and field-of-view consistency.

Dimensions should be a weak cue because the current map deliberately replaces
unreliable learned sizes with canonical footprints. Raw confidence should not
be treated as localization accuracy unless it is calibrated.

JPDA (Joint Probabilistic Data Association) becomes worthwhile later when
several similar vehicles cross closely and assignment ambiguity persists over
multiple frames. It is unnecessary complexity for the first two-ego replay
gate. A Kalman track prediction plus Hungarian assignment is the appropriate
v1.

## Fusion recommendation: one track, all evidence retained

The global map should contain one canonical track per physical object, while
retaining every contributing observation in its provenance. Do not let every
raw report update the map as a separate object, and do not make
highest-confidence selection the normal fusion rule.

Recommended policy:

1. Associate observations and move them to a common timestamp.
2. If two reliable bearing rays have adequate baseline and intersection angle,
   use weighted least-squares triangulation and reject ill-conditioned
   geometry.
3. Otherwise fuse world-XY measurements using measured per-class/per-range
   covariance. Because egos use the same model and may have correlated errors,
   Covariance Intersection is safer than pretending all errors are independent.
4. Feed the resulting measurement to a constant-velocity Kalman track.
5. Keep the highest-confidence observation as a fallback when other reports
   are stale, gated out, geometrically ill-conditioned, or inconsistent.

Highest-confidence-only selection is simple, but it throws away complementary
geometry, causes source-switching jitter, and assumes model confidence predicts
position accuracy. Naive averaging is also insufficient because two biased or
correlated estimates do not automatically become accurate. The right research
comparison is therefore an ablation on the same matched detections:

`best source` versus `mean` versus `covariance/CI` versus `triangulation`.

## Correct occlusion logic

The intersection of two FoV polygons does **not** mean every object there must
be reported by both cars. A missing report can be caused by occlusion, range,
vertical FoV, an edge-of-frame case, a detector false negative, stale timing,
pose error, or a false positive from the reporting car.

For an object reported by A but missing from B, the reasoner should ask:

1. Were A and B observations fresh and aligned to the same time?
2. Is the object inside B's calibrated horizontal/vertical FoV and reliable
   detection range?
3. Was there truly no associated B observation?
4. Does the ray from B to the object intersect a building or vehicle footprint
   closer to B than the object?

If the ray is blocked, label the location `occluded/unknown for B`. If the ray
is clear, label it `unconfirmed detector miss` rather than occluded. A local
free/occupied/unknown visibility grid can later replace or complement the
ray test, especially for radar-native reasoning.

The first implementation can be 2D ground-plane ray casting against CARLA
building polygons plus associated vehicle footprints. True 3D frusta and
height-aware occlusion can follow after the 2D method is measured against
CARLA ground truth.

## Minimal next-step plan

1. **Freeze an offline replay gate.** Use both committed two-ego traces and a
   small CARLA-GT-labelled cooperative scene. Do not depend on a live simulator
   for every algorithm iteration.
2. **Fix observation timing and geometry inputs.** Buffer by CARLA timestamp;
   retain camera bearing/intrinsics, sensing range, and per-observation
   covariance/provenance through server normalization.
3. **Implement and evaluate Hungarian association offline.** Compare it with
   the existing greedy 4 m baseline. Measure association precision/recall,
   duplicate-collapse rate, false-merge rate, and latency.
4. **Create one canonical track per object.** Add constant-velocity prediction,
   lifecycle states, source membership, and raw-observation history.
5. **Run the fusion ablation.** Compare best-source, mean, CI/weighted fusion,
   and condition-gated triangulation on exactly the same associations. The gate
   is fused localization non-inferior to the best contributing view, with no
   loss of class/recall and acceptable latency.
6. **Add occluder-aware reasoning.** Start with 2D ray-versus-building/vehicle
   footprints, then evaluate possible-occlusion, true-occlusion, and clear-ray
   detector-miss labels against CARLA visibility GT.
7. **Integrate into the live moving map.** Keep UI switches for raw-by-source,
   associations, fused tracks, FoVs, and occlusion state so each layer remains
   explainable.
8. **Only then add warnings.** Send a hazard to an ego only from a fresh,
   sufficiently certain fused track with a validated visibility state.

## Advisor-ready conclusion

The project has already solved the integration question: multiple moving cars
can publish lightweight object reports into one common-world, ego-following
map. It also has working starter mathematics for association, triangulation,
and FoV reasoning. The remaining research contribution is to turn those
separate prototypes into a measured cooperative world model: align streams,
associate objects reliably, fuse them into stable tracks, distinguish true
occlusion from detector misses, and validate the resulting warnings.

In one sentence: **the moving multi-car map works; reliable cooperative object
identity, state estimation, and visibility reasoning are the next build.**
