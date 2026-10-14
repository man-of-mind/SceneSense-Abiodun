# Replay-first spatial map foundation

Run from the repository root:

```bash
python3 -m spatial_map_coop.replay_ui_v1.server
```

Open **http://127.0.0.1:8765**. Select a sequence, press Play, or choose a retained
camera frame to pause at its exact installed-map event. Space plays/pauses;
left/right seek one second. The radius starts at 45 m. Heading-up, north-up,
raw/debug, source, observation policy, speed, and an object audit are available.

The default reads the sealed M3 V10 prospective directory referenced by
`experiments/splitfusion_m3_route_b_v10/preregistration_01/PREREGISTRATION.json`.
It needs those existing map, pose, ingest, and retained-input files at their
registered location. `--capture` can point to a relocated copy of that capture;
`--port` changes the local port. Other run/cell identities are refused: the
clock/calibration contract is qualified here only for this recorded V10 session. This entry point has **no live ingest or write
endpoint**, CARLA dependency, inference invocation, or experiment connection.
Existing NumPy, SciPy, Pillow and Flask installations suffice; no frontend
framework or application dependency was added.

## Evidence and demonstration

- [Forward observation view](artifacts/01-forward.png)
- [Raw measurements and object audit](artifacts/02-raw-audit.png)
- [Strict visibility limitation](artifacts/03-visibility-limitation.png)
- [Junction and ego turn](artifacts/04-junction.png)
- [Stopped ego](artifacts/05-stopped-ego.png)
- [Browser demo recording](artifacts/replay-demo.webm)
- [Browser measurements and assertions](artifacts/BROWSER_CHECK.json)
- [Object-by-object source/pose/transform audit](artifacts/OBJECT_TRACE.json)

The roughly 40-second video records real Chromium control interactions and
real-time replay. Browser inspection exercised playback, pause, seek, two map
orientations, radius, raw inspection, strict visibility, four sequences,
correction, fading, expiry, keyboard controls and a narrow viewport. No external
or live-service request occurred. Screenshots preserve dense detector candidates;
there is no confidence cutoff added for appearance.

## What explains the surprising objects

**Reproduced presentation causes:**

1. **Observer location was shifted.** The legacy server labels the forward-biased
   view center `focus_view.ego_pose`. The old viewer follows/draws that as ego.
   Recorded frame 166810 places it **10.5 m forward of the recorded sensor**.
   The new camera center, true capture ego pose, and mounted sensor pose are
   separate values. The legacy sensor position is not itself claimed to be the
   vehicle origin.
2. **Sources were mixed without an observation-region policy.** The existing
   inventory returns all raw objects when the snapshot is not conservative.
   Legacy frame 166812 contains 11 objects, including **5 from `fusion_ego_b`**
   while following `fusion_ego`. The old inventory admits all 11. It also lacks
   a followed-source forward/FoV/range check. Legacy recordings lack sufficient
   capture-clock binding to establish actual camera visibility, so this is a
   reproduced display-policy problem, not proof of a particular occlusion.
3. **Object yaw was lost at installation.** All **112,381 installed objects in
   3,026 V10 events have yaw 90°**. The direct server passes p025 records to the
   baseline normalizer, which reads `yaw_deg`/`model_yaw_deg`, defaults to 0,
   and adds its configured 90° map offset. p025 supplies `yaw_sin`/`yaw_cos`.
   Frame 1480 candidate 0 has upstream yaw **123.879°**, installed model yaw 0°,
   and installed map yaw 90°. The old viewer then replaces vehicle headings
   with the nearest road tangent. This replay labels installed heading as
   unqualified and derives display heading from consistent motion; it never
   rewrites the installed record or presents a road tangent as a measurement.
4. **Arrival-relative prediction omitted existing age.** The old predictor's
   horizon begins at arrival. The sampled V10 observations were already
   **184.5–201.6 ms old at installation**. This replay includes that recorded
   same-host age before advancing its presentation clock.
5. **Dense candidates are present upstream.** Frames 1480, 2020 and 2748 have
   38, 40 and 54 installed objects. Their XYZ and scores match the corresponding
   retained prediction records **exactly**. There are 15, 11 and 109 candidate
   pairs within 1 m, respectively; these are nearby hypotheses, not adjudicated
   physical identities. Raw mode preserves all of them.

**Checks that did not reproduce a cause:** Across those three synchronized
frames (132 objects), multiplying each prediction's local coordinates by the
retained capture-time camera matrix reproduces its world coordinates within
**5.5 micrometers**. Full frame/source/capture/run/cell joins, sensor calibration
hashes and the 1.8 m / 1.55 m / −4° camera mount pass. These samples do not support
an axes swap, yaw-unit error in the world transform, wrong mount, or stale-pose
explanation. This is a small sample, not a whole-capture transform qualification.

The reconstructed RGB inputs and top-six predicted image boxes were inspected
at [1480](artifacts/camera-1480-prediction-boxes.png),
[2020](artifacts/camera-2020-prediction-boxes.png), and
[2748](artifacts/camera-2748-prediction-boxes.png). They show, respectively, traffic
including a bus, the junction/police vehicle, and the stopped-ego junction view.
No GT actor IDs or GT matches were used to repair or identify detections.
Actual false positives, per-object physical identity, depth/localization errors
and general occlusion remain unresolved.

Legacy traces contain repeated source-frame snapshots (2/21 and 1/36). New
presentation age is independent of polling: duplicate and out-of-order events
cannot refresh an observation, and silence advances age until expiry.

## Display and clock contract

| Setting | Policy |
|---|---|
| Default view | Latest observations from the selected source only; explicit unknown visibility |
| Geometry | Original center tested with its own capture pose; full calibrated camera horizontal/vertical center frustum plus forward/range checks; predicted center rechecked against the displayed sensor region and ego-forward half-plane |
| FoV | 120° from the hash-validated 1280×720 camera intrinsics; no fallback angle |
| Range | 120 m configured radar acquisition/preprocessing envelope, explicitly labelled; not a calibrated camera visibility distance or guarantee of detection |
| View radius | 45 m initially; 20–90 m control, independent of sensor range |
| Current freshness | At most 350 ms capture age; tracks absent from the latest frame are coasting and withheld |
| Prediction | At most 500 ms **total capture age**, including pre-arrival age; then holds |
| Correction | 120 ms; residual over 3 m resets display motion and fades the new display generation in, rather than drawing an unsupported traversal |
| Motion | Existing class speed caps; two measured displacement intervals must agree within 6 m/s; average those intervals; latest speed below 0.25 m/s stops extrapolation; gaps over 500 ms have no velocity |
| Heading | Motion estimate, retained through stopping, at most 60°/s; unknown initially; road tangent contributes only 15% when within 20° of a moving object's heading |
| Coasting | Optional separate policy, explicitly labelled; fades from 500 to 750 ms and is removed at 750 ms |
| Visual size | Existing canonical class sizes; fixed screen-size bounds; measured coordinates are never snapped to lanes |
| Memory | At most 256 events per clip, 256 presentation tracks, 8 sources, 600 timing samples; backend LRU of 3 clips and 3 images; pose/event index also bounded |
| Confidence | No additional score gate, no deduplication or quality-based filtering |

The processing/rendering projection is isolated from the production map.
`ConservativeTrackStore` and its existing one-to-one class-gated association run
on a private replay copy at capture times. IDs are **presentation associations**,
not guaranteed actor identities; dense nearby candidates can remain ambiguous.
Server/source provenance and raw measurements remain attached to each entry.
Seeking or changing sequence resets browser tracking and causally reconstructs
only the preceding second. Rebuilding a seek gives the same rendered positions.

Production capture nanoseconds and installation wall seconds share the recorded
host clock in this V10 capture, as documented in its `analysis_01/CLOCK_MAPPING.json`.
The adapter subtracts integer nanoseconds before conversion to browser numbers;
absolute ns identities travel as strings. Browser `performance.now()` advances
only replay elapsed time. Ego and current objects use the same replay display
time; correction offsets are explicitly presentation-only. Expired/coasting
poses hold bounded historical estimates and are not current observations.

A missing clock mapping uses explicitly labelled arrival-relative animation;
measurement age is unknown. A missing/mismatched/invalid frame pose disables FoV
and ego observations. No guessed pose is drawn. A previously valid map center
may remain as a navigation reference, without an ego/FoV claim.

**FoV is not visibility.** Map building outlines, single rays, model confidence,
and GT actor AVO labels are insufficient to assign visibility to these detector
candidates. The adapter therefore supplies `unknown` for every recorded object.
The strict-visible policy honestly withholds them and displays the limitation.
The pure policy also supports explicitly reviewed evidence bound to the exact
source/frame/capture/object/pose, tested with synthetic fixtures; no such reviewed
visibility evidence is claimed for these recordings. Native sensor/GT evidence
is not silently converted into a general occlusion classifier.

## Validation

```bash
node spatial_map_coop/tests/js/replay_tests_v1.js
node spatial_map_coop/tests/js/presentation_tests_v1.js
python3 -m unittest spatial_map_coop.tests.test_replay_ui_v1 \
  spatial_map_coop.tests.test_viewer_presentation_v1 \
  spatial_map_coop.tests.test_coop_input_v1
python3 -m spatial_map_coop.replay_ui_v1.audit
```

Results: **23 new JS scenarios**, **105 existing JS checks**, **12 new Python
checks**, and **99 existing viewer/adapter Python checks**, all passing. These
are focused presentation/adapter checks; no experiment qualification suite ran.
Python replay tests require the existing registered V10 recording. Tests hash
map/pose/ingest inputs before and after repeated replay and camera reads and
assert that raw measurements and scientific files are unchanged.

Browser verification uses Playwright as development tooling only, with a running
local replay server:

```bash
PLAYWRIGHT_MODULE=/path/to/playwright \
PLAYWRIGHT_BROWSERS_PATH=/path/to/playwright-browsers \
node spatial_map_coop/replay_ui_v1/browser_check.cjs
```

The saved run used headless Chromium 133, 1440×1100 at DPR 1, with video capture.
The final warmed 600-frame sample measured **60.00 fps**, frame interval P95
**16.8 ms**, and render-work P95 **1.4 ms**. The animated junction inspection
sampled 751 frames, 337 with ego movement between incoming updates, and a 64.6°
ego turn. Fading and eventual removal were observed on the no-message tail.
There were no page errors or non-local requests. These are measurements of this
browser/host run, not a live-service or GPU performance qualification.

Inspection found a 15.4 m correction residual on a noisy candidate, causing a
3.32 m frame step. The consistency/reset guards above reduced the maximum
within-display-generation step to **0.74 m** on the repeated junction run.
Display resets are explicit in the audit; model/association uncertainty remains.

## Remaining live integration

1. Feed a bounded copy **after successful map installation**, without polling or
   calling a stateful scientific snapshot builder to refresh observations.
2. Bind full run/cell/stream/frame/capture identity using the existing
   `coop_input_v1` normalization and frame-keyed pose-sidecar seam. Keep frozen
   `splitfusion_direct_object_map_update.v1` unchanged; attach camera mount,
   intrinsics and configured range from validated calibration separately.
3. Register and validate the actual live clock-domain mapping and reset behavior.
   The offline same-host mapping cannot be assumed for a different host/clock.
   Without it, explicitly retain the arrival-relative limitation.
4. Preserve source ownership. Remote-only observations need a separately labelled
   cooperative mode, never the current ego-observation inventory.
5. Resolve the installed yaw conversion loss upstream in a separately reviewed
   integration change, if qualified measured yaw is needed. This UI does not
   change the production normalizer or historical measurements.
6. Obtain qualified per-observation visibility evidence before making a
   visible-only claim. The current capture's model errors and crowded candidate
   identities also need separate perception/tracking work.

No live experiment was connected, launched or changed. No training, GT/scorer,
historical metric, commit, push or deletion operation was performed.
