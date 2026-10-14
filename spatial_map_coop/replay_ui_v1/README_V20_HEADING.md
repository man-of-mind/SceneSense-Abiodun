> **Superseded for vehicle glyph orientation and the default view by [README_V20_2_ORIENTATION.md](README_V20_2_ORIENTATION.md).**
> This v20.1 display is frozen in `baseline_v20_1/`; its evidence below is unchanged.

# V20 offline display refinement

Ready for user review before M4. The v3.1 layout, framework, pedestrian policy and recorded evidence are preserved.
V20 remains **STOP**. This work changes presentation only; it does not repair or reclassify the experiment.

Open the [short normal-speed comparison](artifacts_v20_heading/v20-heading-before-after-1x.webm), or start the offline viewer:

```bash
OPENBLAS_NUM_THREADS=1 python3 -m spatial_map_coop.replay_ui_v1.recorded_tap_v20 --port 8776
# http://127.0.0.1:8776/compare-heading  (before | after)
# http://127.0.0.1:8776/                 (current display)
```

The default capture is the V20 worktree path identified in its report; `--capture PATH` overrides it. This adapter
reads the sealed map/pose taps, reuses the live viewer's exact pose/producer binding, and preserves recorded
capture-to-install intervals. It does not start LiveFeed, CARLA, OAI or inference. Browser clips contain 76–186 events,
below the existing 256-event bound. Recorded camera images are not supplied by these taps. Clip IDs are local to a
causal replay/reseek; continuous-route IDs below come from the full diagnostic pass. Installation timing is retained;
the original browser's transport/scheduling delay is not reconstructed.

## Diagnosis before edits

The [frame-level evidence](artifacts_v20_heading/DIAGNOSIS_BEFORE.json) retains raw qualified yaw, score, measurement
position, display ID/position/heading, motion fit, pending votes and geometry for these cases. Approximate original
video times are mapped from installation stamps; use frame numbers for exact identification.

| Case | Original video / frames | Confirmed finding |
|---|---|---|
| Parked/stationary example, T246 | 229–239 s; 3490–3620 | Same display identity throughout, 57 matched updates; median-anchor displacement 1.63 m. Producer yaw changes greatly while the display motion state stays stationary. At 3552/3554/3556, qualified raw yaw is −2.0° / 106.6° / 36.2°, scores .644 / .739 / .845. Pending votes advance 1 / 2 / accept, although those observations disagree. The rolling mean was allowed to accept a new axis without consistent fresh support. |
| Moving and turning, T325 | 318–341 s; 4770–5090 | A coherent observed trajectory bends from south toward west: approximately (110, −25) to (47, −68). The same display ID persists. Motion heading follows the turn; producer yaw often disagrees strongly. This is evidence of a real trajectory turn, not a calibrated ground-truth heading measurement. |
| Moving source handoff, T325 | 345–350 s; 5140–5200 | While still moving straight west, fitted speed drops through 2.5 m/s: 2.88 at 5186, 2.42 at 5188, then 2.28 / 2.27 / 2.49. The old heading switches from motion to producer and back at 5196. This threshold handoff is independent of an ID change. |
| Building-centre overlap | 278.3 s; 4192–4194 | A tentative vehicle's **raw** centre (63.976, 119.804) is inside approximate polygon 436, only .288 m from its west edge and .571 m from its north edge. The display uses that centre. Its 5.01 × 2.05 m footprint is not enlarged by the renderer's pixel minimum at the demonstrated viewport/radius. |
| Second building edge | 596.7 s; 8486 | Raw centre (−16.284, 44.062) is inside polygon 223, .020 m from its north edge. Approximate footprint contact cannot establish whether the detection or building boundary is wrong. |

All **3,576/3,576** frames pass the existing exact pose and producer-record joins. These validate binding, not detector
accuracy. The static map uses the same world-XY projection as glyphs; ego-to-road P95 distance is .982 m, with no fitted
transform. No global coordinate error or pixel-size defect was demonstrated in these examples. The buildings are
filtered environment bounding boxes, not facade meshes or occupied 3D volumes. Detector position error and backdrop
approximation remain unresolved explanations. Existing visibility policy intentionally retains intersecting footprints
as uncertain; deleting them would conceal that uncertainty.

Code inspection and fixtures also confirm that the old moving-heading path interpolated front/back reversals through
the body, and that a held producer update could cut short the previous interpolation. No claim is made that either
explains every vehicle in the original recording.

## Changes and trade-offs

- `replay_tracks_v3.js`: a vehicle-only heading path requires four consecutive strong, qualified producer updates
  supporting one pending axis within 12°, spanning at least 350 ms, before changing a held axis by ≥20°. Contradictory,
  weak or missing updates reset pending support. The accepted target uses a recent 800-ms axis mean with resultant ≥.9.
  Existing interpolation continues while held. Sustained detector bias can still be accepted; first-heading error is
  not corrected by pretending the road is ground truth.
- Motion-heading entry requires three qualifying motion updates. Once entered, it remains active through the old
  2.5-m/s threshold crossing, until motion stops or falls below the existing .6-m/s / significance-1.5 exit conditions.
  It follows the same causal velocity fit through turns. Entry can take roughly two additional update intervals.
  Vehicle motion estimation itself, positions, pedestrian estimation and confirmation thresholds are unchanged.
- Body rotation takes the shortest **modulo-180°** path. A separate producer-supported front/back marker can swap ends
  without rotating the body. Travel bearing remains separate in Inspect, so reverse travel does not imply a reversed
  vehicle front. Stationary centroid drift cannot supply a front marker. Missing/unqualified heading retains the last
  axis without front certainty; a new track with no heading stays unoriented. Matching's vehicle yaw penalty also uses
  the body axis rather than front/back direction.
- Both replay and live hide low-score haze/floating labels by default. **Inspect → Show low-confidence hypotheses haze
  and labels** restores them; closing Inspect hides them again. Raw candidates, dispositions and counts remain. The
  separate pedestrian unresolved-motion halo and established-track conflict envelopes remain drawn.
- Inspect includes the complete heading state, raw heading qualification/contract, producer join provenance, displayed
  angle, and static-visibility contact details with polygon IDs. **No new geometric withholding rule** was added.
  Building policy and object centres are unchanged. A small canvas-size guard fixes an observed startup/resize race
  that could otherwise stop the render loop with a negative-radius exception.

## Same-input results

These are presentation diagnostics, not scientific quality scores. The full route runs at recorded installation
times, 1×, sampled at 60 Hz, 45 m forward view, existing building withholding enabled. Stationary means the existing
display motion classification; there is no GT-based selection. See [60-Hz results](artifacts_v20_heading/COMPARISON_60HZ.json)
and [update/disposition results](artifacts_v20_heading/COMPARISON_UPDATES.json).

| Measure | Preserved v3.1 | Refined |
|---|---:|---:|
| Stationary axis variation, degrees / stationary track-second | 5.861 | 3.856 (−34.2%) |
| Mean angular step, 60-Hz vehicle samples | .226° | .182° |
| Vehicle angular steps >20° between matched updates | 112 | 54 |
| Stationary update-angle P95 | 10.016° | 1.934° |
| Selected parked track total axis variation, all matched updates | 139.0° | 107.7° |
| Selected turn: display lag at −135° crossing, relative to causal motion fit | 33 ms | 33 ms |
| Position step P95 / maximum, 60 Hz | .13076 / 5.087 m | .13105 / 5.087 m |
| Display tracks created / established, all classes | 1,961 / 678 | 1,953 / 676 |
| Visible appearances over the route, all classes | 2,232 | 2,222 |
| Absorbed raw candidates, existing footprint policy | 20,220 | 20,283 |
| Conflict track-samples at 60 Hz | 6,699 | 6,695 |
| Existing building withholding, 60-Hz track-samples | 33 | 33 |
| Pedestrian samples differing in position, heading, motion, tier, conflict or visibility | — | **0 / 31,305 frames** |
| Browser frame interval P95, standalone | 33.3 ms | 33.3 ms |
| Browser draw cost P95, standalone | 16.3 ms | 16.3 ms |

Both versions already limit individual 60-Hz angular steps to 1.5° (90°/s); the >20° row measures **update-to-update**
steps instead. Reduced variation does not prove more accurate yaw. The motion fit itself has estimation lag; 33 ms is
only the extra displayed-axis lag at the selected crossing. Of 197,557 samples matched by latest raw candidate,
3,413 have differing positions due to changed vehicle matching/footprint histories, despite unchanged position code.
Raw-candidate absorption increases by 63 (.31%); full reason counts are retained. This is a real heading/grouping
interaction, not an overlay count reduction. Adjacent parked-car fixtures retain two separate IDs through alternating
yaw and 180° ambiguity, and route-level track creation/appearances do not increase. Scientific identity accuracy is
not assessed here.

The browser was inspected at 2560×1000 for the short side-by-side video and 1440×1000 without recording for render
timing. These software-rendering timings describe this offline check, not V20 host qualification. [Browser checks](artifacts_v20_heading/BROWSER_CHECK.json)
pass with zero page errors, and both replay and the actual live renderer pass the default/toggle/unresolved-person
checks. The live renderer is exercised with local recorded-message fixtures, not a running pipeline.

Screenshots: [parked interval](artifacts_v20_heading/parked-comparison.png), [turn](artifacts_v20_heading/turn-comparison.png),
[building overlap at frame 4192](artifacts_v20_heading/building-comparison.png), [Inspect](artifacts_v20_heading/inspect-overlay.png),
[offline live-renderer fixture](artifacts_v20_heading/live-renderer-offline-fixture.png).

## Reproduce the checks

Use a new export directory; export is create-only and does not rewrite the capture:

```bash
OPENBLAS_NUM_THREADS=1 python3 -m spatial_map_coop.replay_ui_v1.recorded_tap_v20 --export /tmp/v20-review-input
node spatial_map_coop/replay_ui_v1/heading_diagnostics_v20.cjs /tmp/v20-review-input /tmp/v20-before before
node spatial_map_coop/replay_ui_v1/heading_diagnostics_v20.cjs /tmp/v20-review-input /tmp/v20-after after
node spatial_map_coop/replay_ui_v1/heading_compare_v20.cjs /tmp/v20-review-input /tmp/v20-comparison.json
node spatial_map_coop/tests/js/replay_vehicle_heading_v20.js
node spatial_map_coop/tests/js/replay_tracks_v3.js
node spatial_map_coop/tests/js/replay_pedestrian_motion_v3.js
node spatial_map_coop/tests/js/replay_refinements_v2.js
node spatial_map_coop/tests/js/replay_tests_v1.js
node spatial_map_coop/tests/js/presentation_tests_v1.js
OPENBLAS_NUM_THREADS=1 python3 -m unittest spatial_map_coop.tests.test_live_feed_v1 spatial_map_coop.tests.test_replay_heading_v2 spatial_map_coop.tests.test_replay_ui_v1
# With the offline server above running:
PLAYWRIGHT_MODULE=/home/shr_aisvcs/.cache/ms-playwright-go/1.50.1/package \
PLAYWRIGHT_BROWSERS_PATH=/tmp/scenesense-replay-browser \
node spatial_map_coop/replay_ui_v1/browser_heading_v20.cjs --checks-only
# Omit --checks-only to regenerate the short comparison video as well.
```

Passed: 15 focused new cases, 165 existing JS checks, 27 existing Python tests, exact joins, full-route pedestrian
equivalence and browser checks. No experiment qualification suite was run. The baseline assets are frozen with a
[hash manifest](baseline_v3_1/manifest.json). Input hashes and the preserved V20 report/video hashes are in
`artifacts_v20_heading/INPUTS.json` and `PRESERVED_EVIDENCE.json`. Earlier artifacts and the original video remain.

Remaining limitations: coherent producer bias can still give a wrong parked axis; noisy vehicle positions can still
produce an erroneous moving fit; heading-sensitive grouping can change vehicle histories; approximate buildings
cannot classify detection truth. These limits are visible rather than resolved by road snapping, invented centres or
new suppression. **Future live viewing and recording must run off the experiment host**, as documented in
[README_LIVE.md](README_LIVE.md). No new resource study, live capture or M4 implementation was performed.
