# Ego traffic view v3: tiered display tracks

> Vehicle orientation and the default hypotheses overlay have since been refined using recorded V20 evidence.
> See [README_V20_HEADING.md](README_V20_HEADING.md) for the current policy, comparison and limits. The v3 results below are historical.

> **Update v3.1 (pedestrian motion) and erratum:** see [README_V3_PEDESTRIAN.md](README_V3_PEDESTRIAN.md). The archive's pedestrian `object_speed_mps` is always 0. The replay pedestrians described below as standing were actually walking at 0.9–1.4 m/s, and the stationary-actor rows in the table below include them. Corrected vehicle-only values are in `artifacts_v3_ped/DISPLAY_COMPARISON_V3_1.json`.

```bash
python3 -m spatial_map_coop.replay_ui_v1.server --open     # from the repository root
```

- `/` is the v3 view.
- `/v2` is the previous UI.
- `/compare-v3` shows v2 and v3 side by side.
- `/before` and `/compare` are the preserved v1 and v1|v2 pages.

The app is read-only: it starts no CARLA, OAI, inference or ingest.

Artifacts are in [`artifacts_v3/`](artifacts_v3/):

- [normal-speed v2|v3 video](artifacts_v3/v2-vs-v3-1x.webm): departure, turn, stopped ego and stream expiry. Both sides use 1×, 45 m and heading-up.
- Screenshots 01–06.
- [Synchronized upstream examples](artifacts_v3/07-upstream-limit-synchronized.png).
- [Deterministic metrics](artifacts_v3/DISPLAY_COMPARISON_V3.json).
- [Browser check](artifacts_v3/BROWSER_CHECK_V3.json).

## What was wrong

1. **The renderer drew every candidate as a car, and most candidates are not cars.** The map gets the raw FCOS output with no score gate: 35–53 candidates per frame, median score 0.05. Only 2–5 per frame reach the project's registered evaluator thresholds (`splitfusion_minimal_quality_scorer_v1`: vehicle ≥ 0.20, person ≥ 0.25).
   - Against the archived GT, only 9–20% of sub-0.08 candidates lie within 3 m of any dynamic actor or parked car.
   - For candidates ≥ 0.2, that share is 59–80%.
   - All of them were drawn as equally certain vehicle outlines, at about 6 px/m.
2. **The "hundreds of new display IDs" were mostly flicker in those low-score candidates.** They were not a bookkeeping bug.
   - 80–92% of v2 display IDs were born from candidates below 0.08, and a typical one lived 2–3 frames.
   - Because every candidate entered one global assignment, even IDs born from confident detections lived a median of only 2 frames: low-score neighbours kept taking over the match.
3. **The detections themselves are noisy at the display scale.**
   - Range error is ±1–2 m and drifts as the ego approaches, so parked objects appear to slide at 1–3 m/s.
   - Duplicate range hypotheses appear 2–3 m apart.
   - A qualified heading exists in only about half of the frames.

## Presentation policies (private display layer; raw, association, GT and metrics untouched)

| Policy | Rule |
|---|---|
| Confident vs low-confidence | Registered evaluator thresholds (vehicle 0.20, person 0.25) |
| Tracked object | Confirmed by ≥ 2 confident detections within 3 frames |
| Tentative | Drawn immediately after 1 confident detection, as a dashed outline; dropped after 2 misses |
| Low-confidence detections | Can sustain a tracked object for ≤ 6 frames (tight 1.5 m gate); can never create one |
| Unresolved hypotheses | Low-confidence detections not explained by a track. Drawn as a decaying amber field (τ 350 ms, 1.5 m cells), labelled "unresolved hypotheses"; never a car glyph and never counted as objects |
| Footprint exclusion | A candidate whose centre lies inside a track's footprint (+0.6 m) is recorded as `OVERLAPS_TRACK_FOOTPRINT` and does not become a new object. Neighbours parked side by side or bumper to bumper do not overlap, so they stay separate. Nothing is averaged |
| Conflicting pair | Two tracks occupying one footprint are drawn as outlines inside a dashed envelope labelled "N conflicting hypotheses". A tentative track in conflict cannot be confirmed |
| Stopped objects | Drawn at the median of the last 20 measurements, with no extrapolation |
| Motion | Requires a weighted LSQ fit with \|v\|/se ≥ 3, speed > 2.0 m/s (person 1.0) and displacement > 2.0 m (person 1.5). Radial speed must also exceed 2.0 + 0.15·\|range rate\| (the detector range-bias drift) |
| Heading | Body axis is taken mod 180° from producer world yaw (the opt-in heading adapter). It changes only after an axis change > 20° holds for 3 updates. The travel direction (nose) is drawn only when motion or a consistent producer majority supports it. A direction disagreement swaps the nose without rotating the body |
| Timing | 0 ms input buffer. Missed update after 350 ms, fade from 450 ms, expiry at 750 ms, prediction ≤ 500 ms, 150 ms correction blend. All v2 bounds reused |

The Inspect drawer lists every raw candidate of the latest frame with its score, tier, disposition and display track. Clicking a row shows the unchanged measurement and its provenance. The drawer can also overlay every raw candidate, show track IDs, switch the view to tracked objects only or to strict visible-only, and toggle building withholding. Building line-of-sight shading, frame/pose binding and the clock are reused unchanged from v2.

## Measurements (same clips, 60 Hz, current-forward view, 45 m; GT used only to score)

v2 → v3 (`compare_metrics_v3.cjs`). Building withholding is off in both.

| Measure | Departure | Turn | Stop (ego stopped) | Avenue |
|---|---|---|---|---|
| Glyphs on screen (mean) | 21.3 → 3.9 | 25.4 → 7.0 | 20.0 → 2.8 | 18.5 → 4.8 |
| Glyph appearances/s | 70.8 → 4.5 | 77.1 → 9.6 | 57.1 → 1.9 | 65.5 → 5.0 |
| Display IDs created (v3 tracked) | 961 → 86 (28) | 865 → 176 (74) | 578 → 33 (17) | 1130 → 91 (38) |
| GT visible actors with a glyph ≤ 3 m | 0.738 → 0.747 | 0.680 → 0.688 | 0.766 → 0.840 | 0.816 → 0.798 |
| ...with a tracked (non-tentative) glyph | — → 0.725 | — → 0.657 | — → 0.817 | — → 0.769 |
| Glyphs with no real object ≤ 3 m | 0.75 → 0.36 | 0.69 → 0.45 | 0.78 → 0.16 | 0.71 → 0.28 |
| Glyph identity switch per GT actor (≤ 300 ms) | 0.26 → 0.05 | 0.25 → 0.08 | 0.22 → 0.08 | 0.24 → 0.09 |
| Stationary-actor glyph apparent speed P50 (m/s) | 1.89 → 0.56 | 1.75 → 0.44 | 1.56 → 0.68 | 1.55 → 0.52 |
| Stationary-actor body-axis rotation P95 (°/GT frame) | 25.8 → 21.0 | 24.2 → 13.0 | 28.8 → 16.7 | 25.0 → 18.7 |
| Per-frame step P95 (m) | 0.305 → 0.111 | 0.283 → 0.091 | 0.181 → 0.096 | 0.323 → 0.134 |
| Moving-actor along-track error P50 (m) | −0.14 → 0.13 | −0.43 → −0.32 | 0.27 → 1.16 (n=7) | −0.31 → −0.03 |

- **Delay.** No input is buffered.
  - The first glyph is a tentative outline drawn on the first confident detection.
  - Promotion to "tracked" takes a median of 113–144 ms of arrival time (P95 226–322 ms).
  - Showing tracked objects only would cost 2–3 points of GT coverage.
  - The correction blend P95 is 0.74–0.95 m, with 0 snaps.
- **Rendering.** Standalone Chromium at 1440×1000: 60 fps both ways; draw time P50 1.9 → 1.2 ms, P95 4.1 → 3.4 ms.
- **Live 1× side by side (browser).** Glyphs 25 → 4 (departure), 30 → 8 (turn), 19 → 2 (stop). Appearances 97 → 4, 109 → 14 and 56 → 1.5 per second.

## Remaining limits (upstream; the renderer does not fix accuracy)

- **Far tracks are often unsupported.** At 35–45 m, 51–79% of tracked glyphs have no real object within 3 m. Under 20 m it is 0–25%.
  - Example: at frame 2020, the confident boxes for a car at the junction lie about 9 m from it. v3 shows them as a conflicting pair rather than hiding them.
- **Tracked identities are still short.** Their median life is 0.36–0.98 s, because confident detections of the same object are intermittent.
- **Some real objects are only weakly detected.** Example: a pedestrian at frame 2748 scores below threshold and appears only as haze.
- **Parked objects still drift by about 0.5 m/s median.** This is range-bias drift that the stationary anchor damps but cannot remove.
- **Single most useful next step:** a per-frame NMS / calibrated-confidence stage upstream of the map, plus range-error reduction beyond 30 m. Score each change with the same coverage and unsupported-glyph measures.

## Reproduce

```bash
node spatial_map_coop/tests/js/replay_tracks_v3.js                  # 14 v3 policy checks
python3 -m spatial_map_coop.replay_ui_v1.gt_audit_v3 --out /tmp/gt_audit.json   # offline scoring input only
mkdir -p /tmp/clips && for c in departure turn stop avenue; do curl -s http://127.0.0.1:8765/api/replay/clip/$c > /tmp/clips/$c.json; done
node spatial_map_coop/replay_ui_v1/compare_metrics_v3.cjs /tmp/clips /tmp/gt_audit.json out.json
PLAYWRIGHT_MODULE=… CHROMIUM=… node spatial_map_coop/replay_ui_v1/browser_compare_v3.cjs
```

Existing suites still pass: 23 replay, 14 v2 refinement, 105 presentation and 21 Python checks. No capture, pinned or production source, GT/scorer, metric or alert was changed. Nothing was deleted, committed or pushed.
