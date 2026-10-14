# v3.1: pedestrian display motion (focused refinement)

## How to view it

Run from the repository root:

```bash
python3 -m spatial_map_coop.replay_ui_v1.server --open
```

- `/` is v3.1.
- `/?ped=v3.0` is the previous pedestrian policy.
- `/compare-ped` shows both side by side.
- New replay clips:
  - `05 · Pedestrians: slow down, pause, resume`: real actor 137, frames 2934–2982.
  - `06 · Pedestrians standing, ego approaching`: actors 136 and 137, frames 3352–3424.

Artifacts are in [`artifacts_v3_ped/`](artifacts_v3_ped/):

- [Normal-speed video](artifacts_v3_ped/pedestrian-v3.0-vs-v3.1-1x.webm) with three segments: pause and resume, standing pedestrians, stopped ego.
- Three screenshots.
- `FIXTURES_{BEFORE_v3.0,AFTER_v3.1}.json` and the matching `.txt` files.
- `PED_REPLAY_COMPARISON.json`.
- `BROWSER_CHECK_PED.json`.
- `DISPLAY_COMPARISON_V3_1.json`: the general v3 comparison, re-run with corrected stationary-vehicle metrics.

## Finding

**v3.0 treated "not enough evidence of movement" as "stationary".** For a non-moving track, the display anchor was reset every update to the median of the last 20 measurements. Entering `moving` then required the newest measurement to be more than 1.5 m from that same anchor, and the speed to exceed 1 m/s.

Because the anchor chases a walker, this gap is only about speed × 1.2 s. The clean control (no noise, no misses) proves the defect:

- Walkers at 0.3, 0.6 and 1.0 m/s were never shown moving.
- They were drawn about 1.47 s behind, with errors of 0.44, 0.88 and 1.46 m.

The noisy stationary pedestrian was meanwhile shown `moving` 19% of the time. Presence was never affected: visibility depends only on detections and expiry.

**Measured pedestrian noise** (confident detections vs archived GT, whole route):

| Error source | Value |
|---|---|
| Range (radial) | sd 0.96 m; correlation 0.42 at 240 ms |
| Lateral | sd 0.23 m (MAD 0.09 m) |
| Missed confident detections | 42% |
| Outliers beyond 1.5 m | 10% |

**Erratum (v3 report).** The archive's `object_speed_mps` is 0 for every pedestrian. GT positions show all 26 replay pedestrians walking at 0.9–1.4 m/s. The v3 note called them "standing", and its stationary-actor metrics included them. Corrected stationary-vehicle drift, v2 → v3, is 1.55 → 0.24, 1.21 → 0.27 and 0.71 → 0.04 m/s (departure, turn, avenue; `DISPLAY_COMPARISON_V3_1.json`).

## Change (pedestrians only; display-only, causal, 0 ms buffering)

`replay_tracks_v3.js` adds `fitPedestrian` and `pedestrianMotion: 'v3.1'`:

- **Fit:** robust 2.5 s least-squares fit in radial and lateral coordinates. Outliers beyond 3σ are dropped before refitting.
- **Uncertainty:** noise uses the measured priors (radial 0.9 m, lateral 0.2 m, 4 pseudo-observations). Radial standard error is inflated for the measured correlation. Ego range-rate drift adds systematic radial uncertainty, replacing the hard range guard for pedestrians.
- **`moving`:** requires displacement significance z ≥ 3 on 2 consecutive updates, with no speed floor. It is displayed with a direction tick and ≤ 500 ms prediction.
- **Leaving `moving`:** when z < 1.5 on 2 updates. A causal change point (the last 0.8 s inconsistent with the long fit for 2 updates) drops older samples, so stops register sooner.
- **`stationary`:** only when the speed upper bound (|v| + 2·se) is below 0.3 m/s on both axes.
- **`unresolved` otherwise:**
  - Position is the window mean plus a shrunk motion term, weighted z²/(z² + 4).
  - There is no forward prediction.
  - It is drawn with a faint ellipse showing the measured range uncertainty along the camera ray.

Unchanged:

- Vehicles: geometry, state and tiers are identical in all 7,026 replay samples. Only the shared display-ID counter shifts vehicle labels, which are visible only with Inspect → "Show track IDs".
- Confidence thresholds, confirmation, grouping, visibility, coasting, expiry and the prediction bound.

## Results

### Fixtures

30 seeds each, using the measured noise and miss model, 120 ms cadence and 186 ms arrival delay. Values are v3.0 → v3.1.

| Fixture | Shown moving after (noisy, P50; runs never moving) | Clean control | Position error P50 | Jump P99 / max |
|---|---|---|---|---|
| Standing (jitter) | false-moving time 19% → 2% | — | — | 0.28/0.60 → 0.08/0.22 m |
| Lateral 0.3 m/s (established) | 1.5 s (11/30 never) → 1.7 s (0) | never → 1.1 s | 0.61 → 0.48 m | 0.28 → 0.10 |
| Lateral 0.6 m/s (new) | 2.5 s → 1.2 s | never → 0.9 s | 0.89 → 0.61 | 0.31 → 0.14 |
| Lateral 1.0 m/s (new) | 2.0 s → 1.0 s | never → 0.7 s | 0.97 → 0.64 | 0.31 → 0.15 |
| Lateral 1.5 m/s (new) | 1.4 s → 0.8 s | 1.9 → 0.7 s | 1.03 → 0.72 | 0.32 → 0.18 |
| Radial 0.3 m/s (established) | 2.7 s (8 never) → `unresolved` (27/30 never moving) | never → `unresolved` (error 0.44 → 0.32) | 0.51 → 0.39 | 0.29 → 0.08 |
| Radial 0.6 m/s (new) | 2.7 s → 3.0 s (17/30 never) | never → 2.4 s | 0.76 → 0.62 | 0.31 → 0.11 |
| Radial 1.0 / 1.5 m/s (new) | 2.3 / 2.0 s → 4.2 (5 never) / 3.5 s | never / 1.9 → 1.9 / 1.6 s | 0.85/0.88 → 0.69/0.75 | ≈0.31 → ≈0.14 |
| Stop–pause–resume, lateral 1.2 | settle 1.1 → 1.3 s; resume 1.3 s (3 never) → 1.0 s (0) | settle 1.15 → 0.92 s | 1.03 → 0.73 | 0.32 → 0.18 |
| Stop–pause–resume, radial 1.2 | settle 0.8 → 0.7 s; resume 1.0 s (7 never) → 2.2 s (18 never) | — | 0.79 → 0.76 | 0.31 → 0.13 |
| Standing, ego approaching at 5 m/s | apparent speed P50/P95 0.31/4.22 → 0.42/2.91 m/s; false-moving 11% → 5% | — | — | 0.30 → 0.09 |

- Visibility is 0.96–0.99 before and after in every fixture.
- Display IDs per run are equal or fewer (2–4 → 2–3).
- The displayed position starts following real motion (moved ≥ 50% of true displacement) at about the same time or sooner, 0.6–2.1 s, in all cases.

### Recorded replay

GT is used only for scoring; moving means GT speed ≥ 0.3 m/s derived from positions. Values are v3.0 → v3.1.

| Clip | Pedestrian visibility | ID switch | Walkers shown as moving | Moving error P50 | Jump P99 / max |
|---|---|---|---|---|---|
| departure | 0.53 → 0.54 | 0.15 → 0.08 | 0.45 → 0.59 | 0.93 → 0.87 m | 0.21/0.55 → 0.16/0.42 |
| turn | 0.53 → 0.54 | 0.13 → 0.15 | 0.11 → 0.27 | 1.18 → 1.12 | 0.19/0.44 → 0.14/0.20 |
| stop (ego stopped) | 0.84 → 0.86 | 0.09 → 0.07 | 0.31 → 0.74 | 0.85 → 0.77 | 0.21/0.36 → 0.12/0.17 |
| avenue | 0.74 → 0.73 | 0.17 → 0.13 | 0.23 → 0.42 | 0.97 → 0.70 | 0.22/0.49 → 0.11/0.18 |
| 05 pause/resume | 0.88 → 0.90 | 0.10 → 0.07 | 0.25 → 0.61 | 0.94 → 0.74 | 0.25/0.38 → 0.11/0.17 |
| 06 standing | 0.69 → 0.69 | 0.06 → 0.06 | 0.09 → 0.38 | 0.80 → 0.56 | 0.32/0.55 → 0.13/0.25 |

**The one real stop** (actor 137) leaves `moving` after 0.49 s, versus 0.99 s for v3.0. On resume, v3.1 shows it moving again after about 0.8 s under the same display ID. v3.0 drew it `stationary` while it walked from 9.3 to 11.6 s, then changed its ID.

**Real standing or slowed pedestrians** are rare: only 5–34 GT frames per clip. False-moving falls in 2 clips (0.80 → 0.40, 0.29 → 0.21) and is unchanged in 1 (0.57). Apparent glyph speed P50 is 0.75–0.79 m/s in v3.1 (v3.0 0.79–1.55), and its P95 is higher in two clips (3.6 vs 1.9 m/s).

**Moving-position error P95** is worse in 3 clips (stop 2.10 → 2.28, avenue 1.93 → 2.62, pause 2.37 → 2.43 m) and better in 3. Remaining large errors come from the 1–2 m range error.

**Draw cost** is unchanged in kind: one extra ellipse per unresolved pedestrian.

## Remaining limits

- **Slow radial walking is not identifiable** with this detector. At 0.3–0.6 m/s toward or away from the camera, the measured 0.9 m correlated range error hides the motion. v3.1 shows these walkers as `unresolved` (ellipse) at a lower-error position, rather than as stationary or as confidently moving. This is honest, but it means the walking tick is often absent for radial walkers, and radial resume after a pause is labelled later (2.2 s, 18/30 never in the fixture).
- **No pedestrian is ever confirmed `stationary` under the measured noise.** Standing pedestrians stay `unresolved` with an ellipse. This is intended.
- **Pedestrian identity still breaks often upstream.** There are 42% missed confident detections. Clip 06 showed about 30 distinct pedestrian display IDs in 6 s, in both versions.
- **The replay has only one clean real stop and one stuttering resume.** Most delay evidence is from the fixtures.
- **This is replay only.** It is not a live qualification and makes no safety or collision-avoidance claim.

## Reproduce

```bash
node spatial_map_coop/tests/js/replay_pedestrian_motion_v3.js        # 9 focused regression checks
node spatial_map_coop/replay_ui_v1/ped_motion_fixtures_v3.cjs v3.0   # or v3.1  [out.json]
node spatial_map_coop/replay_ui_v1/ped_replay_metrics_v3.cjs <clipsDir> <gt_audit.json> out.json
PLAYWRIGHT_MODULE=… CHROMIUM=… node spatial_map_coop/replay_ui_v1/browser_compare_ped_v3.cjs
```

Clips come from `curl /api/replay/clip/<name>`. GT scoring input comes from `python3 -m spatial_map_coop.replay_ui_v1.gt_audit_v3 --out …`, which never feeds the UI.

The existing suites still pass: 14 v3, 14 v2 refinement, 23 replay, 105 presentation and 21 Python checks. Earlier artifacts (`artifacts/`, `artifacts_v2/`, `artifacts_v3/`) are untouched.
