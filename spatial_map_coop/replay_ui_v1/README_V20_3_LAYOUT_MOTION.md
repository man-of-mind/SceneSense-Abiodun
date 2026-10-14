# V20.3: map-freshness strip, legend below the map, and a motion-smoothing test (rejected)

Display-only refinement of the accepted [v20.2](README_V20_2_ORIENTATION.md) viewer (replay **and** live). v20.2 is
frozen in `baseline_v20_2/`. The display engine `viewer_assets/replay_tracks_v3.js` is **byte-identical** to v20.2
(sha256 `50a58726…`), so association, identities, positions, motion, orientation, pedestrians, expiry, the map, GT,
the scorer and the runtime are all unchanged.

```bash
OPENBLAS_NUM_THREADS=1 python3 -m spatial_map_coop.replay_ui_v1.recorded_tap_v20 --port 8791
# http://127.0.0.1:8791/                  current display (v20.3)
# http://127.0.0.1:8791/compare-layout    BEFORE v20.2 | AFTER v20.3
# http://127.0.0.1:8791/compare-motion    CURRENT motion (kept) | REJECTED EMA candidate (evaluation only)
```

Evidence is in [`artifacts_v20_3_layout_motion/`](artifacts_v20_3_layout_motion/):
- `v20.3-layout-and-motion-1x.webm` (normal speed, about 36 s): layout before/after on the turn clip, then current
  motion vs the EMA candidate on the turn and pedestrian clips;
- screenshots `01`–`05`;
- `BROWSER_CHECK.json`;
- `MOTION_SMOOTHING_EVAL.json`;
- `FIXTURES_*.json`;
- `INNOVATION_DIAGNOSIS.txt`.

## 1. Layout (shipped)

- **"How to read" is below the map**, as a three-column card under the transport bar. The map stage is pixel-identical
  to v20.2 (1084×844 at 1440×1000; checked in the browser). Ego-follow, the clean default and Inspect are unchanged.
- **The MAP FRESHNESS strip** sits in the IN VIEW NOW card, directly under the object counts
  (`viewer_assets/replay_qos_v1.js`, shared by replay and live). It shows only values that can be derived from stamps
  the feed already carries:

| Shown | Definition | Replay source | Live source |
|---|---|---|---|
| Capture → install | install − capture, latest frame | recorded install stamp | event `timeline.install_ms` |
| Displayed data age | display time − capture of the latest displayed frame (frame/map age) | recorded | host clock mapped to the browser |
| Update rate | map updates/s, last 5 s (needs at least 2) | recorded installs | browser receipts |
| Since last update | display time − when the latest update reached the view | recorded install | browser receipt |
| Object observation age | median and oldest age of each **shown object's own** latest observation | engine rows | engine rows |
| Prediction / UE feedback | **not available · not in this feed** | — | — |

- Anything missing renders as **"not available"**. It is never shown as 0 and never estimated. Examples:
  - before the first update;
  - a live event without `timeline`;
  - capture and install stamps not on the same host wall clock.
- The object age is kept separate from the frame age because a coasting object is older than the latest frame.
- No prediction-ready or UE-feedback stamp reaches the viewer with a frame/source join. The edge timing inside the
  map record is not forwarded, and no telemetry path was added. That line therefore always reads "not available".
- The live page keeps its existing feed card (capture→install / install→viewer / viewer→browser) and status strip.

## 2. Motion smoothing: tested, rejected; the current filter is kept

**Inspection first.** `analysis/innovation_diagnosis.cjs` logged every per-update correction that the current engine
blends over its 150-ms bridge (the median anchor for stationary objects, the LSQ fit endpoint for moving ones, the
pedestrian radial/lateral fit), across 8 retained clips:
- the remaining jitter is mostly **radial** (range error);
- consecutive corrections are **positively correlated** (lag-1 +0.3 to +0.6 in nearly every class and state), so they
  behave like persistent drift, not white noise;
- an EMA can only defer persistent drift; it cannot cancel it;
- moving vehicles carry the largest corrections (P50 about 0.35 m, P95 1–3.5 m per update); stationary vehicles are
  already nearly still (P50 0.01–0.07 m).

**One candidate** (`motion_ema_candidate_v1.js`, evaluation only, parameters fixed before evaluation, no search):
- Per display track, in world XY, the bridge applies only a fraction α = 1 − exp(−Δt_capture/250 ms) of each
  correction. The remainder is held as an offset that rides with the fit (a causal EMA with constant-velocity
  prediction).
- It resets to a full correction on a motion-state change or a correction above 1.5 m (vehicle) / 1.0 m (pedestrian).
  Snaps are the base engine's own decision.
- Identity, source and session changes reset it by construction.
- It never averages across tracks, hypotheses or UEs, and association, fits, tiers and expiry never read it.

**Result.** The table reads current → candidate, scored against GT interpolated to the display instant (GT only scores).
Clips: V10 departure/turn/stop/ped_pause/ped_standing and V20 parked/turn/pedestrians.

| Measure | Better in | Worse in | Examples |
|---|---|---|---|
| Typical calmness: drawn-path 2nd difference P95 | 8/8 veh, 7/8 ped | — | veh 0.032→0.022 m |
| Moving per-tick jitter P95 | 6/8 veh, 7/8 ped | 2/8, 1/8 | V10 turn veh 0.175→0.244 |
| **Jitter tails**: per-tick P99 | 2/8 veh, 1/8 ped | 6/8, 7/8 | ped 0.113→0.166 |
| **Pedestrian jump** (step P99) | 0/8 | **8/8** | 0.119→0.154 m |
| **Added lag**, moving pedestrians (P50) | 0/8 | **8/8** | +8 to +48 ms behind |
| Added lag, moving vehicles (P50) | 2/8 | 6/8 | V10 turn −76→−131 ms |
| Stopped-vehicle error P95 | 2/8 | 6/8 | 0.77→1.05 m |
| Turn error P50 | — | 3/3 | 0.79→0.90, 2.28→2.52 m |
| Stop overshoot (actor 137; same event in two overlapping clips) | — | yes | 1.79→2.00 m |
| Pedestrian visibility (GT ped with a glyph ≤3 m) | 1/8 | 5/8 | up to −1.1 pp |

- Pedestrian visibility changes only because building/FOV/radius checks use the drawn position. The rules themselves
  are untouched.
- Synthetic pedestrian fixtures (`FIXTURES_*.json`, 30 noisy seeds each):
  - a standing pedestrian looks calmer (apparent speed P50 0.39→0.23 m/s, P95 2.55→1.79);
  - the drawn position follows real walking later in **every** moving fixture (+50 to +480 ms to reach 50% of the true
    displacement, e.g. radial 1.0 m/s 717→1050 ms);
  - moving error P50 rises by 0.02–0.09 m;
  - visibility, motion state, IDs and resume timing are unchanged.

**Verdict: REJECTED.** The calmer typical frame is bought by deferring real motion: more lag, larger turn and stop
errors, heavier jump tails, and slightly lower pedestrian visibility. The current filter stays. The candidate is not
loaded by any default page.

## Checks

- `BROWSER_CHECK.json`: PASS, 0 page errors. It verifies:
  - the stage and canvas match v20.2 exactly;
  - the legend is below the map in replay and live;
  - the QoS strip is under the counts and inside the first viewport;
  - before any data, all QoS values are "not available";
  - while playing, the shown values equal the engine-derived numbers;
  - the default view draws 0 hypothesis labels, and Inspect restores them;
  - drawn objects are **identical to v20.2** at 147 paused times (1,047 rows, 468 pedestrians);
  - the live renderer, fed 40 recorded frames locally, shows capture→install with a timeline stamp and "not available"
    without one.
- `node tests/js/replay_qos_v1.js`: 7 checks. All existing suites pass unchanged (105 presentation, 23 replay, 14 v2,
  14 v3, 9 pedestrian, 15 heading, 11 v20.2 display), as do the 18 Python live-feed/replay tests.

## Reproduce

```bash
# Clips: TapRecording/Recording .clip(); GT: gt_audit_v3 format (scoring only)
node spatial_map_coop/replay_ui_v1/motion_smoothing_eval_v1.cjs out.json <dir>:<clip> ...
TRACKS_MODULE=$PWD/spatial_map_coop/replay_ui_v1/motion_ema_candidate_v1.js node spatial_map_coop/replay_ui_v1/ped_motion_fixtures_v3.cjs '' out.json
PLAYWRIGHT_MODULE=/home/shr_aisvcs/.cache/ms-playwright-go/1.50.1/package PLAYWRIGHT_BROWSERS_PATH=/tmp/scenesense-replay-browser \
  REPLAY_URL=http://127.0.0.1:8791 node spatial_map_coop/replay_ui_v1/browser_layout_motion_v20_3.cjs [--checks-only]
```

## Limits

- The recorded clips contain few clean start/stop events (6 stops, 1 start). Start response therefore rests mainly on
  the synthetic fixtures.
- The QoS strip describes display freshness only. It is not the 170-ms scientific deadline and not over-the-air
  latency.
- Prediction/feedback timing would need a verified per-frame join in the feed. That is out of scope here.
