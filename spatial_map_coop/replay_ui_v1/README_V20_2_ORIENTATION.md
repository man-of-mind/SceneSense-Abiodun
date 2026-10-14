# V20.2: vehicle display orientation and a clean default view

Display-only refinement of the V20 offline/live viewer. It is built on [v20.1](README_V20_HEADING.md), which is frozen in
`baseline_v20_1/` and keeps the v3.1 layout and pedestrian policy. V20 remains **STOP**. Nothing here changes
association, track identities, positions, motion, pedestrians, the map, GT, the scorer or the runtime.

```bash
OPENBLAS_NUM_THREADS=1 python3 -m spatial_map_coop.replay_ui_v1.recorded_tap_v20 --port 8777
# http://127.0.0.1:8777/compare-orientation   BEFORE v20.1 (what was reviewed) | AFTER v20.2
# http://127.0.0.1:8777/compare-heading       BEFORE preserved v3.1 (historical) | AFTER current
```

Main evidence: the [normal-speed v20.1 | v20.2 clip](artifacts_v20_2_orientation/v20.1-vs-v20.2-1x.webm) (≈45 s). It has four
segments in this order: parked headings, parked avenue, turn, pedestrians. There are also four same-time screenshots
(`*-v20.1-vs-v20.2.png`), plus `default-view.png`, `inspect-view.png` and `live-renderer-offline-fixture.png`.

## Why vehicles still looked diagonal after v20.1

v20.1 reduced jitter, not error. Archived V20 GT scored the display (GT is used for diagnosis only; it never reaches
the engine; [details](artifacts_v20_2_orientation/DISPLAY_ORIENTATION_VS_GT.json)). For established stationary vehicles
within 3 m of a GT vehicle, v20.1 drew the axis **more than 20° off on 39% of samples** (1,663 / 4,266). The causes of
those samples:

| Cause | Share |
|---|---:|
| The **first** producer yaw was accepted. The 4-consecutive-update rule then rarely let it change, although the recent raw mean was right ("held stale/initial axis") | 31% |
| **Persistent producer bias**: the raw axis is consistent but wrong, often near-perpendicular | 30% |
| **Producer jitter**: no consistent raw axis | 27% |
| GT vehicle actually moving/turning while the display classifies it stationary | 11% |
| Legitimate angle (GT off the lane tangent) | 0.5% |
| Identity change (track matched to more than one GT object) | 8% (overlaps the rows above) |

The producer yaw itself is poor ([summary](artifacts_v20_2_orientation/PRODUCER_YAW_VS_GT.json)). For 4,529 strong
qualified vehicle observations, median axis error is 22° and 53% are more than 20° off. By object type:

- parked environment cars: median 9.6°;
- stopped actors: 28°;
- moving actors: 33°.

26 of 69 stationary objects have a persistent mean bias above 20°, and 11 have one above 60°. **This bias is a detector
property and cannot be fixed from the available evidence.** The display can only avoid drawing it.

Angled parking essentially does not occur next to lanes on this route. All 136 parked GT vehicles within 4 m of a lane
polyline are within 20° of the lane. Beyond 7 m, in lots and off-road, 16 of 32 are angled.

## Change (vehicles only, glyph only)

`replay_tracks_v3.js`, policy `vehicleDisplay:'v20.2'` (`'v20.1'` restores the previous glyph). The association axis is
unchanged. A separate *display* axis is computed for each vehicle:

1. **Moving** (the existing qualified motion mode): the motion axis, exactly as before. Turns are followed, not frozen.
2. **Otherwise, evidence** is the score-weighted axis mean of *every* strong qualified producer yaw since the track last
   moved. When a track stops, this is seeded with its last trajectory axis (worth 4 observations). This replaces
   "first observation, then 4 consecutive agreeing updates".
3. **Weak lane prior.** It applies only within 4.5 m of a lane polyline, when every segment within 4.5 m agrees within
   15° (so not at a junction and not on a tight curve). It also requires at least 3 observations, an observed axis
   within 25° of the lane (released at 35°), and resultant ≥ .5 (released at .35). If all hold, the lane tangent is
   drawn, labelled *lane-tangent display prior* in Inspect.
4. **Observed axis too far from the lane** (for example perpendicular bias, or angled parking beside a lane): the
   **orientation-neutral** glyph is drawn. The vehicle is never forced straight.
5. **No usable lane** (off-road or a junction): the observed axis is drawn only if it is consistent (at least 4
   observations, resultant ≥ .85, released at .75). Otherwise the neutral glyph is drawn.

Positions are never moved. Lane polylines come from the validated static geometry and are switched off if it is not
validated.

The default presentation (replay **and** live) has these changes:

- no low-confidence haze or its labels;
- no "N conflicting hypotheses" canvas labels (v20.1 still drew 136–243 per clip segment by default);
- no amber low-confidence count;
- no raw-candidate line;
- no hypothesis legend entry.

Conflicting tracks are drawn as **one** orientation-neutral dashed outline covering all members. No member is chosen as
real. **Inspect** restores everything: members, the labelled envelope, the haze toggle, counts and the raw line. Inspect
detail also shows `display_orientation` (mode, source, observed axis, resultant, lane distance/junction) next to the
association axis. Pedestrian drawing and policy are untouched.

## Results (same recorded inputs, full route)

GT is used for scoring only.

| Measure | v20.1 | v20.2 |
|---|---:|---:|
| Established stationary vehicles drawn >20° from GT | 39.0% | **0.9%** (and 25% drawn neutral) |
| …of which persistent producer bias | 501 samples | 12 |
| Established moving vehicles drawn >20° from GT | 19.8% | 10.4% (6.6% neutral) |
| Tentative / conflicting tracks drawn >20° from GT | 68% / 69% | 0% / 9% (mostly neutral) |
| Displayed-axis error P50 / P90, all matched vehicles | 9.5° / 52.2° | 1.0° / 7.7° |
| Stationary drawn-axis change, °/s | 3.25 | 0.31 |
| Established tracks with no GT vehicle within 3 m: drawn neutral | 0% | 35% |
| 90° turn T325: drawn axis vs v20.1, P95 / time neutral while moving | — | 0.45° / 0 |

Equivalence ([EQUIVALENCE_60HZ.json](artifacts_v20_2_orientation/EQUIVALENCE_60HZ.json)): 33,028 samples at 60 Hz,
209,475 view rows, 65,929 of them pedestrians. **Zero differences** in identity, tier, pose, motion, association axis,
conflicts, visibility, alpha, counts and field. Only the vehicle `displayYaw` differs. The browser pedestrian check
compared 98 paused same-time states (302 pedestrian rows, 192 unresolved): identical.

Browser ([BROWSER_CHECK.json](artifacts_v20_2_orientation/BROWSER_CHECK.json), PASS, 0 page errors):

- default replay draws 0 hypothesis/conflict labels and Inspect restores them;
- presence is unchanged by Inspect;
- the actual live renderer, fed 40 recorded frames locally, has the same default/Inspect behaviour.

## Limits

- **Persistent detector heading bias remains unresolved.** v20.2 hides it behind a neutral glyph; it does not correct it.
  About a quarter of stationary vehicles therefore show no orientation. A real angled car next to a lane also shows
  neutral (none exists on this route).
- Remaining moving errors come from the existing motion fit: noise, and range-drift false motion of stationary GT objects.
  The motion fit was left unchanged.
- A short-lived neutral→oriented step occurs once per track when evidence arrives. Over the route, oriented→neutral
  losses happen about 35 times.
- GT parity covers one route and one session. The lane-agreement thresholds were chosen on this route; they are not
  qualified elsewhere.

## Reproduce

```bash
OPENBLAS_NUM_THREADS=1 python3 -m spatial_map_coop.replay_ui_v1.recorded_tap_v20 --export /tmp/v20in    # create-only
node spatial_map_coop/tests/js/replay_vehicle_display_v20_2.js      # 11 focused checks
# offline GT diagnosis/equivalence scripts: artifacts_v20_2_orientation/analysis/ (GT only scores)
PLAYWRIGHT_MODULE=/home/shr_aisvcs/.cache/ms-playwright-go/1.50.1/package PLAYWRIGHT_BROWSERS_PATH=/tmp/scenesense-replay-browser \
  node spatial_map_coop/replay_ui_v1/browser_orientation_v20_2.cjs [--checks-only]   # needs the :8777 server
```
