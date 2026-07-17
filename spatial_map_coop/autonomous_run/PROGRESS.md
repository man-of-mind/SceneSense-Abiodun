# Autonomous run — progress log

**Status reconciled 2026-07-16.** The original autonomous scope was Stage 2 + infrastructure
(replay/synthetic harness, color-by-source hardening, gallery). It was later expanded only far enough
to establish a **synthetic Stage-3 FoV baseline**; it did not implement or validate real-data occlusion.
Data policy: capture real traces when the pipeline is up, otherwise use synthetic scenes.
Read this file first, then `GALLERY.md`.

## Current handoff

- Stage-2 replay and source-colored rendering are complete.
- Synthetic FoV-membership occlusion is complete and passes its known scene.
- FoV membership alone over-flags real scenes; ray/occluder or visibility-grid disambiguation is open.
- Real CARLA occlusion ground truth, formal precision/recall, association/fusion, and warnings are open.
- The daemon log below is historical evidence of its capture window, not a currently running monitor.

## Done (offline, verified) — 2026-07-02
- `record_trace.py` — standalone recorder (polls the live API → JSONL; captures static geometry once).
- `synthetic_scenes.py` — CARLA-free two-ego scene matching the live `/latest` schema (incl. a
  same-object-seen-by-both duplication).
- `replay_trace.py` — headless matplotlib renderer mirroring the live canvas Stage-2 logic
  (color-by-SOURCE when >1 stream, else by type). Works from a trace or `--synthetic`.
- **Verified:** rendered `figs/stage2_synthetic_two_ego.png` — two sources colored + legend, duplication
  visible, backdrop/anchors/ego all correct. (Reviewed by eye.)
- `autonomous_capture.sh` — daemon that captures a real trace + renders PNGs every 30 min *when the
  pipeline is up* (deterministic, no LLM).
- **Orientation fix (road-snap):** `replay_trace.nearest_road_heading` / `object_heading` — vehicle
  boxes snap to the nearest road-centreline orientation (fixes the unreliable model-yaw slant); default
  `--heading road`, so daemon renders use it. Verified: `heading_methods={road:5, model:2}` (5 vehicles
  snapped, 2 pedestrians kept). See the model-vs-road figs. NOTE: this is in the OFFLINE renderer only;
  wiring it into the live canvas/server is deferred (needs the pipeline up). Velocity-heading (needs a
  frame-to-frame tracker) is the further refinement, also deferred.

## What the daemon will add over the next ~2 days
- Real two-ego traces in `recordings/auto_*.jsonl` and rendered PNGs in `figs/replay_*.png`, each time
  it finds the live pipeline up. Timestamps below.

## Stage-3 GROUNDWORK (added to run; scoped — synthetic only) — 2026-07-03
- `synthetic_scenes.occlusion_scene()` — pedestrian visible to A, occluded from B by a truck (known GT).
- `stage3_occlusion.py` — bridges scenes → `spatial_map_geometry.LocalSensorMap` (FoV via
  `sensor_fov_polygon`), runs the EXISTING `occlusion_reasoner`, verifies vs GT, renders overlay.
  **Result: precision=recall=1.0** (fig `stage3_occlusion_synthetic.png`). Baseline FoV-membership reasoner
  only. NOVEL ray/visibility-grid disambiguation + real-data occlusion HELD for collaboration (see
  STAGE3_NOTES.md — incl. the DOGMa/ICRA'24 visibility-grid idea, and the server change needed to expose
  per-stream pose+fov for real-data Stage 3).

## Next steps when you're back (not in autonomous scope)
- Eyeball `figs/replay_*.png` from real data; confirm color-by-source + duplication look right, and that
  world orientation isn't mirrored (if it is, one sign flip in `replay_trace._oriented_corners`/ylim).
- Then Stage 3: reuse `../spatial_map_geometry` (FoV overlap → occlusion reasoning → ray-vs-occluder).

## Daemon activity log
[2026-07-02 23:58:15] capture daemon started (interval=1800s, capture=90s, api=http://127.0.0.1:35011)
[2026-07-02 23:58:15] pipeline DOWN — waiting
[2026-07-03 00:28:15] pipeline DOWN — waiting
[2026-07-03 00:58:15] pipeline DOWN — waiting
[2026-07-03 01:28:15] pipeline DOWN — waiting
[2026-07-03 01:58:15] pipeline DOWN — waiting
[2026-07-03 02:28:15] pipeline DOWN — waiting
[2026-07-03 02:58:15] pipeline DOWN — waiting
[2026-07-03 03:28:15] pipeline DOWN — waiting
[2026-07-03 03:58:15] pipeline DOWN — waiting
[2026-07-03 04:28:15] pipeline DOWN — waiting
[2026-07-03 04:58:15] pipeline DOWN — waiting
[2026-07-03 05:28:15] pipeline DOWN — waiting
[2026-07-03 05:58:15] pipeline DOWN — waiting
[2026-07-03 06:28:15] pipeline DOWN — waiting
[2026-07-03 06:58:15] pipeline DOWN — waiting
[2026-07-03 07:28:15] pipeline DOWN — waiting
[2026-07-03 07:58:15] pipeline DOWN — waiting
[2026-07-03 08:28:15] pipeline DOWN — waiting
[2026-07-03 08:58:15] pipeline DOWN — waiting
[2026-07-03 09:28:15] pipeline DOWN — waiting
[2026-07-03 09:58:15] pipeline DOWN — waiting
[2026-07-03 10:28:15] pipeline DOWN — waiting
[2026-07-03 10:58:15] pipeline DOWN — waiting
[2026-07-03 11:28:15] pipeline DOWN — waiting
[2026-07-03 11:58:15] pipeline DOWN — waiting
[2026-07-03 12:28:15] pipeline DOWN — waiting
[2026-07-03 12:58:15] pipeline DOWN — waiting
[2026-07-03 13:28:15] pipeline DOWN — waiting
[2026-07-03 13:58:15] pipeline DOWN — waiting
[2026-07-03 14:28:15] pipeline DOWN — waiting
[2026-07-03 14:58:15] pipeline DOWN — waiting
[2026-07-03 15:28:15] pipeline DOWN — waiting
[2026-07-03 15:58:15] pipeline DOWN — waiting
[2026-07-03 16:28:15] pipeline DOWN — waiting
[2026-07-03 16:58:15] pipeline DOWN — waiting
[2026-07-03 17:28:15] pipeline DOWN — waiting
[2026-07-03 17:58:15] pipeline DOWN — waiting
[2026-07-03 18:28:15] pipeline DOWN — waiting
[2026-07-03 18:58:15] pipeline DOWN — waiting
[2026-07-03 19:28:15] pipeline DOWN — waiting
[2026-07-03 19:58:15] pipeline DOWN — waiting
[2026-07-03 20:28:15] pipeline DOWN — waiting
[2026-07-03 20:58:15] pipeline DOWN — waiting
[2026-07-03 21:28:15] pipeline DOWN — waiting
[2026-07-03 21:58:15] pipeline DOWN — waiting
[2026-07-03 22:28:15] pipeline DOWN — waiting
[2026-07-03 22:58:15] pipeline DOWN — waiting
[2026-07-03 23:28:15] pipeline DOWN — waiting
[2026-07-03 23:58:15] pipeline DOWN — waiting
[2026-07-04 00:28:15] pipeline DOWN — waiting
[2026-07-04 00:58:16] pipeline DOWN — waiting
[2026-07-04 01:28:16] pipeline DOWN — waiting
[2026-07-04 01:58:16] pipeline DOWN — waiting
[2026-07-04 02:28:16] pipeline DOWN — waiting
[2026-07-04 02:58:16] pipeline DOWN — waiting
[2026-07-04 03:28:16] pipeline DOWN — waiting
[2026-07-04 03:58:16] pipeline DOWN — waiting
[2026-07-04 04:28:16] pipeline DOWN — waiting
[2026-07-04 04:58:16] pipeline DOWN — waiting
[2026-07-04 05:28:16] pipeline DOWN — waiting
[2026-07-04 05:58:16] pipeline DOWN — waiting
[2026-07-04 06:28:16] pipeline DOWN — waiting
[2026-07-04 06:58:16] pipeline DOWN — waiting
[2026-07-04 07:28:16] pipeline DOWN — waiting
[2026-07-04 07:58:16] pipeline DOWN — waiting
[2026-07-04 08:28:16] pipeline DOWN — waiting
[2026-07-04 08:58:16] pipeline DOWN — waiting
[2026-07-04 09:28:16] pipeline DOWN — waiting
[2026-07-04 09:58:16] pipeline DOWN — waiting
[2026-07-04 10:28:16] pipeline DOWN — waiting
[2026-07-04 10:58:16] pipeline DOWN — waiting
[2026-07-04 11:28:16] pipeline DOWN — waiting
[2026-07-04 11:58:16] pipeline DOWN — waiting
[2026-07-04 12:28:16] pipeline DOWN — waiting
[2026-07-04 12:58:16] pipeline DOWN — waiting
[2026-07-04 13:28:16] pipeline DOWN — waiting
[2026-07-04 13:58:16] pipeline DOWN — waiting
[2026-07-04 14:28:16] pipeline DOWN — waiting
[2026-07-04 14:58:16] pipeline DOWN — waiting
[2026-07-04 15:28:16] pipeline DOWN — waiting
[2026-07-04 15:58:16] pipeline DOWN — waiting
[2026-07-04 16:28:16] pipeline DOWN — waiting
[2026-07-04 16:58:16] pipeline DOWN — waiting
[2026-07-04 17:28:16] pipeline DOWN — waiting
[2026-07-04 17:58:16] pipeline DOWN — waiting
[2026-07-04 18:28:16] pipeline DOWN — waiting
[2026-07-04 18:58:16] pipeline DOWN — waiting
[2026-07-04 19:28:16] pipeline DOWN — waiting
[2026-07-04 19:58:16] pipeline DOWN — waiting
[2026-07-04 20:28:16] pipeline DOWN — waiting
[2026-07-04 20:58:17] pipeline DOWN — waiting
[2026-07-04 21:28:17] pipeline DOWN — waiting
[2026-07-04 21:58:17] pipeline DOWN — waiting
[2026-07-04 22:28:17] pipeline DOWN — waiting
[2026-07-04 22:58:17] pipeline DOWN — waiting
[2026-07-04 23:28:17] pipeline DOWN — waiting
[2026-07-04 23:58:17] pipeline DOWN — waiting
[2026-07-05 00:28:17] pipeline DOWN — waiting
[2026-07-05 00:58:17] pipeline DOWN — waiting
[2026-07-05 01:28:17] pipeline DOWN — waiting
[2026-07-05 01:58:17] pipeline DOWN — waiting
[2026-07-05 02:28:17] pipeline DOWN — waiting
[2026-07-05 02:58:17] pipeline DOWN — waiting
[2026-07-05 03:28:17] pipeline DOWN — waiting
[2026-07-05 03:58:17] pipeline DOWN — waiting
[2026-07-05 04:28:17] pipeline DOWN — waiting
[2026-07-05 04:58:17] pipeline DOWN — waiting
[2026-07-05 05:28:17] pipeline DOWN — waiting
[2026-07-05 05:58:17] pipeline DOWN — waiting
[2026-07-05 06:28:17] pipeline DOWN — waiting
[2026-07-05 06:58:17] pipeline DOWN — waiting
[2026-07-05 07:28:17] pipeline DOWN — waiting
[2026-07-05 07:58:17] pipeline DOWN — waiting
[2026-07-05 08:28:17] pipeline DOWN — waiting
[2026-07-05 08:58:17] pipeline DOWN — waiting
[2026-07-05 09:28:17] pipeline DOWN — waiting
[2026-07-05 09:58:17] pipeline DOWN — waiting
[2026-07-05 10:28:17] pipeline DOWN — waiting
[2026-07-05 10:58:17] pipeline DOWN — waiting
[2026-07-05 11:28:17] pipeline DOWN — waiting
[2026-07-05 11:58:17] pipeline DOWN — waiting
[2026-07-05 12:28:17] pipeline DOWN — waiting
[2026-07-05 12:58:17] pipeline DOWN — waiting
[2026-07-05 13:28:17] pipeline DOWN — waiting
[2026-07-05 13:58:17] pipeline DOWN — waiting
[2026-07-05 14:28:17] pipeline DOWN — waiting
[2026-07-05 14:58:17] pipeline DOWN — waiting
[2026-07-05 15:28:17] pipeline DOWN — waiting
[2026-07-05 15:58:17] pipeline DOWN — waiting
[2026-07-05 16:28:17] pipeline DOWN — waiting
[2026-07-05 16:58:17] pipeline DOWN — waiting
[2026-07-05 17:28:17] pipeline DOWN — waiting
[2026-07-05 17:58:17] pipeline DOWN — waiting
