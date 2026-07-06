# Autonomous-run gallery

PNGs live in `figs/`. Open them to visually confirm the map logic.

## Built + verified offline (no CARLA needed)
- **`figs/stage2_synthetic_two_ego.png`** — Stage 2 color-by-source on a synthetic two-ego scene.
  What to look for: cyan = `fusion_ego`, orange = `fusion_ego_b`, legend top-right; **two boxes
  (one cyan, one orange) sitting on the same vehicle near (18, 1)** = the same object detected by both
  cars, *unfused* — the "before fusion" duplication Stage 3 will resolve. Roads (gray cross), building
  (dark rect), traffic-light anchors (gray triangles), ego marker (yellow).

## Orientation fix (road-snap) — verified offline
- **`figs/stage2_synthetic_two_ego_model.png`** vs **`figs/stage2_synthetic_two_ego_road.png`** —
  same scene with deliberately-bad model yaws (38°/-47°/55°…). `model` shows the slant;
  `road` snaps each **vehicle** box to the nearest road orientation → boxes align to the roads
  (pedestrians keep model yaw; box orientation is irrelevant at their size). `road` is the default,
  so daemon-rendered real traces use it. Live canvas/server integration is deferred (needs the pipeline).

## Stage-3 groundwork (occlusion) — verified offline on synthetic GT
- **`figs/stage3_occlusion_synthetic.png`** — two egos (cyan A, orange B) with FoV cones; a truck between
  B and a pedestrian. The pedestrian is **red-circled: "occluded from fusion_ego_b (seen by fusion_ego)"**,
  with B's blocked sightline drawn. `stage3_occlusion.py` bridges the scene into the reusable
  `spatial_map_geometry` reasoner and checks it against known GT: **precision = recall = 1.0** (flagged the
  known occlusion, no false positives). Uses only the baseline FoV-membership reasoner — the novel
  ray/visibility-grid disambiguation (STAGE3_NOTES.md) is HELD for collaboration.

## LIVE two-ego (real data, 2026-07-05)
- **`figs/stage2_LIVE_two_ego.png`** (= replay_00010) — REAL run confirming Stage 2 end-to-end: cyan
  `fusion_ego` (4) + orange `fusion_ego_b` (7) colored by source; vehicle boxes road-snapped; pedestrians
  detected (dots); both cars viewing the same area on the real Town10 backdrop. Trace:
  `recordings/two_ego_live.jsonl`. Concurrency: both egos fresh in 9/21 snapshots (~43%) — the two clients
  publish slowly + independently, so pick both-fresh frames for cooperative analysis.

## Captured from live CARLA (filled in by the daemon)
- `figs/replay_XXXXX.png` — real traces rendered offline by `autonomous_capture.sh` whenever the live
  two-ego pipeline was up. Same color-by-source logic on real detections/poses.
  (Empty until the pipeline runs with both egos while the daemon is active.)
