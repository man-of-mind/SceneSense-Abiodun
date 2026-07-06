# Autonomous run plan (~2 days) — toward a working cooperative spatial map

## Governing constraint
CARLA is flaky on relaunch on this box (stable only on first launch after a full boot; relaunch → crash;
needs a machine reboot I can't do). So this plan is deliberately **CARLA-independent**: capture a data
trace now (supervised), then develop + self-verify entirely **offline** via record→replay + synthetic
scenarios. This mirrors what `../spatial_map_geometry/README.md` already recommends ("prototype the
geometry offline before wiring into live CARLA").

## What gets set up before you leave (5 min, supervised)
1. Run the live pipeline with **both egos** (Stage 2 recipe in README).
2. In another terminal, capture a trace:
   `python3 record_trace.py --out recordings/two_ego.jsonl --hz 5 --duration-s 600`
   (10 min is plenty; ideally drive past a crosswalk + have the two cars' views overlap.)
3. Confirm `recordings/two_ego.jsonl` has lines and `recordings/two_ego.jsonl.static.json` exists.
   That trace + synthetic scenarios are all I need for 2 days offline.

If no trace is captured, I fall back to **synthetic-only** (hand-built two-view / occlusion scenes) —
less realistic but still exercises all the geometry/reasoning logic.

## Offline backlog (prioritized; each item self-verified via a rendered PNG I inspect + asserts)
**A. Replay + synthetic harness (unlocks everything)**
- `replay_trace.py`: feed recorded snapshots to the renderer/reasoners, emit PNGs. No CARLA.
- `synthetic_scenes.py`: generate two-ego and occlusion scenes as local maps (like
  `spatial_map_geometry/demo_two_view_overlap.py`).

**B. Stage 2 hardening**
- Verify color-by-source rendering on replay + synthetic; produce a PNG gallery.
- Surface the duplication case (both cars detect the same object → two boxes) — this is the "before
  fusion" picture that motivates Stage 3.

**C. Stage 3 groundwork — reuse `spatial_map_geometry` (the real payoff)**
- Build a per-stream **FoV polygon** from each packet's `camera.fov` + pose (already in the data) via
  `geometry.sensor_fov_polygon`.
- Run `geometry.overlap_area` + `occlusion_reasoner.infer_overlap_disagreements` on synthetic + replayed
  data → first-pass "seen by A, inside B's FoV, missing from B" hypotheses. Render overlays.
- Prototype the **ray-vs-occluder disambiguation** (the novelty): cast ego→object rays, test against
  static building footprints (from `static_geometry`) + other cars' boxes as occluders, to separate true
  occlusion from a detector miss. Test on a hand-built "truck hides pedestrian" scene.

**D. Evaluation scaffolding (synthetic GT)**
- Association precision/recall + occlusion-warning precision/recall on synthetic scenes where occlusion
  is known by construction. Tables + a plot (dataviz principles).

## How I'll let you verify it worked (visualization + logging)
- **`autonomous_run/figs/`** — a PNG per milestone (Stage-2 color-by-source, FoV overlap, occlusion
  hypotheses, the truck-pedestrian occlusion case). I render headless and inspect each one myself.
- **`autonomous_run/GALLERY.md`** — indexes every PNG with a one-line caption of what it should show.
- **`autonomous_run/PROGRESS.md`** — timestamped log: what was built, what passed/failed, decisions,
  and the exact next step. **Read this first when you return.**
- Geometry **assertions** (polygon areas, overlap ratios, point-in-polygon) as lightweight tests.

## Troubleshooting tips (for you, on return)
- Start at `PROGRESS.md` → then browse `GALLERY.md` PNGs.
- To re-see any milestone offline: `python3 replay_trace.py --trace recordings/two_ego.jsonl --render figs/`.
- To re-run against fresh CARLA: relaunch the Stage-2 pipeline, re-record, re-run replay.
- If a figure looks wrong (mirrored/rotated), it's almost always a world→screen transform sign — noted
  per-figure in PROGRESS.md.

## Guardrails (what I will NOT do)
- Won't launch or depend on live CARLA (flaky); won't reboot the machine.
- Won't edit shared top-level scripts — only files under `spatial_map_coop/`.
- Won't touch OAI/5G, won't install system packages, cautious with the shared `shr_aisvcs` account.
- All work isolated under `spatial_map_coop/autonomous_run/`; everything reversible.
- Checkpoint often; if blocked on an item, log it and move to the next rather than spin.

## Honest scope
After 2 days you'll have: tested offline modules for Stage 2 + Stage 3 geometry/occlusion, a verified
synthetic occlusion demo, an eval harness, and a gallery + log — all ready to wire onto live CARLA when
you're back. I will **not** have advanced the *live* multi-car integration (needs CARLA up), but the
logic it depends on will be built and proven offline.
