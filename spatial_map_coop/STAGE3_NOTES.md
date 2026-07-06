# Stage 3 (occlusion deduction) — design notes + scope

## Scope decision (for the overnight run)
- **IN (autonomous, synthetic-GT verifiable):** bridge snapshots → `spatial_map_geometry.LocalSensorMap`
  (FoV polygon from pose + `camera.fov` via `geometry.sensor_fov_polygon`); run the EXISTING
  `occlusion_reasoner.infer_overlap_disagreements` on synthetic occlusion scenes with known ground truth;
  verify it flags the right "seen-by-A / missing-from-B" cases; render overlays; precision/recall.
- **HELD for collaboration (judgment-heavy / needs live pipeline):**
  1. The **novel disambiguation** (below) — deciding true occlusion vs detector-miss vs edge-of-FoV is
     subtle; unattended work risks plausible-but-wrong logic.
  2. **Real-data occlusion** needs a small **server change**: `/api/spatial_map/latest` currently exposes
     only the *followed* ego's pose (`focus_view.ego_pose`), not every stream's pose+FoV. Stage-3 on real
     data needs per-stream `pose` + `fov_deg` + `range_m` in the snapshot (the client already ships them in
     each packet's `camera` block — just surface them per active stream). Plus CARLA occlusion GT for eval.

## The occlusion primitive (baseline, from the scaffold)
`occlusion_reasoner`: object in A's list, inside B's FoV polygon, but with no matching object in B →
*possible* occlusion. Conservative (it lists the ambiguity: occlusion / miss / stale / edge-of-FoV / FP).
Necessary but not sufficient — FoV membership ≠ proof of occlusion.

## The novel upgrade — cooperative visibility (idea from DOGMa, ICRA'24)
Ref: "Dynamic Occupancy Grids for Object Detection: A Radar-Centric Approach" (Ronecker et al., ICRA 2024,
arXiv:2402.01488). A *single-vehicle* radar dynamic occupancy grid — no cooperation, no occlusion — but
its **inverse sensor model + FoV → free / occupied / unknown** per cell is exactly the visibility idea we
want, and it's **radar-native** (plays to our RGB+radar strength).

Plan: each car rasterizes a LOCAL visibility grid — cells seen empty = free, cells with a return =
occupied, cells it couldn't see (beyond range or shadowed behind an obstacle) = **unknown**. Occlusion
then falls out cleanly and more rigorously than FoV membership: if car A reports an object at a cell that
is **unknown/occluded in car B's grid**, B is provably blind there (captures shadowing by obstacles, not
just the FoV cone). Keep the grid **local** (not shared) — still share only the cheap object list + maybe a
compact visibility polygon over V2X, so the bandwidth advantage (measured in the pps study) is preserved.
Extending a local radar DOGMa into a **cooperative, occlusion-deducing** system is genuinely novel (the
paper does neither). This is an alternative/complement to the pure ray-cast frustum in `spatial_map_geometry`.

Bonus: a dynamic (velocity) grid is an object-free way to get motion/heading — relates to the orientation
fix (road-snap now; velocity-heading later).
