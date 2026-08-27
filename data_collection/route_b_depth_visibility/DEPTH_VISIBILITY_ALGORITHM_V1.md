# Depth visibility contract — frozen algorithm v1 (registered BEFORE the smoke)

Written 2026-08-27, before any CARLA process was launched for this study.
Scope: decide whether a synchronized colocated depth camera can separate
visible / partially visible / fully occluded pedestrian actor boxes well enough
to become a future Route B GT eligibility contract. Nothing here is integrated
into the canonical v2 collector by this study.

## 1. Geometry reused verbatim from the canonical Route B v2 path
Source of every constant: `data_collection/run_route_b_perception_collection_v2.py`
(`PilotEpisode.args`, lines 372-437) and
`carla_split_inference_udp_fusion_object_pole_client_spatial_stream_oai.py`
(`fusion_runtime`).

| quantity | value | source |
|---|---|---|
| RGB / depth resolution | 1280 x 720 | `camera_width`, `camera_height` |
| horizontal FOV | 120.0 deg | `camera_fov` |
| model input resolution | 768 x 432 | `model_input_width/height` |
| ego camera transform | x=1.8, y=0.0, z=1.55, pitch=-4.0, yaw=0.0, roll=0.0 | `DEFAULT_EGO_CAMERA_*` |
| intrinsics | `fusion_runtime.intrinsics_at(w, h, fov)` | pinhole, f=(w/2)/tan(fov/2) |
| world tick | 20 Hz synchronous, `sensor_tick=0.0` (free-running) | v2 cadence contract |

Camera convention (identical to `carla_collect_parked_ego_fusion_training_data.py`):
camera-local `x` is forward/depth, `u = cx + (y/x)*fx`, `v = cy - (z/x)*fy`,
in-front test `x > 0.05`.

## 2. Depth decode (repository convention)
`cooperative_fusion/engine_gt_prototype.py:decode_depth_m`:
`norm = (R + G*256 + B*256^2) / (256^3 - 1)`, `depth_m = 1000.0 * norm`.
BGRA raw buffer, so `B = a[:,:,0]`, `G = a[:,:,1]`, `R = a[:,:,2]`.
This is CARLA planar depth along the camera forward axis — the same axis as the
camera-local `x` used for the actor box, so the two are directly comparable.

## 3. Per-actor per-frame computation
1. `center_world, corners_world = actor_bbox_world_points(actor)` — the exact
   canonical 8-corner helper (`bbox.location` offset + `bbox.extent` corners,
   pushed through the actor world matrix).
2. `corners_cam = camera_inverse_matrix @ corners_world`; `depth = corners_cam[:,0]`;
   keep `depth > 0.05`.
3. Projected axis-aligned box = min/max of `(u, v)` over in-front corners,
   clipped to the frame. `projected_area_px = w * h` (float, full resolution) —
   byte-identical to `project_world_points_to_bbox`.
4. Actor depth interval `[near, far] = [min(depth_infront), max(depth_infront)]`.
5. Fixed tolerance **`DEPTH_TOLERANCE_M = 0.25`**, applied symmetrically:
   a pixel inside the projected box is **depth-consistent** when
   `near - 0.25 <= d <= far + 0.25`.
   Rationale: the actor AABB already spans the body's own depth extent, so the
   tolerance only absorbs box-vs-mesh slack and pose jitter. It is a single
   documented constant and is not tuned after results.
6. Per-actor outputs:
   - `projected_area_px` (full res, canonical formula)
   - `roi_px` (integer pixel count actually sampled)
   - `depth_consistent_px` (full res)
   - `visible_fraction = depth_consistent_px / roi_px`
   - `model_input_visible_px` — the full-res consistency mask nearest-neighbour
     resampled to 768 x 432 and counted (not an area-scaling estimate)
   - `occluder_closer_fraction` — `d < near - tol`
   - `background_farther_fraction` — `d > far + tol`

## 4. Registered provisional future-GT eligibility rule (frozen)
An actor is eligible only if **all** hold:
- `gt_distance_m <= 40.0`  (existing Route B evaluation filter)
- `projected_area_px >= 12.0`  (existing Route B evaluation filter)
- `model_input_visible_px >= 12`
- `visible_fraction >= 0.10`

Sensitivity at `visible_fraction >= 0.05` and `>= 0.20` is reported for
information only. The 0.10 threshold is **not** replaced after seeing results.

## 5. What is deliberately NOT used
Semantic / instance camera tags are not used to validate pedestrians: walker
pixels are known to be absent or wrong in this CARLA 0.10 build. Only the depth
camera and the actor bounding boxes participate.

## 6. Prior-art check: `rasterize_person_regions_depth`
`pole_lraspp_multimodal_fusion/pole_lraspp_multimodal_fusion/common.py:320`
carves a person mask with depth, but it is **not usable as a visibility
contract** as written:
- it tests against a single scalar actor distance with asymmetric pads
  (`-1.5 m / +1.0 m`), not the actor's own near/far interval;
- it has a 12%-of-region fallback (line 350) that **restores the full ellipse**
  when depth carving removes almost everything — i.e. a fully occluded
  pedestrian is silently repainted as fully visible. That is exactly the
  failure mode this study exists to remove.
This smoke therefore implements the interval test above rather than reusing it.

## 7. Runtime gates (all must pass)
RGB/depth identical CARLA frame id; timestamp delta <= 1e-4 s (the established
v2 synchronous tolerance); no missing / duplicate / out-of-order sensor frames;
both images non-empty; depth finite and within `(0, 1000] m`; clearly visible
actors materially above fully occluded actors; fully occluded actors fail the
registered rule; clearly visible actors pass it; actor + sensor cleanup;
clean CARLA shutdown.
