/goal Implement and run one bounded controlled qualification of the registered renderer z-buffer visibility protocol v2. Stop before traffic collection, model inference or evaluation.

The previous instance-mask protocol correctly stopped because this CARLA 0.10 build does not provide pedestrian instance/silhouette pixels. Do not attempt to recover or infer a CARLA-provided person silhouette. Protocol v2 instead derives actor support from isolated actor-only depth rendering.

## Frozen registration — read first and do not modify

- Registration commit: `51724a5`
- `pole_lraspp_multimodal_fusion/object_head_pilot_v1/splitfusion_fcos_r50_fpn_p2_p7_service_candidate_v1/PERCEPTION_BASELINE_LOCK_V1.json`
- `pole_lraspp_multimodal_fusion/object_head_pilot_v1/splitfusion_fcos_r50_fpn_p2_p7_service_candidate_v1/PUBLICATION_EVALUATION_PROTOCOL_V2.json`
- Blocked v1 evidence:
  `data_collection/experiments/route_b_publication_instance_visibility_v1/qualification_20260901_005600/CONTROLLED_RENDERER_BLOCKED_EVIDENCE.json`

Verify their recorded hashes and fail closed on drift. The v2 protocol SHA-256 is `361bd50f9f94f18689a67fbb2cfb3ed0ac02f668677daf3ab861405210adb728`.

Work on local `master` only. Preserve the dirty `OAI/openairinterface5g` submodule and all blocked-v1 evidence unchanged. Do not push. You are authorized to create one local commit on `master` only if the controlled v2 qualification passes.

## Exact v2 definition

Using losslessly decoded CARLA depth in metres and the fixed registered tolerances `tau_empty = tau_match = 0.02 m`:

```text
A_i(p) = D_actor_i(p) + 0.02 < D_empty(p)
V_i(p) = A_i(p) and abs(D_scene(p) - D_actor_i(p)) <= 0.02
visibility_i = pixels(V_i) / pixels(A_i)
```

- `D_empty`: isolated reference camera with no actor.
- `D_actor_i`: the same camera with only actor `i`, reproducing its blueprint, camera-relative transform and articulation/pose.
- `D_scene`: synchronized normal-scene depth.
- `A_i`: renderer-derived in-frame actor support. It is not a CARLA-provided person silhouette annotation.
- `V_i`: the part of that support whose expected actor surface remains the front-most scene surface.

This is not the old projected-box depth interval. Never accept a pixel simply because its scene depth falls somewhere between actor near/far box limits.

## Focused implementation

Create fresh v2 packages:

- `data_collection/route_b_publication_zbuffer_visibility_v2/`
- `pole_lraspp_multimodal_fusion/object_head_pilot_v1/publication_zbuffer_visibility_evaluation_v2/`

Do not commit the untracked blocked-v1 implementation directories. You may reuse small transform, walker-pose and create-only I/O helpers after reviewing them, but the production v2 path must not depend on pedestrian instance or semantic masks.

Implement only what the controlled proof needs plus small reusable primitives:

1. Exact lossless CARLA depth decoding from raw BGRA:
   `(R + 256*G + 65536*B)/(16777215)*1000 m`.
2. Synchronized RGB and normal-scene depth capture.
3. An isolated reference rig with identical image size, FOV and intrinsics that captures:
   - one empty depth frame;
   - one actor-only depth frame for each target.
4. Exact camera-relative actor transform reproduction.
5. Exact pedestrian walker-bone capture/copy and verification.
6. Computation and lossless persistence of `A_i`, `V_i`, depth-difference maps, counts and visibility.
7. Provenance keyed by actor, class, source frame and reference frame.

The instance camera is optional and diagnostic only for the controlled vehicle. If used, discover its rendered token from the controlled component; never use `actor.id` or `clone.id` as an instance token. The blocked v1 code at `reference_renderer.py:131` and `controlled_scene.py:244` made exactly that invalid assumption and must not be copied.

Add only focused CPU tests for depth decoding, `A_i`, `V_i`, bounded visibility, transform reproduction and protocol/hash loading. Do not add a general framework, repeated deterministic replays, hash-chained amendments, continuous polling, notifier infrastructure, large review packets or exhaustive audits.

## One controlled CARLA qualification

Use exactly one vehicle and one pedestrian. For each class, capture three deterministic conditions using a depth-rendered opaque occluder:

- clear;
- partial occlusion;
- full occlusion.

Required gates:

1. RGB and normal depth have identical frame and timestamp.
2. The empty reference contains no unexpected nearby geometry.
3. Actor-only depth produces a positive `A_i` for both vehicle and pedestrian.
4. Reference camera intrinsics and pixel coordinates equal the normal camera.
5. Actor transform reproduction maximum error is within the existing `1e-4` matrix tolerance.
6. Walker bone-pose reproduction maximum error is at most `1e-3`.
7. Every saved depth and visibility value is finite; every visibility is in `[0,1]`.
8. Clear visibility is at least `0.98` for both classes.
9. Partial visibility is strictly between clear and full for both classes.
10. Full visibility is at most `0.02` for both classes.
11. For the clear vehicle only, the depth-derived `A_i` agrees with the working isolated vehicle instance mask at IoU at least `0.98`. This is a diagnostic cross-check, not the source of `A_i`.
12. No instance, semantic, filled-box, ellipse, learned mask or broad near/far interval contributes to person support.

Generate one six-case contact sheet. Each panel must contain RGB, `A_i`, `V_i`, depth-difference visualization, class, range, visibility, actor/reference pose errors and the fixed `0.02 m` tolerance. Visually inspect that the pedestrian support follows the rendered body rather than its projected box and that partial/full occluders remove the expected pixels.

Run no traffic smoke, publication episode, model checkpoint, inference, validation, test access, optimizer, compression, q/AE work or 288 measurements.

## Stop and report

If all gates pass:

- commit only the new v2 source/tests and a compact README on local `master`;
- preserve all create-only controlled evidence outside the commit;
- report the commit, six visibility values, support pixel counts, pose errors, vehicle cross-check IoU, contact-sheet path, evidence hashes, tests and exact next traffic-smoke command without running it.

If the depth camera does not render the pedestrian, the isolated pedestrian pose cannot be reproduced, or any clear/partial/full gate fails, do not change tolerances or definitions. Emit the blocked terminal and stop.

Terminal must be exactly one of:

- `PUBLICATION_ZBUFFER_VISIBILITY_CONTROLLED_QUALIFIED_AWAITING_TRAFFIC_SMOKE`
- `PUBLICATION_ZBUFFER_VISIBILITY_CONTROLLED_BLOCKED`
- `PUBLICATION_ZBUFFER_VISIBILITY_IMPLEMENTATION_FAILED`
