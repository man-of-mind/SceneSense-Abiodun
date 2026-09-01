/goal Implement and qualify the ground-truth pipeline for the registered Route B publication instance-visibility evaluation. Stop before full data collection or model inference.

This is a focused implementation-and-smoke task. Do not train, tune, evaluate a model, run the four registered publication episodes, access old test data, start compression/q/AE work, or start the 288 measurements.

## Scientific registration — read first and do not modify

- `pole_lraspp_multimodal_fusion/object_head_pilot_v1/splitfusion_fcos_r50_fpn_p2_p7_service_candidate_v1/PERCEPTION_BASELINE_LOCK_V1.json`
- `pole_lraspp_multimodal_fusion/object_head_pilot_v1/splitfusion_fcos_r50_fpn_p2_p7_service_candidate_v1/PUBLICATION_EVALUATION_PROTOCOL_V1.json`

These files freeze the accepted FCOS checkpoint, post-processing, visibility thresholds `0.10/0.25/0.50/0.70/0.85`, four future episode seeds, models, and metrics. Fail closed if either JSON file is invalid or if any bound checkpoint/configuration hash differs.

The lock commit is `56ba02a`. Do not edit the four files introduced by that commit.

Work on local `master` only. Preserve the existing dirty `OAI/openairinterface5g` submodule unchanged and unstaged. Do not push. You are authorized to make one local commit on `master` after the implementation and bounded qualification pass.

## Objective

Build a new, isolated package that measures per-actor visibility from actual CARLA instance pixels:

`visibility = |visible actor mask ∩ unoccluded actor mask| / |in-frame unoccluded actor mask|`

The visible mask must come from the synchronized normal-scene CARLA instance-segmentation camera. The unoccluded mask must be rendered by CARLA for the same actor blueprint, articulation/pose, camera-relative transform, resolution, FOV and intrinsics, with external occluders absent. Self-occlusion remains in both masks.

Do not substitute:

- depth-consistent projected-box occupancy;
- semantic-class pixels without actor identity;
- filled projected 2D/3D boxes;
- ellipses or class-average silhouettes;
- learned amodal completion.

If the installed CARLA shipping build cannot produce an actor-specific unoccluded renderer mask with defensible pose and camera alignment, emit `PUBLICATION_VISIBILITY_GROUND_TRUTH_BLOCKED` and stop. Do not silently fall back to the old approximation.

## Minimal implementation

Create:

- `data_collection/route_b_publication_instance_visibility_v1/` for synchronized instance capture, actor-state capture, actor-ID mapping, isolated unoccluded rendering, mask comparison and the future four-episode runner;
- `pole_lraspp_multimodal_fusion/object_head_pilot_v1/publication_instance_visibility_evaluation_v1/` for loading the registered protocol and future scoring/report generation.

Reuse the existing Route B v3 route, RGB/radar/depth collection, camera calibration, cadence and object metadata. Do not edit or reinterpret the historical evaluator. Store the new evidence under create-only paths in `data_collection/experiments/route_b_publication_instance_visibility_v1/`.

Implementation requirements:

1. Add a normal-scene `sensor.camera.instance_segmentation` camera with exactly the RGB camera transform, resolution, FOV and cadence.
2. Bind every decoded rendered instance ID to the correct CARLA actor ID. Verify the mapping with controlled actors; do not assume the encoded ID equals `actor.id` without proof.
3. Record enough state to reproduce the actor silhouette:
   - actor ID and blueprint;
   - actor and camera transforms;
   - camera intrinsics;
   - walker bone pose for pedestrians;
   - vehicle state that materially changes its silhouette, if present.
4. Produce the unoccluded mask using an isolated CARLA rendering of the same actor at the same camera-relative transform. A practical implementation may place a reference camera/actor rig in an isolated empty region, provided the normal and reference image coordinates and intrinsics are identical and no external geometry contaminates the actor mask.
5. Save lossless visible and unoccluded binary masks, their pixel counts, overlap, visible-outside-reference pixels, visibility value, in-frame/truncation data, class, range and provenance keyed by `sample_id`, `frame_id` and `gt_actor_id`.
6. Reuse the existing non-visibility geometry/range eligibility through 40 m unchanged. Only the visibility signal is replaced.
7. Prepare—but do not execute—the future evaluator for the registered five visibility views and four distance bands. Preserve the historical 3 m world-XY service matching as one result family and add separate IoU-based AP50/AP50–95 and true visible-pixel segmentation metrics as registered.

Keep the implementation small. Add only tests that protect the visibility equation, ID mapping, transform reproduction, frozen protocol loading, and fail-closed behavior. Do not build a generalized framework or add unrelated audits.

Do not add repeated deterministic replays, hash-chained amendments, continuous log polling, notifier/sentinel infrastructure, extensive review packets, or exhaustive tensor audits. One controlled scene and one create-only smoke are enough for this stage.

## Bounded qualification only

Run no model checkpoint and no optimizer.

### A. Controlled CARLA scene

Use one vehicle and one pedestrian, with fixed camera/actor poses, in three deterministic conditions:

- no external occluder;
- partial opaque occluder;
- full opaque occluder.

For the pedestrian, copy the walker bone pose into the isolated render. Verify:

- exact RGB/instance frame synchronization;
- unique actor-to-instance mapping;
- positive unoccluded mask area;
- finite visibility in `[0,1]`;
- no-occluder visibility is at least `0.98` for both classes;
- partial visibility lies strictly between the clear and full-occlusion values;
- full-occlusion visibility is at most `0.02`;
- visible-mask pixels outside the unoccluded reference are reported, not silently discarded.

### B. One short traffic smoke

Collect at most 128 saved frames using the existing 50/50 Route B setup and fresh smoke-only seeds that are not any registered publication or canonical test seeds. Do not run a full 600-second episode.

Require:

- all sensor streams and metadata reconcile exactly by frame/sample;
- every geometry-qualified vehicle/person has an unoccluded render with positive area;
- every visible instance ID maps unambiguously to one actor;
- all ratios are finite and bounded;
- no depth-box or semantic-only fallback was invoked;
- output paths are create-only and hashes/manifests reproduce the saved evidence.

Generate one contact sheet with exactly 24 deterministic examples spanning vehicles/people, near/far, clear/partial/heavy occlusion. Each panel must show RGB, visible actor mask, unoccluded actor mask, overlay, actor ID, range and computed visibility. Inspect the contact sheet and record any mapping, pose, alignment, road/background contamination or clipping defect.

This smoke is a ground-truth qualification, not a model validation result. Do not report FCOS or LR-ASPP accuracy from it.

## Stop boundary and report

After qualification, stop. Do not launch the four registered episodes and do not load any model checkpoint.

Report only:

- implementation files and local commit;
- exact visibility definition;
- how actor-ID mapping and unoccluded rendering were proven;
- controlled-scene results;
- smoke counts and reconciliation;
- contact-sheet path;
- qualification defects, if any;
- estimated wall time and disk cost for the later four-episode collection;
- exact future commands, but do not run them.

Terminal must be exactly one of:

- `PUBLICATION_VISIBILITY_IMPLEMENTATION_QUALIFIED_AWAITING_COLLECTION_AUTHORIZATION`
- `PUBLICATION_VISIBILITY_GROUND_TRUTH_BLOCKED`
- `PUBLICATION_VISIBILITY_IMPLEMENTATION_FAILED`

Do not turn a blocked renderer/ID-mapping issue into another metric definition. The purpose of this stage is to catch that issue before expensive collection and inference.
