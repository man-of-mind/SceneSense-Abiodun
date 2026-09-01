/goal Repair only the controlled occluder camera-plane bug in the uncommitted z-buffer visibility v2 implementation, then run exactly one new create-only controlled qualification. Stop before traffic smoke or model access.

The renderer z-buffer method has now produced:

- vehicle actor-only support: 5,156 pixels;
- pedestrian actor-only support: 568 pixels;
- vehicle-clear visibility: `1.0`.

The last run failed only because `static.prop.container` was hard-coded at camera depth `4.5 m` even though its rotated camera-forward half-extent was large enough to place bounding-box corners behind the camera. This is a controlled-scene placement bug, not a failure of the registered visibility equation.

## Frozen scientific state

- Current local `master`: `2ce208e`
- Protocol v2 and amendment 001 remain binding and unchanged.
- Keep `A_i`, `V_i`, `tau_empty=tau_match=0.02 m`, actors, poses, clear/partial/full requirements and all other primary gates unchanged.
- Keep the instance diagnostic optional and nonblocking.

Do not modify any protocol/lock file, checkpoint, evaluator definition or post-processing. Preserve dirty OAI, all earlier evidence and blocked-v1 files. Do not push. A local commit is authorized only if the controlled qualification passes.

## One implementation repair

Replace the fixed `OCCLUDER_DEPTH_M=4.5` assumption with deterministic bounding-box-safe placement.

Use this geometry-only rule before capturing partial/full cases:

1. Consider opaque props in fixed order: `static.prop.box03`, `box02`, `box01`, `streetbarrier`, then `container` if available.
2. At the intended occluder rotation, calculate the actual camera-forward minimum, centre and maximum depth of all eight bounding-box corners.
3. Select the first prop and centre depth satisfying all of:
   - every corner is at least `0.50 m` in front of the camera;
   - every corner is at least `0.50 m` closer than the nearest target-support surface;
   - when centred, its projected box fully covers the target support box for the full condition.
4. The preferred centre depth is `4.5 m`; increase it only enough to satisfy the camera-plane margin. If that makes the prop too close to the target or too small in projection, deterministically try the next prop.
5. Partial placement must put one projected occluder edge through the horizontal centre of `A_i` and remain vertically centred. Full placement must centre the prop and cover the complete `A_i` bounding rectangle.
6. Recompute and assert the eight-corner depths and projected coverage after final placement, before ticking sensors.

Prop selection may depend only on these geometry constraints—not visibility results. Do not change the registered actor-depth tolerance or tune occluder placement from measured visibility.

Persist each vehicle/person reference record, including transform and walker-bone errors, before beginning occluder placement so an unrelated later failure cannot erase those primary facts.

Add one small CPU regression for the safe-centre-depth calculation and camera-plane margin. Run the existing focused tests, compilation and diff checks. Do not add further audits, replays, diagnostics, retries or infrastructure.

## One controlled run

Run exactly one fresh create-only qualification. Do not overwrite either previous run.

Required primary outcome remains six cases:

- vehicle clear/partial/full;
- pedestrian clear/partial/full;
- clear visibility at least `0.98`;
- partial strictly between clear and full;
- full at most `0.02`;
- finite positive `A_i` for both classes;
- transform error at most `1e-4` and walker-bone error at most `1e-3`;
- inspected six-case contact sheet showing RGB, `A_i`, `V_i` and depth difference;
- no person instance/semantic/box/ellipse/learned-mask/broad-depth-interval input.

If any primary depth or visibility gate fails, stop without further changes. If all pass, commit only the v2 source/tests/README to local `master`. Do not commit generated evidence, report drafts, blocked-v1 code or OAI.

Do not run traffic smoke, publication episodes, model inference, validation, test, training, compression, q/AE work or 288 measurements.

Terminal must be exactly one of:

- `PUBLICATION_ZBUFFER_VISIBILITY_CONTROLLED_QUALIFIED_AWAITING_TRAFFIC_SMOKE`
- `PUBLICATION_ZBUFFER_VISIBILITY_CONTROLLED_BLOCKED`
- `PUBLICATION_ZBUFFER_VISIBILITY_IMPLEMENTATION_FAILED`
