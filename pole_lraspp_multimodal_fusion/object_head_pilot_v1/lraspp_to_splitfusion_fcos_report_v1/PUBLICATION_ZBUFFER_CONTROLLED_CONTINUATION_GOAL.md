/goal Apply the single registered amendment to the uncommitted z-buffer visibility v2 implementation, then run exactly one new create-only controlled qualification. Stop before traffic smoke or model access.

The previous run did not disprove the z-buffer method. It produced 5,156 finite vehicle actor-support pixels and then stopped because an optional isolated instance-camera diagnostic returned no vehicle component. It never reached pedestrian depth support or the six clear/partial/full cases.

## Frozen inputs

- Current local `master`: `2ce208e`
- Base protocol:
  `pole_lraspp_multimodal_fusion/object_head_pilot_v1/splitfusion_fcos_r50_fpn_p2_p7_service_candidate_v1/PUBLICATION_EVALUATION_PROTOCOL_V2.json`
- Binding amendment:
  `pole_lraspp_multimodal_fusion/object_head_pilot_v1/splitfusion_fcos_r50_fpn_p2_p7_service_candidate_v1/PUBLICATION_EVALUATION_PROTOCOL_V2_AMENDMENT_001.json`
- Amendment SHA-256:
  `20f2fd616ab1e498c1c859dcee5c57b8a233cbd80d25e149f721cc2d3f911228`
- Previous failure evidence:
  `data_collection/experiments/route_b_publication_zbuffer_visibility_v2/qualification_20260901_020000/controlled_qualification/controlled_qualification_failure.json`

Verify all hashes and do not modify the protocol, amendment, perception lock, checkpoint or post-processing.

Work on local `master` only. Preserve dirty `OAI/openairinterface5g`, all prior evidence, and the blocked v1 directories unchanged. Do not push. A local commit is authorized only after all primary controlled gates pass.

## Exactly one code correction

In the untracked v2 implementation:

1. Load and bind amendment 001 in the protocol loader.
2. Keep saving the raw isolated vehicle instance image.
3. Attempt the vehicle instance comparison only as an optional diagnostic.
4. If no unique vehicle component exists, record:
   - `instance_diagnostic_available=false`;
   - `instance_diagnostic_unavailable_reason`;
   - `vehicle_depth_support_vs_instance_iou=null`.
5. Do not raise, fail a gate, change a terminal, or stop because that optional diagnostic is unavailable.

Do not replace it with another proxy or instance-token heuristic. Never use `actor.id` or `clone.id` as a rendered token.

No other scientific or runtime change is authorized:

- keep the registered `A_i`, `V_i` and visibility equations exact;
- keep `tau_empty=tau_match=0.02 m`;
- keep transform and walker-pose tolerances;
- keep the same vehicle, pedestrian and occluder construction;
- keep all remaining clear/partial/full gates;
- do not add retries, repeated replays, audits, new metrics or new diagnostics.

Add or update only the small CPU regression proving that a missing optional instance component is recorded and does not block primary depth qualification. Run the existing six CPU checks plus this regression, compilation and diff checks.

## One new controlled run

Run exactly one new create-only controlled qualification in a fresh output directory. Do not reuse or overwrite `qualification_20260901_020000`.

Primary required results remain:

- positive actor-only depth support for vehicle and pedestrian;
- exact transform and walker-bone reproduction;
- clear visibility at least `0.98` for both classes;
- partial visibility strictly between clear and full for both classes;
- full visibility at most `0.02` for both classes;
- finite bounded metrics;
- one inspected six-case RGB/`A_i`/`V_i`/depth-difference contact sheet;
- no person instance, semantic, box, ellipse, learned-mask or broad depth-interval contribution.

If any primary depth gate fails, stop without changing tolerances or definitions. If they all pass, commit only the v2 source/tests/README on local `master`. Do not commit generated evidence, blocked-v1 source, report drafts or OAI.

Do not run traffic smoke, publication episodes, checkpoint loading, inference, validation, test, training, compression, q/AE work or 288 measurements.

Report the six visibility values, vehicle/person support pixels, transform and walker errors, optional diagnostic status, contact-sheet and evidence hashes, tests, commit, and the exact next traffic-smoke command without running it.

Terminal must be exactly one of:

- `PUBLICATION_ZBUFFER_VISIBILITY_CONTROLLED_QUALIFIED_AWAITING_TRAFFIC_SMOKE`
- `PUBLICATION_ZBUFFER_VISIBILITY_CONTROLLED_BLOCKED`
- `PUBLICATION_ZBUFFER_VISIBILITY_IMPLEMENTATION_FAILED`
