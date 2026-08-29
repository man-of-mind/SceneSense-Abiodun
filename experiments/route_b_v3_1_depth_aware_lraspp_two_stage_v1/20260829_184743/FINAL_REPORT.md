# Final Route B v3.1 task-separated LR-ASPP report

1. **Terminal.** `TWO_STAGE_LRASPP_STAGE2_OBJECT_FAILED_CLOSE_LRASPP`. This final task-separated LR-ASPP hypothesis is closed; no immediate LR-ASPP variation is proposed.

2. **Local master commits; no push.**
```text
b779b1bef485f12e32043d4015db93b7492c23c5 Record two-stage qualification and scientific launch
85228b8698c273e12b5f8ebd39f58b382e8a42d6 Register bounded qualification harness repair
13c206eb83a0af4aeb4f854fa5c6b5bb498f27ec Repair bounded isolation qualification learning rates
d2faa795b0cdb629ada1dfe5b905cf9706866c99 Preregister final two-stage LR-ASPP experiment
d7909496c8709c1788aebe2e6b3fa72ac138116a Pin two-stage LR-ASPP source provenance
a27b19f32e9b5807100d6abdeb7e927e8689d78e Add preregistered task-separated LR-ASPP pipeline
```
The work remained on local `master`; no branch or push was made. The final-report commit necessarily follows this generated document and is identified in the handoff.

3. **Data, cache and official seed.** Train 16,827 frames/10 episodes; validation 3,345 frames/two disjoint episodes; test absent and unopened. Manifest `5d65e6eb14aadea11ca6bab6e82f0c94c31a50746611d167d282d8988a4504c2`. Official MobileNetV3 V2 `5c1a416349c4cf298f2a6a5e2600ed0ee55e604713578f5e74e6bc8bcaef7997`. Train-cache hashes: `{"CACHE_REPORT.json": "6413abfce4f9600b579e9f30c621db16da63c6b8f24dd0196eab6d4ad9d5ddbb", "depth_forward_f16.bin": "ec75d0a776097f6fb8a582e98e1fe907a7d0032267c1fcae27eb5a8937bf00ed", "index.csv": "d978b1c93a2bcd6292d0e320f5ceb4b04a73d7c82c29537038ddab96b57db13b", "radar_consistency_f32.bin": "89f630ddeb32fbf41c83eed42b7ae7dd78ea10c492984fe6ae2764483ebbbdcf", "valid_u8.bin": "5ec480cb3d2eefa9cba3d35368484ae22d09e253dbc337d7ba10230b67304ee8"}`.

4. **Architecture and split contract.** One seven-channel RGB-radar stem, one official-seeded MobileNetV3/LR-ASPP trunk, fused `low/high` interface, identity compression, shared depth-aware neck, segmentation and training-only dense-depth decoders, and two class-private object trunks/field heads. Qualification raw split/monolithic parity: `True`; transported names `low/high`; no RGB/radar/depth side channel enters the tail.

5. **Stage-1 counts.** {"frozen": 315935, "frozen_tensors": 48, "total": 4174643, "trainable": 3858708, "trainable_tensors": 195}. Object-private parameters were frozen and excluded.

6. **Stage-1 epochs 10/20.**

| Epoch | Vehicle IoU | Person box-mask IoU | Foreground mIoU | Depth overall | 20–30 m | 30–40 m | Pass |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 10 | 0.908392 | 0.573299 | 0.740846 | 0.090117 | 0.198723 | 0.292029 | True |
| 20 | 0.911365 | 0.579190 | 0.745277 | 0.082922 | 0.181368 | 0.259758 | True |

7. **Constant-depth baseline and gates.** Train median 3.220703125 m (`log1p` 1.440001731); per-frame→episode→equal-episode baseline {"20_30": 1.7883755037653106, "30_40": 2.1318111606309573, "overall": 0.7163404399368064}. Candidate limits were 90% overall and 95% in each far band.

8. **Stage-1 selection.** `{"created_utc": "2026-08-29T19:36:41.393320+00:00", "evaluated_epochs": [10, 20], "passing_epochs": [10, 20], "schema": "two_stage_lraspp_stage1_selection_v1", "selected_checkpoint": "/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/route_b_v3_1_depth_aware_lraspp_two_stage_v1/20260829_184743/stage1/checkpoints/epoch_010.pt", "selected_checkpoint_sha256": "f35fbb356570747638273f753ecced61de080eaea8a0847f7b320cfff1944bd9", "selected_epoch": 10, "selection_rule": "earliest passing epoch", "stage2_authorized": true, "wall_seconds": 354.53801056277007}`

9. **Stage-2 reset/fresh optimizer.** `{"fresh_optimizer": true, "reset_checks": {"person": {"all_final_weights_exact_zero": true, "dimension_bias": [-0.5108255743980408, -0.5108255743980408, 0.5306282639503479], "heatmap_bias": [-4.599999904632568], "subcell_bias": [0.0, 0.0], "trunk_kaiming_nonzero": true, "yaw_bias": [0.0, 1.0]}, "vehicle": {"all_final_weights_exact_zero": true, "dimension_bias": [1.3862943649291992, 0.5877866148948669, 0.4700036644935608], "heatmap_bias": [-4.599999904632568], "subcell_bias": [0.0, 0.0], "trunk_kaiming_nonzero": true, "yaw_bias": [0.0, 1.0]}}, "reset_seed": 20260832, "segmentation_dense_reset_bit_identical": true, "selected_stage1_epoch": 10, "stage2_epoch000": {"bytes": 16939469, "path": "/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/route_b_v3_1_depth_aware_lraspp_two_stage_v1/20260829_184743/stage2/checkpoints/stage2_epoch_000.pt", "sha256": "9a502aeb5e6c541292ea0f98877f79c3961b55f1ba511dfadc8736dce348efa5", "verified": true}, "verified_selected_sha": true}`

10. **Stage-2 frozen audit.** `All 30 epoch-boundary representation hashes matched 1ba7a43f2182c5934e69fa9bc586b6cf61e8c00517c01a8da43c8c48acbd07a2`

11. **Stage-2 epochs 10/20/30.**

| Epoch | Eligible | Veh P/R/F1 | Person P/R/F1 | Veh/Person XY m | Veh IoU | Person mask IoU | FG mIoU |
|---:|:---:|---|---|---|---:|---:|---:|
| 10 | False | 0.219609/0.777526/0.342484 | 0.348635/0.527634/0.419852 | 0.981646/1.294063 | 0.908392 | 0.573299 | 0.740846 |
| 20 | False | 0.289466/0.780312/0.422281 | 0.345689/0.568440/0.429925 | 0.973437/1.310881 | 0.908392 | 0.573299 | 0.740846 |
| 30 | False | 0.303966/0.782169/0.437796 | 0.354519/0.563275/0.435156 | 0.968500/1.307190 | 0.908392 | 0.573299 | 0.740846 |

The complete fixed-threshold records—including 0.02 recall ceilings, IoU50 diagnostics, dimensions/yaw, duplicate-FP and failure taxonomy, and depth/range/radar/visibility slices—are preserved in `stage2/evaluation/epoch_010.json`, `epoch_020.json`, and `epoch_030.json`.

12. **Selected deltas versus epoch-40 inherited baseline.**
Not applicable.

13. **Nine service gates/material gain.** No checkpoint was eligible, so the official selected-candidate gate objects are null and service readiness is false. For completeness, the final epoch's diagnostic nine-gate vector was: vehicle precision false (0.303966), vehicle recall false (0.782169), person precision false (0.354519), person recall false (0.563275), vehicle XY true (0.968500 m), person XY false (1.307190 m), vehicle IoU true (0.908392), person box-mask IoU true (0.573299), and foreground mIoU true (0.740846). Material gain also failed: person F1 failed (0.435156 < 0.577617), person recall failed (0.563275 < 0.568079), person XY passed, vehicle eligibility failed, and Stage-1 gates passed.

14. **Selected Stage-2 checkpoint.** `{"checkpoint": null, "epoch": null, "sha256": null}`

15. **v0.25 sensitivity.** `null` It was run only when a selected eligible Stage-2 checkpoint existed.

16. **Runtime, VRAM, latency and transport.** Stage 1 training: 2304.5 s, peak 8490 MiB; Stage 2 training: 2350.5 s, peak 3180 MiB. Stage-2 epoch 10: inference 266.4 s, peak 156.0 MiB, latency median 2.118 ms, transport 5806080 bytes; epoch 20: inference 252.6 s, peak 156.0 MiB, latency median 2.130 ms, transport 5806080 bytes; epoch 30: inference 243.5 s, peak 156.0 MiB, latency median 2.092 ms, transport 5806080 bytes. Qualification batch 16/accumulation 1 memory: `{"pass": true, "stages": {"stage1": {"accumulation": 1, "finite": true, "limit_mib": 12288.0, "pass": true, "physical_batch": 16, "reserved_mib": 8040.0}, "stage2": {"accumulation": 1, "finite": true, "limit_mib": 12288.0, "pass": true, "physical_batch": 16, "reserved_mib": 2562.0}}}`.

17. **No inference-time depth.** Qualification sentinel passed `{'depth_argument_absent': True, 'depth_open_attempts': 0, 'sentinel_input_exact': True, 'signature': "(self, dataset_root: 'Path', rows: 'Sequence[dict[str, str]]') -> 'None'"}`. Deployable Stage-2 prediction signatures accepted RGB-radar only and recorded zero depth paths/labels.

18. **Prohibited scope.** Test, CARLA, OAI contents, q/AE, live split runtime, and the 288 measurements were not opened, run, altered, or started. The pre-existing `OAI/openairinterface5g` gitlink dirtiness was preserved.

19. **Exact version-controlled changed files.** Generated checkpoints, caches, predictions and per-epoch telemetry are intentionally excluded from Git.
```text
experiments/route_b_v3_1_depth_aware_lraspp_two_stage_v1/20260829_184743/CACHE_REFERENCE.json
experiments/route_b_v3_1_depth_aware_lraspp_two_stage_v1/20260829_184743/COMPLETION_SENTINEL
experiments/route_b_v3_1_depth_aware_lraspp_two_stage_v1/20260829_184743/FINAL_REPORT.md
experiments/route_b_v3_1_depth_aware_lraspp_two_stage_v1/20260829_184743/LAUNCH_MANIFEST.json
experiments/route_b_v3_1_depth_aware_lraspp_two_stage_v1/20260829_184743/NOTIFICATION.json
experiments/route_b_v3_1_depth_aware_lraspp_two_stage_v1/20260829_184743/PIPELINE_COMPLETE.json
experiments/route_b_v3_1_depth_aware_lraspp_two_stage_v1/20260829_184743/QUALIFICATION_REPORT.json
experiments/route_b_v3_1_depth_aware_lraspp_two_stage_v1/20260829_184743/QUALIFIED_RUNTIME.json
experiments/route_b_v3_1_depth_aware_lraspp_two_stage_v1/20260829_184743/REGISTERED_QUALIFICATION_REPAIR.json
experiments/route_b_v3_1_depth_aware_lraspp_two_stage_v1/20260829_184743/REGISTERED_TWO_STAGE_DESIGN.json
experiments/route_b_v3_1_depth_aware_lraspp_two_stage_v1/20260829_184743/REGISTERED_TWO_STAGE_DESIGN.md
experiments/route_b_v3_1_depth_aware_lraspp_two_stage_v1/20260829_184743/RESOLVED_CONFIG.json
pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_depth_aware_lraspp_two_stage_v1/__init__.py
pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_depth_aware_lraspp_two_stage_v1/build_depth_cache.py
pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_depth_aware_lraspp_two_stage_v1/common.py
pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_depth_aware_lraspp_two_stage_v1/config.json
pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_depth_aware_lraspp_two_stage_v1/data.py
pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_depth_aware_lraspp_two_stage_v1/decode.py
pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_depth_aware_lraspp_two_stage_v1/evaluate_stage1.py
pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_depth_aware_lraspp_two_stage_v1/evaluate_stage2.py
pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_depth_aware_lraspp_two_stage_v1/finalize.py
pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_depth_aware_lraspp_two_stage_v1/infer_stage2.py
pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_depth_aware_lraspp_two_stage_v1/losses.py
pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_depth_aware_lraspp_two_stage_v1/model.py
pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_depth_aware_lraspp_two_stage_v1/preregister.py
pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_depth_aware_lraspp_two_stage_v1/qualify.py
pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_depth_aware_lraspp_two_stage_v1/run_pipeline.py
pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_depth_aware_lraspp_two_stage_v1/train_stage.py
pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_depth_aware_lraspp_two_stage_v1/transition_stage2.py
pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_depth_aware_lraspp_two_stage_v1/two_stage.py
```

20. **Notification and sentinel.** `NOTIFICATION.json`, `PIPELINE_COMPLETE.json`, and `COMPLETION_SENTINEL` were emitted with the same sole terminal. No q/AE follow-on was started.
