# Route B v3.1 native-grid expanded continuation v3 report

Terminal: `LRASPP_EXPANDED_LONGTRAIN_IMPROVED_NOT_SERVICE_READY`

Experiment: `/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/route_b_v3_1_native_grid_expanded_continuation_v3/20260828_045200`

Execution commit on local `master`: `02ebabd2b6b7dd3e1ecb4b199541de2a458fda2e` (the required starting HEAD). The final source/config/report commit is the post-run local `master` HEAD reported at handoff; nothing was pushed.

## Exact continuation state

- Resume checkpoint: `experiments/route_b_v3_1_native_grid_expanded_training_v2/20260828_103000/checkpoints/route_b_v3_1_native_grid_expanded_training_v2/epoch_010.pt`.
- Verified SHA-256: `26763b2955258ba7bc0287a702788d28f03d82fd504921030af53951b5b39e11`.
- State: epoch `10`, optimizer steps `10520`, 1,052 steps/epoch; model, AdamW optimizer, registered H2/J2 scheduler, AMP GradScaler, and Python/NumPy/Torch CPU/CUDA RNG state were present and strictly loadable.
- End-of-epoch-10 inherited/object LR: `{'inherited': 4.908911333958271e-06, 'object': 4.908911333958272e-05}`; these reconcile to the registered cosine schedule.
- Epochs 1–10 were not repeated and the epoch-15 warm start was not used to restart this continuation.
- All recorded view/manifest/GT/ignore/camera-plane hashes passed, including `80688` payload-hash references. Train/validation remained 16,827/3,345 frames from 10/2 episodes, with independent v0.10 and v0.25 contract roots.

## Execution and stop reason

- Continuation epochs completed: `[11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]`.
- Primary decoded epochs: `[10, 20, 30, 40]`.
- Stop reason: `epoch 40 maximum reached; final selection performed`.
- Historical epochs 1–10 training/decision wall: `1554.798 s`; continuation training/evaluation wall: `4118.661 s`; supervisor wall: `4127.947 s`.
- Peak continuation CUDA allocated/reserved: `4390.8/5664.0 MiB`.

## Primary v0.10 comparison

| Candidate | Veh P/R/F1 | Veh R@.02 | Veh XY/dim/yaw | Person P/R/F1 | Person R@.02 | Person XY/dim/yaw | IoU veh/person | fg mIoU | dup FP | targets | eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| amended_baseline | 0.7125/0.8078/0.7572 | 0.8455 | 0.984/0.206/45.74 | 0.4956/0.4641/0.4793 | 0.5607 | 1.396/0.101/85.39 | 0.8655/0.4437 | 0.6546 | 979 | 2/9 | True |
| epoch_010 | 0.6890/0.8579/0.7643 | 0.8875 | 0.865/N/A/N/A | 0.4158/0.5524/0.4745 | 0.6829 | 1.366/N/A/N/A | 0.8679/0.4468 | 0.6574 | 1274 | 3/9 | True |
| epoch_020 | 0.7541/0.8467/0.7977 | 0.8787 | 0.833/0.186/36.91 | 0.4944/0.5315/0.5123 | 0.6477 | 1.339/0.091/83.01 | 0.8691/0.4504 | 0.6597 | 974 | 2/9 | True |
| epoch_030 | 0.7896/0.8433/0.8156 | 0.8776 | 0.832/0.182/35.63 | 0.5088/0.5253/0.5169 | 0.6276 | 1.339/0.090/82.92 | 0.8703/0.4530 | 0.6616 | 632 | 2/9 | True |
| epoch_040 | 0.7949/0.8391/0.8164 | 0.8729 | 0.827/0.180/35.36 | 0.5374/0.5176/0.5273 | 0.6116 | 1.343/0.090/83.50 | 0.8706/0.4537 | 0.6622 | 637 | 2/9 | True |

Epoch-10 dimension/yaw entries are `N/A`: the already-authorized epoch-10 scorer did not record them and its predictions had already been cleaned; epoch 10 was not re-decoded on primary v0.10. Later entries use the frozen 3 m match assignments and reconcile TP/FP/FN/P/R/F1/XY exactly before adding dimension/yaw diagnostics.

## Duplicate FP and world-error taxonomy

- amended_baseline: vehicle FP `{'BACKGROUND_OR_OTHER': 485, 'PREDICTED_DUPLICATE': 979, 'TWO_D_CORRECT_WORLD_WRONG': 1694}`; person FN@0.02 `{'CENTER_PRESENT_WORLD_WRONG': 854, 'HEATMAP_CENTER_MISS': 685, 'MATCHING_CONTENTION': 162}`.
- epoch_010: vehicle FP `{'BACKGROUND_OR_OTHER': 496, 'PREDICTED_DUPLICATE': 1274, 'TWO_D_CORRECT_WORLD_WRONG': 1982}`; person FN@0.02 `{'CENTER_PRESENT_WORLD_WRONG': 816, 'HEATMAP_CENTER_MISS': 263, 'MATCHING_CONTENTION': 149}`.
- epoch_020: vehicle FP `{'BACKGROUND_OR_OTHER': 315, 'PREDICTED_DUPLICATE': 974, 'TWO_D_CORRECT_WORLD_WRONG': 1387}`; person FN@0.02 `{'CENTER_PRESENT_WORLD_WRONG': 819, 'HEATMAP_CENTER_MISS': 380, 'MATCHING_CONTENTION': 165}`.
- epoch_030: vehicle FP `{'BACKGROUND_OR_OTHER': 260, 'PREDICTED_DUPLICATE': 632, 'TWO_D_CORRECT_WORLD_WRONG': 1285}`; person FN@0.02 `{'CENTER_PRESENT_WORLD_WRONG': 779, 'HEATMAP_CENTER_MISS': 495, 'MATCHING_CONTENTION': 168}`.
- epoch_040: vehicle FP `{'BACKGROUND_OR_OTHER': 234, 'PREDICTED_DUPLICATE': 637, 'TWO_D_CORRECT_WORLD_WRONG': 1227}`; person FN@0.02 `{'CENTER_PRESENT_WORLD_WRONG': 772, 'HEATMAP_CENTER_MISS': 560, 'MATCHING_CONTENTION': 172}`.

Duplicate FP remained a reported metric and the final ranking tie-breaker; it was never an intermediate stop condition.

## Final selection and service targets

- Eligible candidates in rank order: `[{'label': 'epoch_010', 'mean_class_f1': 0.6193747871902358, 'mean_xy_mae_m': 1.1152781375209346, 'minimum_class_recall': 0.5524276859504132, 'service_target_count': 3, 'vehicle_duplicate_fp': 1274}, {'label': 'epoch_020', 'mean_class_f1': 0.6549725214194511, 'mean_xy_mae_m': 1.0858565124322528, 'minimum_class_recall': 0.53150826446281, 'service_target_count': 2, 'vehicle_duplicate_fp': 974}, {'label': 'epoch_030', 'mean_class_f1': 0.6662342405405199, 'mean_xy_mae_m': 1.0856446842336203, 'minimum_class_recall': 0.5253099173553719, 'service_target_count': 2, 'vehicle_duplicate_fp': 632}, {'label': 'epoch_040', 'mean_class_f1': 0.6718619589346079, 'mean_xy_mae_m': 1.0850873230255786, 'minimum_class_recall': 0.5175619834710744, 'service_target_count': 2, 'vehicle_duplicate_fp': 637}, {'label': 'amended_baseline', 'mean_class_f1': 0.6182488114739384, 'mean_xy_mae_m': 1.1902139801692706, 'minimum_class_recall': 0.46410123966942146, 'service_target_count': 2, 'vehicle_duplicate_fp': 979}]`.
- Selected checkpoint: `/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/route_b_v3_1_native_grid_expanded_training_v2/20260828_103000/checkpoints/route_b_v3_1_native_grid_expanded_training_v2/epoch_010.pt`.
- Selected SHA-256: `26763b2955258ba7bc0287a702788d28f03d82fd504921030af53951b5b39e11`.
- Exact remaining service gaps: `{'vehicle_precision': 0.11095640643129456, 'person_precision': 0.3841757387247279, 'person_recall': 0.2475723140495868, 'person_xy_mae_m': 0.16571555442917996, 'person_box_mask_iou': 0.05321580736797171, 'foreground_miou': 0.01763328087777083}`.

| Target | Pass |
|---|---:|
| foreground_miou_ge_0_675 | False |
| person_box_mask_iou_ge_0_50 | False |
| person_precision_ge_0_80 | False |
| person_recall_ge_0_80 | False |
| person_xy_mae_le_1_2m | False |
| vehicle_iou_ge_0_85 | True |
| vehicle_precision_ge_0_80 | False |
| vehicle_recall_ge_0_85 | True |
| vehicle_xy_mae_le_1_0m | True |

## Selected-only v0.25 sensitivity

Distinct v0.25 eligible-GT denominators: `{'person': 3376, 'vehicle': 8385}`. Vehicle P/R/F1/R@.02/XY/dim/yaw: `0.6955/0.9085/0.7878/0.9277/0.824/0.194/38.14`. Person: `0.4097/0.5969/0.4859/0.6976/1.361/0.092/85.10`. This is sensitivity only; its distinct eligibility denominator is not described as model improvement.

## Recovery, cleanup, and scope

- Supervisor recoveries: `[{'kind': 'rejected_automation_qualification', 'schema': 'route_b_v3_1_native_grid_expanded_rejected_automation_attempt_v3', 'created_utc': '2026-08-28T11:49:50.386638+00:00', 'experiment': '/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/route_b_v3_1_native_grid_expanded_continuation_v3/20260828_044200', 'reason': 'worker incorrectly promoted a finite-loss GradScaler backoff to a catastrophic regression; this guard was not part of the registered AMP policy', 'epochs_completed': 0, 'continuation_checkpoints_created': 0, 'resume_origin_sha256_after_attempt': '26763b2955258ba7bc0287a702788d28f03d82fd504921030af53951b5b39e11', 'training_or_evaluation_contract_changed': False, 'preserved_renames': {'COMPLETION_SENTINEL': 'REJECTED_AUTOMATION_SENTINEL', 'TERMINAL_VERDICT.txt': 'REJECTED_AUTOMATION_TERMINAL_VERDICT.txt', 'PIPELINE_COMPLETE.json': 'REJECTED_AUTOMATION_PIPELINE_COMPLETE.json', 'FINAL_REPORT.md': 'REJECTED_AUTOMATION_REPORT.md', 'DECISION.json': 'REJECTED_AUTOMATION_DECISION.json'}, 'rejected_pointer': '/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_native_grid_expanded_training_v2/NATIVE_GRID_EXPANDED_CONTINUATION_REJECTED_ATTEMPT_1_EXP_DIR.txt', 'rejected_report': '/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_native_grid_expanded_training_v2/ROUTE_B_V3_1_NATIVE_GRID_EXPANDED_CONTINUATION_REJECTED_AUTOMATION_ATTEMPT_1_REPORT.md'}]`.
- Narrow AMP recovery: `None`.
- Cleanup/retention: `{'performed': True, 'retained_checkpoint': '/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/route_b_v3_1_native_grid_expanded_training_v2/20260828_103000/checkpoints/route_b_v3_1_native_grid_expanded_training_v2/epoch_010.pt', 'retained_checkpoint_sha256': '26763b2955258ba7bc0287a702788d28f03d82fd504921030af53951b5b39e11', 'removed_checkpoints': ['/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/route_b_v3_1_native_grid_expanded_continuation_v3/20260828_045200/checkpoints/route_b_v3_1_native_grid_expanded_continuation_v3/epoch_020.pt', '/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/route_b_v3_1_native_grid_expanded_continuation_v3/20260828_045200/checkpoints/route_b_v3_1_native_grid_expanded_continuation_v3/epoch_030.pt', '/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/route_b_v3_1_native_grid_expanded_continuation_v3/20260828_045200/checkpoints/route_b_v3_1_native_grid_expanded_continuation_v3/epoch_040.pt'], 'removed_prediction_directories': ['/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/route_b_v3_1_native_grid_expanded_continuation_v3/20260828_045200/predictions/continued_epoch_030', '/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/route_b_v3_1_native_grid_expanded_continuation_v3/20260828_045200/predictions/continued_epoch_020', '/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/route_b_v3_1_native_grid_expanded_continuation_v3/20260828_045200/predictions/continued_epoch_040', '/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/route_b_v3_1_native_grid_expanded_continuation_v3/20260828_045200/predictions/selected_sensitivity_epoch_010'], 'datasets_or_contracts_removed': 0}`.
- Desktop notification: `{'command': ['notify-send', 'LR-ASPP expanded continuation complete', 'LRASPP_EXPANDED_LONGTRAIN_IMPROVED_NOT_SERVICE_READY\n/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/route_b_v3_1_native_grid_expanded_continuation_v3/20260828_045200'], 'returncode': 0, 'stdout': '', 'stderr': '', 'delivered': True}`.
- Locked test, CARLA, the pre-existing dirty OAI submodule, q/AE, feature-drop behavior, and the 288 measurements were untouched. No architecture, loss weight, sampler, batch size, optimizer, schedule, seed, decoder, threshold, NMS rule, dataset, or postprocessor changed.

Human inspection commands (read-only):

```bash
jq . /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/route_b_v3_1_native_grid_expanded_continuation_v3/20260828_045200/STATUS.json
tail -n 8 /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/route_b_v3_1_native_grid_expanded_continuation_v3/20260828_045200/PROGRESS.csv
sed -n '1,220p' /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/route_b_v3_1_native_grid_expanded_continuation_v3/20260828_045200/logs/training.log
cat /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/route_b_v3_1_native_grid_expanded_continuation_v3/20260828_045200/COMPLETION_SENTINEL
```
