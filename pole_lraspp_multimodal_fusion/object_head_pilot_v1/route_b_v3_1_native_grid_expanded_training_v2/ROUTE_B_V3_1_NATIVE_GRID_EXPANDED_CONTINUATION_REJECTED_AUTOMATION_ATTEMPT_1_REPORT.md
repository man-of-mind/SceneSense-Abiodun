# Route B v3.1 native-grid expanded continuation v3 report

Terminal: `LRASPP_EXPANDED_LONGTRAIN_CATASTROPHIC_REGRESSION`

Experiment: `/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/route_b_v3_1_native_grid_expanded_continuation_v3/20260828_044200`

Execution commit on local `master`: `02ebabd2b6b7dd3e1ecb4b199541de2a458fda2e` (the required starting HEAD). The final source/config/report commit is the post-run local `master` HEAD reported at handoff; nothing was pushed.

## Exact continuation state

- Resume checkpoint: `experiments/route_b_v3_1_native_grid_expanded_training_v2/20260828_103000/checkpoints/route_b_v3_1_native_grid_expanded_training_v2/epoch_010.pt`.
- Verified SHA-256: `26763b2955258ba7bc0287a702788d28f03d82fd504921030af53951b5b39e11`.
- State: epoch `10`, optimizer steps `10520`, 1,052 steps/epoch; model, AdamW optimizer, registered H2/J2 scheduler, AMP GradScaler, and Python/NumPy/Torch CPU/CUDA RNG state were present and strictly loadable.
- End-of-epoch-10 inherited/object LR: `{'inherited': 4.908911333958271e-06, 'object': 4.908911333958272e-05}`; these reconcile to the registered cosine schedule.
- Epochs 1–10 were not repeated and the epoch-15 warm start was not used to restart this continuation.
- All recorded view/manifest/GT/ignore/camera-plane hashes passed, including `80688` payload-hash references. Train/validation remained 16,827/3,345 frames from 10/2 episodes, with independent v0.10 and v0.25 contract roots.

## Execution and stop reason

- Continuation epochs completed: `[]`.
- Primary decoded epochs: `[10]`.
- Stop reason: `CatastrophicRegression: FP32 segmentation-loss recovery did not clear AMP overflow`.
- Historical epochs 1–10 training/decision wall: `1554.798 s`; continuation training/evaluation wall: `60.029 s`; supervisor wall: `335.598 s`.
- Peak continuation CUDA allocated/reserved: `0.0/0.0 MiB`.

## Primary v0.10 comparison

| Candidate | Veh P/R/F1 | Veh R@.02 | Veh XY/dim/yaw | Person P/R/F1 | Person R@.02 | Person XY/dim/yaw | IoU veh/person | fg mIoU | dup FP | targets | eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| amended_baseline | 0.7125/0.8078/0.7572 | 0.8455 | 0.984/N/A/N/A | 0.4956/0.4641/0.4793 | 0.5607 | 1.396/N/A/N/A | 0.8655/0.4437 | 0.6546 | 979 | N/A/9 | N/A |
| epoch_010 | 0.6890/0.8579/0.7643 | 0.8875 | 0.865/N/A/N/A | 0.4158/0.5524/0.4745 | 0.6829 | 1.366/N/A/N/A | 0.8679/0.4468 | 0.6574 | 1274 | N/A/9 | N/A |

Epoch-10 dimension/yaw entries are `N/A`: the already-authorized epoch-10 scorer did not record them and its predictions had already been cleaned; epoch 10 was not re-decoded on primary v0.10. Later entries use the frozen 3 m match assignments and reconcile TP/FP/FN/P/R/F1/XY exactly before adding dimension/yaw diagnostics.

## Duplicate FP and world-error taxonomy

- amended_baseline: vehicle FP `{'BACKGROUND_OR_OTHER': 485, 'PREDICTED_DUPLICATE': 979, 'TWO_D_CORRECT_WORLD_WRONG': 1694}`; person FN@0.02 `{'CENTER_PRESENT_WORLD_WRONG': 854, 'HEATMAP_CENTER_MISS': 685, 'MATCHING_CONTENTION': 162}`.
- epoch_010: vehicle FP `{'BACKGROUND_OR_OTHER': 496, 'PREDICTED_DUPLICATE': 1274, 'TWO_D_CORRECT_WORLD_WRONG': 1982}`; person FN@0.02 `{'CENTER_PRESENT_WORLD_WRONG': 816, 'HEATMAP_CENTER_MISS': 263, 'MATCHING_CONTENTION': 149}`.

Duplicate FP remained a reported metric and the final ranking tie-breaker; it was never an intermediate stop condition.

## Final selection and service targets

- Eligible candidates in rank order: `None`.
- Selected checkpoint: `None`.
- Selected SHA-256: `None`.
- Exact remaining service gaps: `{}`.

No final service selection was performed.

## Selected-only v0.25 sensitivity

Not available because final selection did not complete.

## Recovery, cleanup, and scope

- Supervisor recoveries: `[{'schema': 'route_b_v3_1_native_grid_expanded_runtime_recovery_v3', 'created_utc': '2026-08-28T11:46:27.263004+00:00', 'attempt': 1, 'failure': {'attempt': 1, 'created_utc': '2026-08-28T11:46:27.205205+00:00', 'error': 'attempt-1 worker exception was masked by an UnboundLocalError in its failure serializer; no epoch completed and epoch 10 remains latest safe', 'kind': 'runtime', 'schema': 'route_b_v3_1_native_grid_expanded_continuation_worker_failure_v3', 'wall_seconds': 0.0}, 'gpu_diagnostic': {'command': ['nvidia-smi', '--query-gpu=name,compute_cap,memory.total,memory.used,memory.free,temperature.gpu', '--format=csv,noheader,nounits'], 'returncode': 0, 'stdout': 'NVIDIA GeForce RTX 5090, 12.0, 32607, 1071, 31031, 38', 'stderr': '', 'torch_cuda_available': True}, 'latest_safe': {'epoch': 10, 'path': '/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/route_b_v3_1_native_grid_expanded_training_v2/20260828_103000/checkpoints/route_b_v3_1_native_grid_expanded_training_v2/epoch_010.pt', 'sha256': '26763b2955258ba7bc0287a702788d28f03d82fd504921030af53951b5b39e11', 'verified_sha256': '26763b2955258ba7bc0287a702788d28f03d82fd504921030af53951b5b39e11', 'required_state_present': True}, 'moved_incomplete_artifacts': [], 'retry_count': 1, 'same_interpreter': '/usr/bin/python3', 'same_configuration_sha256': '0ccf6e428298aeaa1851b19193885c26d240cdeba084143d07bb00eb4ff10037', 'automation_patch': {'scope': 'failure serialization and existing-experiment supervisor recovery only', 'training_or_evaluation_contract_changed': False, 'post_patch_source_hashes': {'pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_native_grid_expanded_training_v2/configs/expanded_continuation_v3.json': '0ccf6e428298aeaa1851b19193885c26d240cdeba084143d07bb00eb4ff10037', 'pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_native_grid_expanded_training_v2/continuation_policy_v3.py': '1a98d5021ec83c680d619ad879bff2192cc9bea961f2a965500c51ee851c0537', 'pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_native_grid_expanded_training_v2/continuation_scoring_v3.py': '3712de191b2d5e175e55e3e41737bb0c9d85833c98a70764902795147c06eed5', 'pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_native_grid_expanded_training_v2/score_continuation_v3.py': '8f7168c261e173a00b0700484f88bdf5470deec78e6ae9ac23a2716b4c4cc46e', 'pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_native_grid_expanded_training_v2/preflight_continuation_v3.py': 'd1733eb2c2912073aed82e0f09a0a6b29d19ad59163b335c986ecca7019bb236', 'pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_native_grid_expanded_training_v2/continue_training_v3.py': '09cb95b6a43a898c7ee5dcebd3cc4387641f188c5af1273b8846dce37a07e62c', 'pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_native_grid_expanded_training_v2/run_continuation_v3.py': '5e92b614c45cd7f895b82527cfca8b37b0dc855c13c1ba0a288dc3fb8ce60552', 'pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_native_grid_expanded_training_v2/test_continuation_policy_v3.py': '351124ce3b07ae99155050e6955d95a28db00d910d40b58af7c42043d9da2545'}}}]`.
- Narrow AMP recovery: `None`.
- Cleanup/retention: `{'performed': True, 'retained_checkpoint': '/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/route_b_v3_1_native_grid_expanded_training_v2/20260828_103000/checkpoints/route_b_v3_1_native_grid_expanded_training_v2/epoch_010.pt', 'retained_checkpoint_sha256': '26763b2955258ba7bc0287a702788d28f03d82fd504921030af53951b5b39e11', 'removed_checkpoints': [], 'removed_prediction_directories': [], 'datasets_or_contracts_removed': 0}`.
- Desktop notification: `{'command': ['notify-send', 'LR-ASPP expanded continuation complete', 'LRASPP_EXPANDED_LONGTRAIN_CATASTROPHIC_REGRESSION\n/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/route_b_v3_1_native_grid_expanded_continuation_v3/20260828_044200'], 'returncode': 0, 'stdout': '', 'stderr': '', 'delivered': True}`.
- Locked test, CARLA, the pre-existing dirty OAI submodule, q/AE, feature-drop behavior, and the 288 measurements were untouched. No architecture, loss weight, sampler, batch size, optimizer, schedule, seed, decoder, threshold, NMS rule, dataset, or postprocessor changed.

Human inspection commands (read-only):

```bash
jq . /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/route_b_v3_1_native_grid_expanded_continuation_v3/20260828_044200/STATUS.json
tail -n 8 /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/route_b_v3_1_native_grid_expanded_continuation_v3/20260828_044200/PROGRESS.csv
sed -n '1,220p' /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/route_b_v3_1_native_grid_expanded_continuation_v3/20260828_044200/logs/training.log
cat /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/route_b_v3_1_native_grid_expanded_continuation_v3/20260828_044200/COMPLETION_SENTINEL
```
