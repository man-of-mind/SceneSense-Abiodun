# Route B v3.1 LR-ASPP person refinement v1 report

Terminal: `LRASPP_PERSON_REFINEMENT_BASE_RECOVERY_FAILED`

Experiment: `/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/route_b_v3_1_person_refinement_v1/20260828_163100`

Execution began on local `master` at required HEAD `b010ec759450854d291eb9cc7ea06f6ff32aa2fc`. Nothing was pushed.

## Contract and recovery

- Exact epoch-10 origin: `experiments/route_b_v3_1_native_grid_expanded_training_v2/20260828_103000/checkpoints/route_b_v3_1_native_grid_expanded_training_v2/epoch_010.pt` (`26763b2955258ba7bc0287a702788d28f03d82fd504921030af53951b5b39e11`).
- Frozen expanded data scope: 16,827 train / 3,345 validation; locked test absent and unopened.
- Clean execution: `/usr/bin/python3`, CUDA sm_120, q=0, AE disabled, no geometric augmentation.
- Runtime retries used: `0` of one; error: `BaseRecoveryFailed: epoch-40 reconciliation failed: {'vehicle_precision': True, 'vehicle_recall': True, 'vehicle_f1': True, 'vehicle_recall_002': True, 'vehicle_xy_mae_m': True, 'person_precision': True, 'person_recall': True, 'person_f1': True, 'person_recall_002': False, 'person_xy_mae_m': True, 'vehicle_iou': True, 'person_box_mask_iou': True, 'foreground_miou': True, 'vehicle_duplicate_fp': True}`.
- Recovered epochs: `[11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]`; decoded epochs: `[40]`.
- Retained base checkpoints: `[{'epoch': 20, 'path': '/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/route_b_v3_1_person_refinement_v1/20260828_163100/checkpoints/route_b_v3_1_person_refinement_v1/epoch_020.pt', 'sha256': '86867e7dbf5d67063aef7b4293870c4b39224779cefb558e20ca975b74ff24d5'}, {'epoch': 30, 'path': '/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/route_b_v3_1_person_refinement_v1/20260828_163100/checkpoints/route_b_v3_1_person_refinement_v1/epoch_030.pt', 'sha256': '01787c40d2ce9235cfaafc973d0bb4ff2c7898dd93c0909aef54ce70c5ac25bd'}, {'epoch': 40, 'path': '/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/route_b_v3_1_person_refinement_v1/20260828_163100/checkpoints/route_b_v3_1_person_refinement_v1/epoch_040.pt', 'sha256': '5c6bb268b43f4dd84bd7a283ff483ec4e87366a50ea51dfacee44979df2bf6e8'}]`.
- Epoch-40 reconciliation failed closed. Person R@0.02 was `0.6151859504` versus reference `0.6115702479`, an absolute delta of `0.0036157025` against the `0.002` tolerance. Every other registered reconciliation gate passed.

## Registered refinement

The prepared private person tail consumes only the transported `low`/`high` feature bundle. It adds person objectness residual, detached 3 m localization quality, eight train-derived range bins plus bounded residual, projected-center offset with external camera unprojection, and an independent person-mask residual. The recovered backbone, shared object trunk, vehicle heatmap, shared regression, grid offset, and vehicle segmentation path are configured frozen; only the inherited person heatmap slice is configured for lower-LR P2 training (epochs 7–18).

The base reconciliation gate failed before Phase B, so the diagnostic, executable qualification, registration artifact, candidate training, candidate scoring, and v0.25 sensitivity were not run. The prepared refinement implementation remains unexecuted scientific design.

## Operational boundary

No q/AE/tracking/calibrated-threshold/live UDP/CARLA/OAI action, alternative architecture, second experiment, locked-test access, or modification of the 288 measurement files was performed. The report is offline validation evidence only; deployment remains a separate decision.

Supervisor wall time: `3578.0` seconds.
