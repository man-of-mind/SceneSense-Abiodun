# Route B v3.1 LR-ASPP accepted epoch-40 person-refinement continuation v2

Terminal: `LRASPP_PERSON_REFINEMENT_RUNTIME_FAILURE`

Experiment: `/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/route_b_v3_1_person_refinement_continuation_v2/20260828_134000`

Accepted decision: `RECOVERED_EPOCH40_ACCEPTED_WITH_FAVORABLE_LOW_THRESHOLD_VARIATION`.
Required starting local master HEAD: `94e13aa2d06f11143d66122a037496b96ecc985c`. Nothing was pushed.
Historical terminal remains `LRASPP_PERSON_REFINEMENT_BASE_RECOVERY_FAILED` in `experiments/route_b_v3_1_person_refinement_v1/20260828_163100` and was not rewritten.
Accepted recovered epoch-40 checkpoint: `experiments/route_b_v3_1_person_refinement_v1/20260828_163100/checkpoints/route_b_v3_1_person_refinement_v1/epoch_040.pt` (`5c6bb268b43f4dd84bd7a283ff483ec4e87366a50ea51dfacee44979df2bf6e8`).
The only old reconciliation variation was favorable person R@0.02 `0.6151859504132231` versus `0.6115702479338843` (`+14` TP). Candidate deltas use the recovered checkpoint's own decoded metrics.
Epochs 11–40 were not repeated.

## Execution

Runtime retries used: `1` of one. Error: `RuntimeError: person training failed before an epoch-boundary recovery checkpoint`.
Notification result: `{'command': ['notify-send', 'LR-ASPP accepted person refinement complete', 'LRASPP_PERSON_REFINEMENT_RUNTIME_FAILURE\n/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/route_b_v3_1_person_refinement_continuation_v2/20260828_134000'], 'returncode': 0, 'stdout': '', 'stderr': '', 'delivered': True}`.

## Frozen design and qualification

The unchanged prepared design uses a person-private fused-feature proposal trunk, objectness and detached localization-quality heads, eight train-derived range bins plus bounded residual, projected-center offset with camera unprojection, an independent person-mask residual, bounded train-only hard-negative mining, and deterministic capped episode/track sampling.
P2 trainable/frozen parameter counts: `{'backbone': {'frozen': 2972528, 'total': 2972528, 'trainable': 0}, 'grid_offset': {'frozen': 258, 'total': 258, 'trainable': 0}, 'model_total': {'frozen': 4931069, 'total': 5126476, 'trainable': 195407}, 'native_shared': {'frozen': 1447680, 'total': 1447680, 'trainable': 0}, 'native_upsampler': {'frozen': 262400, 'total': 262400, 'trainable': 0}, 'person_heatmap': {'frozen': 0, 'total': 129, 'trainable': 129}, 'person_refinement': {'frozen': 0, 'total': 195278, 'trainable': 195278}, 'segmentation': {'frozen': 246526, 'total': 246526, 'trainable': 0}, 'shared_regression': {'frozen': 1548, 'total': 1548, 'trainable': 0}, 'vehicle_heatmap': {'frozen': 129, 'total': 129, 'trainable': 0}}`.
Transported bundle names/shapes/dtypes: `['low', 'high']` / `{'high': [1, 960, 27, 48], 'low': [1, 40, 54, 96]}` / `{'high': 'torch.float32', 'low': 'torch.float32'}`. Raw side channels: `[]`. Monolithic/split bit parity: `True`.
Train-derived range edges/counts: `[0.0, 10.226283381312156, 13.941271025603957, 16.836130343234398, 19.404202822687676, 22.65871454221351, 26.245333730676172, 31.11058219414504, 40.0]` / `[2199, 2198, 2198, 2198, 2199, 2198, 2198, 2199]`. Validation rows used for sampling/mining/training: `0`.

## Scope confirmation

Locked test remained absent and unopened. No q/AE, tracking, CARLA, OAI, live-runtime, calibrated-threshold, alternative-architecture, second-hyperparameter, or 288-measurement work was performed. Datasets, predictions, and checkpoints are not committed.

Supervisor wall time: `61.9` seconds.
