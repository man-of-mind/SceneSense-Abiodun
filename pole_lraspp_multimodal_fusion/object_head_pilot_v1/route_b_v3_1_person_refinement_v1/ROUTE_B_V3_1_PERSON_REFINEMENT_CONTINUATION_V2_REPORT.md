# Route B v3.1 LR-ASPP accepted epoch-40 person-refinement continuation v2

Terminal: `LRASPP_PERSON_REFINEMENT_CONTRACT_INVALID`

Experiment: `/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/route_b_v3_1_person_refinement_continuation_v2/20260828_134000`

Accepted decision: `RECOVERED_EPOCH40_ACCEPTED_WITH_FAVORABLE_LOW_THRESHOLD_VARIATION`.
Required starting local master HEAD: `94e13aa2d06f11143d66122a037496b96ecc985c`. Nothing was pushed.
Historical terminal remains `LRASPP_PERSON_REFINEMENT_BASE_RECOVERY_FAILED` in `experiments/route_b_v3_1_person_refinement_v1/20260828_163100` and was not rewritten.
Accepted recovered epoch-40 checkpoint: `experiments/route_b_v3_1_person_refinement_v1/20260828_163100/checkpoints/route_b_v3_1_person_refinement_v1/epoch_040.pt` (`5c6bb268b43f4dd84bd7a283ff483ec4e87366a50ea51dfacee44979df2bf6e8`).
The only old reconciliation variation was favorable person R@0.02 `0.6151859504132231` versus `0.6115702479338843` (`+14` TP). Candidate deltas use the recovered checkpoint's own decoded metrics.
Epochs 11–40 were not repeated.

## Execution

Runtime retries used: `0` of one. Error: `ContractInvalid: repaired person-refinement qualification failed`.
Notification result: `{'command': ['notify-send', 'LR-ASPP accepted person refinement complete', 'LRASPP_PERSON_REFINEMENT_CONTRACT_INVALID\n/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/route_b_v3_1_person_refinement_continuation_v2/20260828_134000'], 'returncode': 0, 'stdout': '', 'stderr': '', 'delivered': True}`.

## Scope confirmation

Locked test remained absent and unopened. No q/AE, tracking, CARLA, OAI, live-runtime, calibrated-threshold, alternative-architecture, second-hyperparameter, or 288-measurement work was performed. Datasets, predictions, and checkpoints are not committed.

Supervisor wall time: `6.0` seconds.
