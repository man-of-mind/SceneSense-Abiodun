# Route B v3.1 native-grid expanded training v2 report

Terminal: `LRASPP_EXPANDED_TRAINING_STOPPED_EPOCH10_INSTABILITY`

Experiment: `/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/route_b_v3_1_native_grid_expanded_training_v2/20260828_103000`

## Expanded-view and immutable-source contract

- Train: 10 episodes / 16,827 frames; validation: 2 episodes / 3,345 frames; test absent.
- 40,132 symlinks; copied corpus payloads: 0.
- v0.10 train/validation positives and ignores: 64,516/290,498 and 13,597/57,601.
- Camera-plane localization-ignore v0.10 train/validation: 184/34; v0.25: 25/1.
- Expanded-view summary SHA-256: `e6bc2a9b5f88ba1176dd2f10807ca8ffd2b0d3f1376cfe3657a2d5b1fbc098a0`.
- Retained validation hashes: `{'v010/val/object_boxes.csv': 'fccf42b17c7468a85bfe367a209fb205a992e6e39b3a0300c5eb9c8b47a6cb08', 'v010/val/object_ignore_regions.csv': '7211d882907432f85abd201ff0e7407642ac2b366d9afa5bb5bf08101484c842', 'v010/val/target_manifest.csv': 'e6105ed3e87c8a5fd39be30ccc10f704fe276e3973c316688220fe6cf36e6ab5', 'v025/val/object_boxes.csv': '1c69b16c3dce41d48057e36d4ceb436f3415529cdc942ff7388e553847325a11', 'v025/val/object_ignore_regions.csv': '51c322a6df0104b387bd0df9bdd580f41d56f8a299d3e9f7c2725357737db50c', 'v025/val/target_manifest.csv': 'f72b89bf7cb51bb0fbb8c507f7d29e7e3c014bf91518e496490b5dbc2da66730'}`.
- Warm start SHA-256: `1245b2028372d486ed0b25b8a6b8a3e8b341257d542ec57cfdabf3b543d7c9ed`.
- Imported native model/target/loss/decoder hashes: `{'configs/native_grid_training_v1.json': '0bf211085af52b1039d886d5c82e03bca9836e3fb5202600ddde39d97508a8d8', 'configs/route_b_v3_1_native_grid_v1.yaml': '111a1bf1d8834941a33fe71d43eaf69429cffd76d79bd3a20f6b36789c517148', 'decode_v1.py': 'c1c963d14dc738599fcad33e413412684f176931838b3eec9bd100ceefb5cd9a', 'infer_native_v1.py': 'a14b7f7b0d49c5b95accc8c6a81a2ca478a885a340b4647e9399986ea1b5a869', 'losses_v1.py': '101887cfb4d675a6b01c172a8be8c6d36bba19c4aed09a0fc3a4ba3b98111828', 'model_v1.py': '8ddafea929ad35a7ad63825d78b96d62cbde223e64314ba9771c73dcfbbabab3', 'targets_v1.py': 'd8488ef9307066dfdb441fe2115fb4f7fa2154f3628b67503fd5bf4c0a025ff6'}`. No immutable native source was edited.

All eight bounded preflight checks passed using `/usr/bin/python3` on the sm_120 RTX 5090, including one real q=0 AMP Stage-H2 batch and exact explicit {low,high} split parity.

## Registered LR schedule and execution proof

H2 epochs 1–5 froze backbone/classifier and all BatchNorm, warmed the object group linearly for 500 optimizer steps to `1e-4`, then held it. J2 epochs 6–7 warmed object/inherited groups linearly to `5e-5/5e-6`; epochs 8–40 used cosine decay toward exactly 10% of each peak. AdamW weight decay was `1e-4`; batch 16; AMP cache disabled.

- First optimizer step: object `2e-07`, inherited `0`.
- Step 500: object `0.0001`, inherited `0`.
- Last executed step 10520 (epoch 10): object `4.90891133396e-05`, inherited `4.90891133396e-06`.
- Checkpoints contain model, optimizer, scheduler, GradScaler, epoch, Python/NumPy/Torch/CUDA RNG states, resolved config, view hashes, and warm-start hash.

## Epoch-wise loss and LR

| Ep | Stage | Object LR start→end | Inherited LR start→end | Train / val loss | Centre / offset / XYZ / bbox | Dim / yaw / seg | V/P/FG IoU |
|---:|:---:|---|---|---|---|---|---|
| 1 | H2 | 2e-07→0.0001 | 0→0 | 4.7806 / 6.8468 | 0.8890 / 0.0663 / 1.7198 / 0.2367 | 0.2057 / 0.5091 / 0.1631 | 0.8656 / 0.4426 / 0.6541 |
| 2 | H2 | 0.0001→0.0001 | 0→0 | 3.5246 / 6.6380 | 0.8813 / 0.0648 / 1.6023 / 0.2381 | 0.2047 / 0.5107 / 0.1631 | 0.8656 / 0.4426 / 0.6541 |
| 3 | H2 | 0.0001→0.0001 | 0→0 | 3.0541 / 6.5799 | 0.8703 / 0.0657 / 1.6008 / 0.2317 | 0.2046 / 0.4891 / 0.1631 | 0.8656 / 0.4426 / 0.6541 |
| 4 | H2 | 0.0001→0.0001 | 0→0 | 2.7795 / 6.5097 | 0.8788 / 0.0649 / 1.5389 / 0.2294 | 0.1996 / 0.4658 / 0.1631 | 0.8656 / 0.4426 / 0.6541 |
| 5 | H2 | 0.0001→0.0001 | 0→0 | 2.5275 / 6.6280 | 0.9090 / 0.0652 / 1.5429 / 0.2253 | 0.1960 / 0.4647 / 0.1631 | 0.8656 / 0.4426 / 0.6541 |
| 6 | J2 | 2.38e-08→2.5e-05 | 2.38e-09→2.5e-06 | 2.1696 / 6.6254 | 0.9154 / 0.0659 / 1.5283 / 0.2200 | 0.1944 / 0.4609 / 0.1624 | 0.8669 / 0.4440 / 0.6554 |
| 7 | J2 | 2.5e-05→5e-05 | 2.5e-06→5e-06 | 2.1217 / 6.7417 | 0.9534 / 0.0663 / 1.5006 / 0.2199 | 0.1979 / 0.4580 / 0.1620 | 0.8676 / 0.4453 / 0.6564 |
| 8 | J2 | 5e-05→4.99e-05 | 5e-06→4.99e-06 | 2.0485 / 6.7084 | 0.9491 / 0.0672 / 1.4939 / 0.2196 | 0.1943 / 0.4545 / 0.1613 | 0.8679 / 0.4452 / 0.6565 |
| 9 | J2 | 4.99e-05→4.96e-05 | 4.99e-06→4.96e-06 | 1.9326 / 6.7632 | 0.9718 / 0.0664 / 1.4696 / 0.2197 | 0.1963 / 0.4497 / 0.1611 | 0.8678 / 0.4454 / 0.6566 |
| 10 | J2 | 4.96e-05→4.91e-05 | 4.96e-06→4.91e-06 | 1.8286 / 6.8018 | 0.9779 / 0.0657 / 1.4785 / 0.2203 | 0.1950 / 0.4596 / 0.1607 | 0.8682 / 0.4459 / 0.6570 |

Loss-best decoded checkpoint: epoch 5 at validation loss `6.627999`, SHA-256 `bfd9ed8c3562763e3f674d9a20d47d75b1e1cd4a6af66c91e2cca3be72aad2b0`. It was not auto-promoted.

## Authorized validation decodes

| Ep | Vehicle P/R/F1 | V XY | V R@.02 | Person P/R/F1 | P XY | P R@.02 | Dup FP | Heatmap miss | V/P/FG IoU |
|---:|---|---:|---:|---|---:|---:|---:|---:|---|
| 5 | 0.6553/0.8550/0.7420 | 0.9170 | 0.8863 | 0.3217/0.5586/0.4083 | 1.3947 | 0.6888 | 1682 | 217 | 0.8655/0.4437/0.6546 |
| 10 | 0.6890/0.8579/0.7643 | 0.8648 | 0.8875 | 0.4158/0.5524/0.4745 | 1.3657 | 0.6829 | 1274 | 263 | 0.8679/0.4468/0.6574 |

Decoded epochs were exactly `[5, 10]`; each used one inference pass at score floor 0.02 for both registered thresholds.

## Decision gates and selection

- Epoch-10 stability: `FAIL`; failed gates: `['duplicate_fp_le_baseline_plus_20pct']`.
- Epoch-20 continuation: Not reached.
- Primary-eligible epochs in registered rank order: `[10]`.
- Selected checkpoint: none.
- Best-ranked regardless of eligibility, retained when no selected checkpoint exists: epoch 10, `/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/route_b_v3_1_native_grid_expanded_training_v2/20260828_103000/checkpoints/route_b_v3_1_native_grid_expanded_training_v2/epoch_010.pt`, SHA-256 `26763b2955258ba7bc0287a702788d28f03d82fd504921030af53951b5b39e11`.

Baseline taxonomy: `{'vehicle': {'PREDICTED_DUPLICATE': 979, 'TWO_D_CORRECT_WORLD_WRONG': 1694}, 'person': {'CENTER_PRESENT_WORLD_WRONG': 854, 'HEATMAP_CENTER_MISS': 685}}`. Selected taxonomy: `None`. Best-ranked diagnostic taxonomy: `{'person_fn_at_0_02': {'counts': {'CENTER_PRESENT_WORLD_WRONG': 816, 'HEATMAP_CENTER_MISS': 263, 'MATCHING_CONTENTION': 149}, 'denominator': 1228, 'labels_sum_to_denominator': True, 'percentages': {'CENTER_PRESENT_WORLD_WRONG': 66.44951140065146, 'HEATMAP_CENTER_MISS': 21.416938110749186, 'MATCHING_CONTENTION': 12.133550488599349}, 'total_labelled': 1228}, 'vehicle_fp_at_0_20': {'counts': {'BACKGROUND_OR_OTHER': 496, 'PREDICTED_DUPLICATE': 1274, 'TWO_D_CORRECT_WORLD_WRONG': 1982}, 'denominator': 3752, 'labels_sum_to_denominator': True, 'percentages': {'BACKGROUND_OR_OTHER': 13.219616204690832, 'PREDICTED_DUPLICATE': 33.95522388059702, 'TWO_D_CORRECT_WORLD_WRONG': 52.82515991471215}, 'total_labelled': 3752, 'unprioritised_overlap': {'both': 406, 'duplicate_any': 1274, 'two_d_any': 2388}}}`.

Selected v0.10 metrics: `None`.

Selected-only v0.25 sensitivity: `None`. Sensitivity reversal gates: `None`. No substitute or additional checkpoint received v0.25 scoring.

Material-gain gates: `None`.

## Blocking service targets

No selected checkpoint; all nine blocking service targets remain unassigned.

q/AE was not started.

## Runtime, resources, cleanup, and scope

- Training/evaluation wall: `1554.798 s`; full pipeline wall: `1561.766 s`.
- Peak CUDA allocated/reserved: `4386.4/4986.0 MiB`.
- Cleanup: `{'schema': 'route_b_v3_1_native_grid_expanded_training_cleanup_v2', 'created_utc': '2026-08-28T11:07:15.700534+00:00', 'terminal': 'LRASPP_EXPANDED_TRAINING_STOPPED_EPOCH10_INSTABILITY', 'cleanup_required': True, 'removed_nonselected_checkpoints': [{'path': '/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/route_b_v3_1_native_grid_expanded_training_v2/20260828_103000/checkpoints/route_b_v3_1_native_grid_expanded_training_v2/epoch_005.pt', 'sha256': 'bfd9ed8c3562763e3f674d9a20d47d75b1e1cd4a6af66c91e2cca3be72aad2b0'}], 'removed_raw_inference_payloads': [{'path': '/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/route_b_v3_1_native_grid_expanded_training_v2/20260828_103000/predictions/expanded_epoch_005', 'checkpoint_epoch': 5, 'checkpoint_sha256': 'bfd9ed8c3562763e3f674d9a20d47d75b1e1cd4a6af66c91e2cca3be72aad2b0', 'detections_sha256': 'd8ab6e465e7aa08be64cfe7635b95267d1cdb87456945fbb01a158026d72a30b', 'prediction_set_sha256': '33b722df8b386ba11e1e453d04d9a974480d9d067c9ae0dbfa9fb74b5aa2d79e'}, {'path': '/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/route_b_v3_1_native_grid_expanded_training_v2/20260828_103000/predictions/expanded_epoch_010', 'checkpoint_epoch': 10, 'checkpoint_sha256': '26763b2955258ba7bc0287a702788d28f03d82fd504921030af53951b5b39e11', 'detections_sha256': '4f5dc943c84ddedd9c4b8e5e09e850447e490a0a0c920b0e505fa12fa0174ce6', 'prediction_set_sha256': '4c3bf24fc4a1642974dabc3e24e8747628a3448cac68f0fefab9636ffa5306fa'}], 'retained_checkpoint_epoch': 10, 'retained_checkpoint': '/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/route_b_v3_1_native_grid_expanded_training_v2/20260828_103000/checkpoints/route_b_v3_1_native_grid_expanded_training_v2/epoch_010.pt', 'retained_checkpoint_sha256': '26763b2955258ba7bc0287a702788d28f03d82fd504921030af53951b5b39e11', 'warm_start_or_corpus_payloads_removed': 0}`.
- Test, CARLA, OAI, containers, q/AE, feature drop, and 288 measurements were untouched. No decoder calibration, threshold/NMS sweep, loss sweep, or follow-up experiment ran.
