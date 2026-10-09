# SplitFusion model agent and OAI pipeline architecture

For Subhramoy, from Abiodun. Snapshot prepared on 8 October 2026 for architecture slides and inspection of the implemented data structures.

This folder contains the FCOS perception model, compression modules, the deployed Run-5 agent and its training implementation, and the current Route-B pipeline through OAI to the spatial map. Start with the links below; the remaining source files support these entry points. The model folders have been shortened and renamed for clarity; other repository-relative paths are preserved under `code/`.

The perception architecture is **SplitFusion FCOS, ResNet-50 with P2–P7 FPN**. Its main implementation is now [fcos_model/network/model.py](code/fcos_model/network/model.py).

```text
code/fcos_model/
  network/                       FCOS front, tail, heads, losses and training
  feature_autoencoders/           AE128, AE64 and AE32
  spatial_compression/           Ranking, dropping and quantization
  postprocessing/                Service detection postprocessing
  person_score_calibration/       Deployed person score filter
  person_instance_consolidation/  Person candidate consolidation
  quality_metrics/               Model quality helpers
  training_target_contract/      Target definitions
  gt_target_refinement/          Supporting target logic
  training_recovery/             Training recovery implementation
  legacy_shared_helpers/         Reused helpers, not the deployed detector
```

**Status:** the deployed agent is frozen seed 43 at update 10,000. Full-route M3 integration is still being repaired: the latest V6 run started successfully but ended before completing Route B after a GT preparation failure. This is a code-reading snapshot, not a standalone runnable release or a claim that full-route validation passed. Files under `pending_fix/` are an offline-tested repair candidate, not part of that live run.

## Model and compression

| What to draw or inspect | Source |
|---|---|
| Seven-channel front, ResNet-50 tail, P2–P7 FPN, detection, geometry and dense heads | [FCOS model.py](code/fcos_model/network/model.py) |
| Input formation and model training targets | [data.py](code/fcos_model/network/data.py), [losses.py](code/fcos_model/network/losses.py), [train.py](code/fcos_model/network/train.py) |
| AE128, AE64 and AE32 encoders and decoders | [SplitFeatureAE](code/fcos_model/feature_autoencoders/ae_model.py) |
| Spatial ranking and keep/drop selection | [ranker.py](code/fcos_model/spatial_compression/ranker.py), [selection.py](code/fcos_model/spatial_compression/selection.py) |
| Quantization, sparse transport and reconstruction | [transport.py](code/rl_agent/splitfusion_live_dispatch_v1/transport.py), [lowbit_transport.py](code/fcos_model/feature_autoencoders/lowbit_transport.py) |
| The actual live postprocessing and record materialization | [context_tail.py](code/rl_agent/splitfusion_live_dispatch_v1/context_tail.py), [p025 runtime](code/fcos_model/person_score_calibration/runtime.py) |

Key dimensions:

- Native camera: 1280 × 720. Model content: 768 × 432; 16 bottom rows are padded after normalization.
- Model input: `[B, 7, 448, 768]`, combining three RGB and four radar channels. The corrected live implementation is [input preparation v2](code/rl_agent/splitfusion_ue_input_prep_v2/candidate.py).
- UE front output and split boundary: C2, `[B, 256, 112, 192]`.
- Transport families: noAE (256 channels), AE128, AE64 and AE32. The three AEs share one parameterized architecture with different latent widths; these are not four unrelated detector architectures. The dispatch uses a shared frozen FCOS tail.
- Edge reconstruction restores C2 before the tail. Outputs include class scores, 2D boxes, localized 3D positions, dimensions/yaw, and semantic labels. See `SplitFusionFCOS`, `GeometryHead`, `P2TaskHead` and the live postprocessor.

Large model checkpoints are not copied. [MODEL_WEIGHT_REFERENCES.json](MODEL_WEIGHT_REFERENCES.json) lists the deployed FCOS, AE and ranker paths and hashes.

## Deployed agent and its training

Start with [FrozenFullRun5Actor](code/rl_agent/splitfusion_run5_full_live_v1/actor.py), [Run-5 model configuration](code/rl_agent/splitfusion_hybrid_sac_run5_v1/run5_models.py), and [Hybrid-SAC networks and equations](code/rl_agent/splitfusion_hybrid_sac_v1/hybrid_sac_models.py).

- Actor: 22 inputs → two 128-wide ReLU layers → 12 categorical mode logits plus 12 conditional means and log standard deviations for spatial drop fraction `q`.
- A mode selects feature family × quantizer: 4 families × UINT8/UINT6/UINT4 = 12 modes. `q` uses the mode-specific allowed interval; larger `q` means more dropping. [ACTION_MODES.csv](ACTION_MODES.csv) contains the exact deployed bounds. Mode 11 is AE32/UINT4.
- Training uses stochastic mode and conditional-q sampling. Live inference uses deterministic argmax mode and its conditional-mean-derived q; all 12 modes remain available. Frequent mode-11 selection is learned behavior, not a mode mask.
- Two critics and two target critics are training-only. Each critic takes the 22 state values, a 12-way mode one-hot and normalized q. There is no online optimizer or critic in deployment.
- The deployed weights are included at [actor_state_dict.pt](code/rl_agent/splitfusion_run5_full_live_v1/actor_export/actor_state_dict.pt). SHA-256: `bb9b0d256de79b4498cd622e1daa59edcf43ff3072dc9f675b265aac579ed3ce`. Seed 43 was preselected for live use, not selected as the best seed afterward.

### The 22 features

Exact order: [AGENT_FEATURES.csv](AGENT_FEATURES.csv). Implementations: [Run-4 prefix contract](code/rl_agent/splitfusion_hybrid_sac_run4_v1/run4_contract.py), [Run-5 SNR contract](code/rl_agent/splitfusion_hybrid_sac_run5_v1/run5_snr_v2.py), and [SNR observation/provider definitions](code/rl_agent/splitfusion_hybrid_sac_run5_v1/run5_state_contract.py).

| Zero-based indices | Contents |
|---|---|
| 0–3 | Camera spatial information, radar P40, prior UE-decoded uplink MCS, pre-action RLC backlog |
| 4–15 | Previous joint-mode one-hot, 12 values |
| 16–20 | Previous q, previous Q_perc, previous latency, previous-present and previous-success flags |
| 21 | Causal acknowledged commanded uplink-SNR proxy, scaled as `(SNR − 5.5) / 19` |

Freshness and identity checks precede actor execution. Missing/stale current observations trigger the external fallback; they are not invented as zero measurements. The SNR input is a commanded proxy, **not** the gNB's measured PUSCH SNR. Previous-outcome masking and session resets follow the shared contract.

### Current training reward and curves

Use [full training orchestrator](code/rl_agent/splitfusion_run5_full_v1/trainer.py), [collector](code/rl_agent/splitfusion_run5_full_v1/collector.py), [latency model](code/rl_agent/splitfusion_run5_full_v1/latency.py) and [quality and reward definition](code/rl_agent/splitfusion_run5_full_v1/quality.py).

The deployed actor was trained with timely localization reward `Y × (0.7 + 0.3 Q_seg)`, zero for timeouts. `Y` weights within-tolerance recall for persons (0.6, 1.2 m) and vehicles (0.4, 1.0 m), normalized over eligible classes. The inclusive feedback deadline is 170 ms. Timely undefined quality is excluded and resets the session; late undefined quality is a timeout. Previous Q_perc remains an input and is not replaced by this reward.

Some reused Run-4/early Run-5 modules still document the historical reward or their original implementation phase. Those comments do **not** override the full-training wrapper above. Likewise, historical live reward logging is distinct from the reward used to train this frozen checkpoint.

- [Replay minibatch reward over 10,000 updates](agent_training_plots/REPLAY_REWARD.png)
- [Training mode shares and timeliness](agent_training_plots/TRAINING_DECISIONS.png)

Both plots are copied unchanged from the deployed actor's training campaign, with seeds 17, 29 and 43. Their small CSVs are included. They are modeled-training curves, not Route-B live performance plots. Later rejected reward/backlog retraining candidates are not the deployed model.

## End to end Route B pipeline

```text
CARLA RGB and radar + ego pose
          │
          ├─ causal scene/radio/history state → frozen Run-5 actor → mode, q
          │
          └─ corrected 7-channel input → UE FCOS front → C2
                → optional AE, spatial selection, quantization and packing
                → OAI UE uplink → gNB/core network → edge reassembly
                → reconstruction/AE decoder → FCOS tail and postprocessing
                       ├─ direct map ingest → map state and durable JSON journal
                       └─ exact-GT scorer → quality/latency feedback to UE
CARLA privileged evaluation inputs → parallel GT preparation ────────┘
```

| Pipeline component | First files to inspect |
|---|---|
| Current launcher and process roles | [V6 host](code/rl_agent/splitfusion_m3_route_b_v6/host.py), [runner](code/rl_agent/splitfusion_m3_route_b_v6/runner.py), [entries](code/rl_agent/splitfusion_m3_route_b_v6/entries.py), [contract](code/rl_agent/splitfusion_m3_route_b_v6/contract.py) |
| Frozen actor integration | [runtime_policy.py](code/rl_agent/splitfusion_run5_full_live_v1/runtime_policy.py) |
| UE front and edge dispatch | [ue_runtime.py](code/rl_agent/splitfusion_live_dispatch_v1/ue_runtime.py), [edge_runtime.py](code/rl_agent/splitfusion_live_dispatch_v1/edge_runtime.py) |
| Continuous action identity and map message | [run4_map_protocol_v2.py](code/rl_agent/splitfusion_hybrid_sac_live_route_b_v2/run4_map_protocol_v2.py) — this retained protocol is also used by Run-5 |
| Direct map ingestion and state | [map_ingest.py](code/rl_agent/splitfusion_direct_edge_map_v1/map_ingest.py), [spatial_map_direct_server_v1.py](code/rl_agent/splitfusion_direct_edge_map_v1/spatial_map_direct_server_v1.py) |
| Correct map classes and persisted JSON | [map_class.py](code/rl_agent/splitfusion_m3_route_b_v3/map_class.py), [map_journal.py](code/rl_agent/splitfusion_m3_route_b_v3/map_journal.py) |
| Concurrent evaluator | [service.py](code/rl_agent/splitfusion_exact_gt_service_proto_v2/service.py), [prep_core.py](code/rl_agent/splitfusion_exact_gt_service_proto_v2/prep_core.py), [scorer_core.py](code/rl_agent/splitfusion_exact_gt_service_proto_v2/scorer_core.py) |
| Route and radio | [Route-B loop](code/data_collection/routes/town10hd_opt_route_b_full_map_loop_v1.json), [GOOD_ALL controller](code/rl_agent/splitfusion_m3_route_b_v3/good_all_controller.py), [actual gNB configuration](radio_config/effective_gnb_100mhz_4d5u.conf) |

Map publication is independent of the 170-ms evaluation-feedback result. A late-feedback prediction can still update the map if it satisfies the separate 500-ms capture-age freshness rule and newest-capture ordering. The journal updates `map_state_latest.json` and appends `installed_map_records.jsonl`; no captured records are included here. Websocket presentation and later alert integration are not claimed as completed by this handoff.

The radio configuration is 100 MHz, 273 PRB, 4 downlink/5 uplink slots. GOOD_ALL currently commands 21.5 dB through the RFsim mapping; it must not be labelled as achieved PUSCH SNR. The separate PUSCH power-control target in the included configuration is 15 dB. Radio retuning has not been applied.

### Data structures useful for slides

- `FrameContextV1`: capture/frame identity, pose and camera calibration identity; [frame_context.py](code/rl_agent/splitfusion_live_dispatch_v1/frame_context.py).
- `Run5ActorDecision`: selected `mode_id`, `q_e4`, support bounds and identity digests; [actor_binding.py](code/rl_agent/splitfusion_run5_live_avo_v1/actor_binding.py).
- `ExecutedActionIdentity` and the wire envelope: bind actual executed mode/q to the decision/hold frame; [transaction_identity.py](code/rl_agent/splitfusion_hybrid_sac_v1/transaction_identity.py), [continuous_execution_v2.py](code/rl_agent/splitfusion_hybrid_sac_live_route_b_v2/continuous_execution_v2.py).
- Prediction record: class, score, world/local xyz, size, yaw, image box and source identity; [infer.py FIELDS and record](code/fcos_model/network/infer.py).
- Map update: stream/frame/capture time, executed action identity, records, segmentation and edge timings; `build_run4_map_update` in the protocol above. Map classes become `Vehicle` and `Pedestrian` from the model's own prediction class, not GT.

## Contents and limitations

No captures, sensor tensors, GT arrays, quality database, bulk training checkpoints, or collected map records are included. Only the explicitly requested training curves and their CSVs are included as results. The small deployed actor weight file is included; large perception weights are references only.

The local OAI telemetry/scheduler modifications are under `code/OAI/`; the complete OAI/core-network tree, CARLA installation and container images are external. See [EXTERNAL_DEPENDENCIES.json](EXTERNAL_DEPENDENCIES.json).

[SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) records each copied file's original path, current snapshot path, size and SHA-256, including the model-folder rename mapping. The snapshot includes uncommitted work; a repository commit alone does not identify this code. The folder renames are for navigation only: source contents remain byte-for-byte unchanged, so Python imports and original path constants still use their original repository names. They identify dependencies, not a different deployed model. This is not a runnable package refactor; do not launch experiments from this folder.
