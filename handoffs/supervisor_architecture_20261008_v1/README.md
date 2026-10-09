# SplitFusion FCOS, agent and full Town10 map pipeline

For Subhramoy, from Abiodun. Prepared on 8 October 2026; updated on **9 October 2026** with the working end-to-end system. This is a code-reading handoff for architecture slides and inspection of the implemented data structures.

This folder contains the FCOS perception model, compression modules, the deployed Run-5 agent and its training implementation, and the full Town10 map pipeline through OAI to spatial-map ingestion and persisted JSON. The repository calls the predefined full-map loop `Route B` on `Town10HD_Opt`; that internal identifier remains in source filenames. Use **full Town10 map** in presentation labels. Start with the links below; the remaining source files support these entry points. The model folders have been shortened and renamed for clarity; other repository-relative paths are preserved under `code/`.

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

**Status:** the end-to-end full Town10 map loop has now completed and passed the registration-bound integration checks in both **GOOD_ALL (V11)** and **BAD_ALL (V12)**. The deployed agent remains frozen seed 43 at update 10,000. V12 uses the same qualified pipeline as V11 with a different channel schedule. Integration success does not mean that the poor-channel timeliness target or robustness across all conditions has been achieved. Good-to-bad and bad-to-good full-map evaluations remain pending.

This is still a code-reading snapshot, not a standalone runnable release or permission to launch an experiment. The original V6 handoff and then-pending repair note are preserved under `history/20261008_v6/`; they describe the old state, not the current implementation. No collected data or new experiment plots were added. The requested deployed-agent training plots are retained unchanged.

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

Both plots are copied unchanged from the deployed actor's training campaign, with seeds 17, 29 and 43. Their small CSVs are included. They are modeled-training curves, not full Town10 map live performance plots. Later rejected reward/backlog retraining candidates are not the deployed model.

## End-to-end full Town10 map pipeline

```text
CARLA RGB and radar + ego pose
          │
          ├─ causal scene/radio/history state → frozen Run-5 actor → mode, q
          │
          └─ corrected 7-channel input → UE FCOS front → C2
                → optional AE, spatial selection, quantization and packing
                → OAI UE uplink → gNB/core network → edge reassembly
                → reconstruction/AE decoder → FCOS tail and postprocessing
                       ├─ direct map ingest → installed map state → map ACK to UE
                       │                         └─ background retrieval → persisted JSON
                       └─ exact-GT scorer → evaluation feedback to UE
CARLA privileged evaluation inputs → parallel GT preparation ────────┘
```

| Pipeline component | First files to inspect |
|---|---|
| Validated good-channel launcher and process roles | [V11 host](code/rl_agent/splitfusion_m3_route_b_v11/host.py), [runner](code/rl_agent/splitfusion_m3_route_b_v11/runner.py), [entries](code/rl_agent/splitfusion_m3_route_b_v11/entries.py), [contract](code/rl_agent/splitfusion_m3_route_b_v11/contract.py) |
| Validated bad-channel variant of the same pipeline | [V12 host](code/rl_agent/splitfusion_m3_route_b_v12/host.py), [entries](code/rl_agent/splitfusion_m3_route_b_v12/entries.py), [contract](code/rl_agent/splitfusion_m3_route_b_v12/contract.py) |
| Frozen actor integration | [runtime_policy.py](code/rl_agent/splitfusion_run5_full_live_v1/runtime_policy.py) |
| UE front and edge dispatch | [ue_runtime.py](code/rl_agent/splitfusion_live_dispatch_v1/ue_runtime.py), [edge_runtime.py](code/rl_agent/splitfusion_live_dispatch_v1/edge_runtime.py) |
| Continuous action identity and map message | [run4_map_protocol_v2.py](code/rl_agent/splitfusion_hybrid_sac_live_route_b_v2/run4_map_protocol_v2.py) — this retained protocol is also used by Run-5 |
| Direct map ingestion and state | [map_ingest.py](code/rl_agent/splitfusion_direct_edge_map_v1/map_ingest.py), [spatial_map_direct_server_v1.py](code/rl_agent/splitfusion_direct_edge_map_v1/spatial_map_direct_server_v1.py) |
| Correct map classes and persisted JSON | [map_class.py](code/rl_agent/splitfusion_m3_route_b_v3/map_class.py), [map_journal.py](code/rl_agent/splitfusion_m3_route_b_v3/map_journal.py) |
| Concurrent evaluator | [service.py](code/rl_agent/splitfusion_exact_gt_service_proto_v2/service.py), [prep_core.py](code/rl_agent/splitfusion_exact_gt_service_proto_v2/prep_core.py), [scorer_core.py](code/rl_agent/splitfusion_exact_gt_service_proto_v2/scorer_core.py) |
| Active GT worker composition | [V11 worker entry](code/rl_agent/splitfusion_m3_route_b_v11/worker_entry.py), [V12 worker entry](code/rl_agent/splitfusion_m3_route_b_v12/worker_entry.py) |
| Full-map completion and final-frame drain | [route completion](code/rl_agent/splitfusion_m3_route_b_v3/route_completion.py), [bounded real-ACK drain](code/rl_agent/splitfusion_m3_route_b_v11/route_drain.py) |
| Planned full Town10 map loop | [loop definition](code/data_collection/routes/town10hd_opt_route_b_full_map_loop_v1.json), [loop runner](code/data_collection/run_route_b_ego_loop.py) |
| Channel control and measured telemetry | [GOOD_ALL controller](code/rl_agent/splitfusion_m3_route_b_v11/good_all_controller.py), [BAD_ALL controller](code/rl_agent/splitfusion_m3_route_b_v12/bad_all_controller.py), [radio telemetry](code/rl_agent/splitfusion_m3_route_b_v11/radio_telemetry.py), [actual gNB configuration](radio_config/effective_gnb_100mhz_4d5u.conf) |
| Registration-bound offline assessment | [V11 postflight](code/rl_agent/splitfusion_m3_route_b_v11_offline_v1/postflight.py), [V12 postflight](code/rl_agent/splitfusion_m3_route_b_v12_offline_v1/postflight.py) |

Map publication is independent of the 170-ms evaluation-feedback result. A prediction completed at 175 ms is not discarded merely for exceeding that feedback deadline: it can still update the map if it satisfies the separate 500-ms capture-age freshness rule and newest-capture ordering. The map-install ACK and GT-based quality feedback are distinct messages; do not combine them into one feedback arrow in the architecture slide. Privileged CARLA GT is used for evaluation, not as an actor input or a replacement for model predictions.

The map retrieval worker journals the already-installed records: `installed_map_records.jsonl` accumulates the updates, and `map_state_latest.json` is atomically replaced with the latest installed stream record. These are model-output records with `Vehicle`/`Pedestrian` types, not GT. JSON persistence is downstream of map installation and is not synchronous work on the actor/deadline path. No captured records are included here. Websocket presentation, alert integration and autonomous UE selection remain later work, not completed features claimed by this handoff.

The full-map scenario uses the **50-vehicle / 50-pedestrian traffic setting**, with the registered seeds, replenishment and permitted roadblock handling. It is not an ego-only scene. The loop has a 3,900-frame safety ceiling rather than a 1,000-frame experiment budget; normal completion is reaching the end of the planned loop.

### Repairs now part of the working pipeline

These are runtime/evaluator changes, not changes to the frozen perception model or agent:

- **Process-role bindings:** each launcher, UE child, edge process and GT worker resolves the intended registration and source checks; see the V11/V12 entries and contracts above. Older version folders remain because current code deliberately reuses some of their implementations.
- **Pose consistency:** [pose_tolerance.py](code/rl_agent/splitfusion_m3_route_b_v3/pose_tolerance.py) applies the registered float32-aware translation check at larger world coordinates. The offline tools use the same P3 rule as the live workers.
- **Preparation exception handling:** [FREEZE lifetime repair](code/rl_agent/splitfusion_gt_freeze_lifetime_v1/repair.py) preserves the original preparation error and a bounded failure snapshot instead of masking it with a mapped-buffer cleanup error.
- **Tilt-aware pedestrian visibility:** [AVO geometry](code/rl_agent/splitfusion_avo_tilt_v1/geometry.py) and [quality adapter](code/rl_agent/splitfusion_avo_tilt_v1/quality.py) handle supported tilted pedestrians using the frozen actor transform. Upright parity and the existing visibility threshold are retained; this is a versioned evaluator change. Unsupported geometry still fails closed.
- **Exact storage accounting V3:** [accounting.py](code/rl_agent/splitfusion_gt_storage_accounting_v3/accounting.py) and [switch.py](code/rl_agent/splitfusion_gt_storage_accounting_v3/switch.py) avoid repeatedly walking unchanged GT history while preserving the original byte count and capacity checks. The checked, bounded fast path is installed only in the preparation worker. History still governs static-object duplicate eligibility; it is not the actor's history or the map's history.
- **End-of-loop drain:** the collector waits for actual outstanding map ACKs within their existing horizons before shutdown. It does not extend the 170-ms evaluation deadline, invent ACKs or omit the final transmitted frame.
- **Assessment binding:** the authoritative postflight reads the current registration's channel and pose rules. The retained older V3 tool is not the current authority; its historical constants must not be mistaken for the V11/V12 schedule.

### Radio quantities and the two completed conditions

Both use 100 MHz, 273 PRB, and the registered 4-downlink/5-uplink-slot configuration. MCS is scheduler-selected, not forced.

| Full-map condition | Commanded SNR target supplied causally to the actor | RFsim noise command | Controller |
|---|---:|---:|---|
| GOOD_ALL | 23.5 dB | −12.0 dB | V11 |
| BAD_ALL | 8.5 dB | −4.0 dB | V12 |

These constant schedules are separate from the older stochastic `FAVORABLE_STABLE` / `ADVERSE_STABLE` profiles. The calibrated per-transmission PUSCH `snrx10` and scheduler `avg_snr` are recorded for analysis. The MAC log's `avg_snr + tpc_in_flight` is a power-control statistic, **not the channel SNR**; the separate power-control target remains 15 dB. Neither gNB-measured series enters the actor: its SNR feature remains the acknowledged commanded target. Some copied controller docstrings retain descriptions of inherited earlier constants; the current contract and registered schedule are authoritative.

[RUNTIME_OVERVIEW.json](RUNTIME_OVERVIEW.json) contains the selected GOOD_ALL/V11 registration fields. [RUNTIME_OVERVIEW_BAD_ALL.json](RUNTIME_OVERVIEW_BAD_ALL.json) contains the BAD_ALL/V12 variant. Each is a verbatim per-run configuration excerpt, not a claim about all subsequent conditions.

### Data structures useful for slides

- `FrameContextV1`: capture/frame identity, pose and camera calibration identity; [frame_context.py](code/rl_agent/splitfusion_live_dispatch_v1/frame_context.py).
- `Run5ActorDecision`: selected `mode_id`, `q_e4`, support bounds and identity digests; [actor_binding.py](code/rl_agent/splitfusion_run5_live_avo_v1/actor_binding.py).
- `ExecutedActionIdentity` and the wire envelope: bind actual executed mode/q to the decision/hold frame; [transaction_identity.py](code/rl_agent/splitfusion_hybrid_sac_v1/transaction_identity.py), [continuous_execution_v2.py](code/rl_agent/splitfusion_hybrid_sac_live_route_b_v2/continuous_execution_v2.py).
- Prediction record: class, score, world/local xyz, size, yaw, image box and source identity; [infer.py FIELDS and record](code/fcos_model/network/infer.py).
- Map update: stream/frame/capture time, executed action identity, records, segmentation and edge timings; `build_run4_map_update` in the protocol above. Map classes become `Vehicle` and `Pedestrian` from the model's own prediction class, not GT.

## Contents and limitations

No captures, sensor tensors, GT arrays, quality database, bulk training checkpoints, or collected map records are included. Only the explicitly requested training curves and their CSVs are included as plots/results. The small deployed actor weight file is included; large perception weights are references only. [REFRESH_RECORD.json](REFRESH_RECORD.json) records the source update and pointers to the authoritative integration evidence without copying that evidence into this folder. The concurrent 12-mode benchmark is not a new deployed agent or a replacement full-map runtime and is outside this update.

The local OAI telemetry/scheduler modifications are under `code/OAI/`; the complete OAI/core-network tree, CARLA installation and container images are external. See [EXTERNAL_DEPENDENCIES.json](EXTERNAL_DEPENDENCIES.json).

[SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) records each copied file's original path, current snapshot path, size and SHA-256, including the model-folder rename mapping. [HANDOFF_SHA256.json](HANDOFF_SHA256.json) covers the full handoff, including this README. The snapshot includes uncommitted work; a repository commit alone does not identify this code. The folder renames are for navigation only: source contents remain byte-for-byte unchanged, so Python imports and original path constants still use their original repository names. They identify dependencies, not a different deployed model. This is not a runnable package refactor; do not launch experiments from this folder.

The original V6 builder is preserved under `history/20261008_v6/build_snapshot.py`. The refresh/verification script is [refresh_snapshot.py](refresh_snapshot.py); in the original workspace, `python3 -B refresh_snapshot.py --verify` checks copied source hashes, registered source coverage, Python syntax and README links without importing the runtime or starting services. Its one-time update mode refuses to overwrite an already refreshed handoff. No production source, launch registration or running experiment is changed by this handoff refresh.
