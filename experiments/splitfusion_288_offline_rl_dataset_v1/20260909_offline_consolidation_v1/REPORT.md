# SplitFusion 288-Cell Live CARLA/OAI Campaign - Offline RL Dataset Consolidation

Steps 1-4 only: evidence verification, one reconciled 288-cell table, the
72-action x 4-profile aggregation, and separation of the latency, delivery,
payload and registered-accuracy components. **No action pruning, Pareto
selection, feasibility masking, reward design, policy design or training is
performed or implied here.** Steps 5-6 belong to a separate reviewer.

This consolidation is strictly offline. No CARLA server, Docker container, OAI
gNB/UE, RFsim, CUDA context or model inference was started, and no
training/holdout/validation/test imagery was read. No campaign directory was
modified. **No claim of 100 ms service readiness is made.**

## 1. Provenance

- Evidence commit: `19c867623f26caca90d0ffcad319f7e68c625e74`
- Starting HEAD: `19c867623f26caca90d0ffcad319f7e68c625e74`
- Campaign root: `experiments/splitfusion_288_live_campaign_v1/20260907_full_288_retry8`
- Completion terminal: `SPLITFUSION_288_CELL_LIVE_CARLA_OAI_CAMPAIGN_COMPLETE`

| artifact | sha256 |
| --- | --- |
| `SPLITFUSION_288_CELL_LIVE_CARLA_OAI_CAMPAIGN_COMPLETE` | `87f8320aa07d85d142af8447c296944885893bedfd67912315350a518ebd7f7c` |
| `campaign_completion.json` | `696af4443635bda2da8d93a2b8fd89808cd5ef04dfe34d30ed7f32a248ee1ee8` |
| `campaign_continuation_binding.json` | `71f969c6e50d5d508d3b0f139354a59671384936c4464a18a246eea37b28007f` |
| `campaign_ledger.json` | `dbbf3588fcaf542a6e3b2304a87267b58469c2dbd46bb1a9818cc2e82bb9d7f5` |
| `campaign_manifest.json` | `dd2b5ce4326284888db6e6e28beae0863f79002d96964bc0d65321d48c3e326b` |
| `rl_agent/configs/ue_288_campaign_v1.yaml` | `f9c8382af5a4b16e95f71fa0e273437ee54eb114b5c39d3d428335a71c9e9057` |
| `rl_agent/splitfusion_action_catalog_v1/splitfusion_72_action_catalog.json` | `07e0690f8a55bdd6068b8b283d14b7e165ccbf44742dd0a9568cfdd5dcac54c3` |

All four declared completion bindings verified, and the cell-mapping digest
was **re-derived independently** from the locked 72-action catalog and the four
configured network profiles rather than trusted from the manifest:

- `cell_mapping_sha256` re-derived = `add96c8f117c4cb8ba427c670ba1698d4a2b8aea84129b4d32ac47fdc56025b1`

## 2. Continuation chain

The retry8 inventory reuses 61 cells by hash reference.
All 61 are references *carried forward*
(0 were re-derived from the immediate source),
so the chain had to be walked to its origins rather than assumed:

| campaign | ledger sha256 | statuses | rerun |
| --- | --- | --- | --- |
| `...retry8` (this) | `dbbf3588fcaf542a...` | {'PASSED': 227, 'REUSED': 61} | `a61__favorable_stable` |
| `...retry7` | `9909179b8b7c3b55...` | {'REUSED': 61, 'FAILED': 1} | ['a61__favorable_stable'] |
| `...retry6` | `831e428768600be6...` | {'REUSED': 58, 'PASSED': 3, 'FAILED': 1} | ['a58__favorable_stable'] |
| `...retry5` | `7c527122f9df911f...` | {'PASSED': 58, 'FAILED': 1} | None |

Origin accounting: 58 reused cells originate in retry5 and 3 in retry6, matching
the completion summary exactly. Every reused cell's continuation reference, source
campaign, source ledger/manifest/terminal binding, original attempt directory and
every registered file hash were verified against the physical source tree.

### Route-summary limitation (stated precisely)

Two distinct things must not be conflated, so both are reported:

- **Durable route summary file present:** 227 executed + 3 retry6-origin reused = 230/288. Absent for the 58 retry5-origin reused cells, which predate the requirement.
- **Registered `route_outcome_classification` present:** 227/288 - the executed cells only.

All 61 reused cells therefore lack the registered classification. The 3
retry6-origin cells do have a durable route summary, but it is the route
*runner's* metrics summary whose `status` field is the density/intervention
state (e.g. `INTERVENED`), not the campaign's `route_outcome_classification`.
Promoting that status to a classification would be an inference, so it was
not done: those cells are `UNAVAILABLE_NOT_INFERRED`, and the raw runner
status is carried separately in
`route_runner_status_raw_not_a_classification`. **No route outcome was
inferred for any reused cell.**

Registered route outcomes over the 227 cells that have one:

- `COLLISION_ONLY_MEASURED_OUTCOME`: 2
- `COMPLETED_NO_BRAKING_EVENT`: 5
- `PERMITTED_INTERVENTION`: 220

## 3. Reconciliation identities

- Expected cells: 288; ledger cells: 288; unique: 288
- Missing: 0; foreign: 0; duplicated: 0
- Actions: 72; network profiles: 4; product: 72 x 4 = 288
- Executed: 227; reused: 61
- Cells with structural acceptance PASS: 288/288
- Registered output hashes verified: 1958
- Cells whose campaign counter reconciliation holds: 288/288
- Cells whose reassembly identity (reassembled = admitted + after-reassembly drops + rejected) was independently re-checked and holds: 288/288
- Cells whose recomputed install-AoI median/p95 reproduce the registered values exactly: 288/288

## 4. Workload and preparation

- Registered preparation target: 10 Hz; minimum coverage contract 0.95.
- Sensor preparation coverage across 288 cells: median 0.9776, min 0.9133, max 0.9932.
- Cells meeting the registered 0.95 coverage contract: 280/288.
  The 8 cells below it span 0.9133-0.9495 and are all large-payload rungs. They still carry structural acceptance PASS because the
  campaign registers low preparation as a measured outcome, not a structural
  invalidity. The 0.95 target was not weakened here, and these cells are
  retained rather than excluded:
  - `a00__fade_recovery`: 0.9312
  - `a00__favorable_stable`: 0.9454
  - `a06__adverse_stable`: 0.9441
  - `a06__fade_recovery`: 0.9391
  - `a06__favorable_stable`: 0.9133
  - `a06__mid_variable`: 0.9417
  - `a12__favorable_stable`: 0.9495
  - `a18__fade_recovery`: 0.9476
- Total frames sent across the campaign: 896,856.
- Sustainable send rate from the median inter-send period: median 10.221 FPS.

Preparation drops by registered reason (campaign totals):

- `SENT`: 896,856
- `DROPPED_REPLACED_BY_NEWER_FRAME`: 21,726
- `DROPPED_INCOMPLETE_RADAR_WINDOW`: 288
- `WARMUP_NO_COMPLETE_RADAR_WINDOW`: 288
- `STALE_BEFORE_SEND`: 106
- `DROPPED_SENSOR_LATE_OR_MISSING`: 39
- `SPLIT_PROCESSING_FAILED`: 0

## 5. Payload

Live measured bytes are kept strictly separate from catalog estimates; catalog
values live in `catalog_*` columns and were never substituted for measured bytes.

- Live scientific inner payload, median over cells: 6,196 B (min) to 3,584,999 B (max).
- SFD1/application-envelope overhead, median over cells: 205.0 B.
- Datagrams per message, median over cells: 1 to 288.

## 6. Delivery

- Cells with at least one installed map: 222/288.
- **Cells with zero delivery: 66/288.** Zero delivery is preserved as a measured outcome, never as missing data.
- Campaign totals: 896,856 sent -> 712,732 complete edge reassemblies -> 366,225 tail completions -> 334,174 map publications -> 334,174 maps installed.
- Feature datagrams transmitted 44,754,702; received at the edge 28,038,030 (ratio 0.6265).

### The two deadlines are reported separately

- Installed within the **100 ms** service reference: **0** frames (0.000000% of sent, 0.000000% of installed).
- Feedback within the **500 ms** ACK timeout: **329,393** frames (36.7275% of sent, 98.5693% of installed).
- Terminal `TIMEOUT_NO_ACK`: 564,580.
- Cells with at least one install inside 100 ms: 0/288.

The 500 ms ACK observation timeout is **not** the 100 ms service target, and the
registered `*_fraction` fields use *installed frames* as their denominator; both
the per-sent and per-installed denominators are retained in the tables.

Across all 288 cells, **not one** of the 896,856 sent frames installed inside the
100 ms service reference. This is a measured campaign outcome and is exactly why
the campaign records `claims_100ms_service_ready: false`; it is reported here
without any readiness claim.

### Per network profile

| profile | sent | installed | installed/sent | within 100 ms | ACK<=500 ms | zero-delivery cells | datagram rx/tx | install AoI median-of-medians |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `FAVORABLE_STABLE` | 223,662 | 100,893 | 0.4511 | 0 | 99,837 | 8/72 | 0.7482 | 344.5 ms |
| `MID_VARIABLE` | 226,716 | 85,861 | 0.3787 | 0 | 84,884 | 16/72 | 0.6379 | 358.6 ms |
| `ADVERSE_STABLE` | 223,753 | 60,454 | 0.2702 | 0 | 58,713 | 29/72 | 0.4525 | 399.7 ms |
| `FADE_RECOVERY` | 222,725 | 86,966 | 0.3905 | 0 | 85,959 | 13/72 | 0.6682 | 353.9 ms |

The ordering is monotone in channel favourability for every delivery measure,
and adverse/fade behaviour is retained per profile in all three tables rather
than being averaged away.

## 7. Latency decomposition

Clock domains were established from the emitting source, not assumed:

- `WALL_HOST` - `time.time()`/CLOCK_REALTIME. The UE CARLA client, the edge
  container (local Docker on the OAI CN5G bridge) and the map process share one
  kernel, so these instants are mutually comparable.
- `UE_PERF` - `time.perf_counter_ns()` in the single UE client process (adapter
  and live dispatch are the same process). Mutually comparable, **not**
  epoch-anchored.
- `EDGE_PERF` - edge-local `perf_counter_ns()`; contributes durations only.

Two naming traps in the registered schema, both preserved rather than papered over:

1. `front_ms` is **not** the model front. It is `ue_prepare_finished_ns -
   capture_started_ns`, the whole UE dispatch span (input assembly, context
   build, front, ranker, AE encode, quantize/pack, zstd, chunking). The model
   front proper is `front_timing_ns.front_backbone`. Pre-front sensor
   preparation is carried in its own `s1_*` fields and is never called
   `front_ms`.
2. `queue_wait_ms` is **not** a pure prepared-queue wait. It is measured from
   the schedule instant to just before dispatch and therefore *encloses* the
   sensor wait and radar/RGB preparation. A same-clock residual is offered
   separately as the true queue wait.

- **Stage 5, one-way feature uplink, is UNAVAILABLE.** The final UE send boundary
  (`send_finished_ns`, UE_PERF) and the complete edge receive boundary
  (`edge_receipt_wall_s`, WALL_HOST) are in non-comparable clock domains and no
  simultaneous cross-clock reading is registered. No one-way uplink latency was
  manufactured by differencing them or by subtracting unrelated medians. A
  round-trip residual and a wall-domain capture-to-edge-receipt span are provided
  instead, each explicitly labelled.
- **Stage 9, tail completion to map publication, is UNAVAILABLE**: map publication
  is registered only as the categorical `map_publication_status`, so the boundary
  has no end event.

Campaign-level medians of the per-cell medians (ms):

| stage | cells with data | median of per-cell medians |
| --- | --- | --- |
| `analytic_model_front_transport_back_feedback_ms` | 222 | 276.512 |
| `s10_ue_result_receipt_to_map_installation_ms` | 222 | 3.305 |
| `s11_feedback_emit_to_ue_receipt_ms` | 222 | 0.089 |
| `s11_map_install_to_ue_feedback_receipt_ms` | 222 | 0.149 |
| `s12_ue_send_to_compact_result_receipt_ms_UE_PERF` | 222 | 284.101 |
| `s13_capture_to_install_aoi_ms` | 222 | 364.544 |
| `s14_capture_to_ue_feedback_aoi_ms` | 288 | 500.646 |
| `s1_pre_front_compute_ms` | 288 | 33.783 |
| `s1_radar_prepare_ms` | 288 | 23.473 |
| `s1_radar_window_ms` | 288 | 5.598 |
| `s1_rgb_convert_ms` | 288 | 2.920 |
| `s1_scene_snapshot_ms` | 288 | 0.116 |
| `s1_sensor_wait_ms` | 288 | 9.318 |
| `s2_derived_true_queue_wait_ms_RESIDUAL` | 288 | 0.183 |
| `s2_queue_depth_at_dispatch` | 288 | 0.000 |
| `s2_recorded_queue_wait_ms_ENCLOSES_SENSOR_PREP` | 288 | 49.622 |
| `s3_ae_encode_ms` | 288 | 0.274 |
| `s3_front_backbone_ms` | 288 | 1.345 |
| `s3_front_ms_FULL_UE_DISPATCH_SPAN` | 288 | 24.817 |
| `s3_front_timing_ns_unparseable_rows` | 0 | n/a |
| `s3_quantize_pack_ms` | 288 | 8.657 |
| `s3_ranker_selection_ms` | 288 | 3.652 |
| `s3_total_ue_preparation_ms` | 288 | 16.420 |
| `s3_zstd_compression_ms` | 288 | 0.428 |
| `s4_datagram_send_loop_ms` | 288 | 0.245 |
| `s4_derived_ue_uninstrumented_span_ms` | 288 | 7.330 |
| `s5_capture_to_edge_receipt_ms_INCLUDES_SENSOR_AND_FRONT` | 222 | 129.278 |
| `s5_derived_roundtrip_residual_ms_RESIDUAL` | 222 | 129.234 |
| `s5_feature_uplink_one_way_ms` | 0 | n/a |
| `s6_derived_edge_queue_wait_ms_RESIDUAL` | 222 | 40.606 |
| `s6_edge_receipt_to_tail_complete_ms` | 222 | 190.787 |
| `s7_ae_decode_ms` | 222 | 1.079 |
| `s7_unpack_dequantize_ms` | 222 | 5.725 |
| `s7_zstd_decompression_ms` | 222 | 0.183 |
| `s8_frozen_tail_ms` | 222 | 111.634 |
| `s8_output_serialization_ms` | 222 | 19.630 |
| `s8_total_edge_processing_ms` | 222 | 144.669 |
| `s9_tail_complete_to_edge_evidence_install_ms` | 222 | 2.702 |
| `s9_tail_complete_to_map_publication_ms` | 0 | n/a |
| `s9_tail_complete_to_ue_result_receipt_ms` | 222 | 11.554 |

`s3_front_timing_ns_unparseable_rows` is a parsing diagnostic, not a latency
stage: it shows 0, i.e. every instrumented timing blob parsed. The two `n/a`
stages are the two genuinely unavailable boundaries named above.

- Registered capture-to-install AoI (`install_aoi_ms`), median of per-cell medians: **364.5 ms** over the 222 cells that installed anything.
- `s14_capture_to_ue_feedback_aoi_ms` sits near 500 ms because it includes every
  `TIMEOUT_NO_ACK` row, whose receipt instant is the 500 ms timeout sweep rather
  than a remote ACK. It must not be read as a network latency.
- Every published map was installed (`survival_installed_per_published` = 1.0 wherever
  publication occurred); the loss is upstream, in datagram delivery, reassembly and
  the edge deadline gates.

The per-frame component medians are **not** expected to sum to the median
end-to-end latency: medians do not add, the stages have different valid-sample
sets, and two stages are unavailable. Any residual field is labelled
`_RESIDUAL` and carries its own negative-sample count.

### Residual sign check (clock-domain sanity)

Every derived residual is formed by subtracting a same-frame duration from a
same-frame interval in a *compatible* domain. If a residual had in fact crossed
incomparable clocks, negative values would be expected. Observed:

| residual | samples | negative | worst minimum |
| --- | --- | --- | --- |
| `s2_derived_true_queue_wait_ms_RESIDUAL` | 896,856 | 0 | 0.0638 ms |
| `s4_derived_ue_uninstrumented_span_ms` | 896,856 | 0 | 1.8854 ms |
| `s5_derived_roundtrip_residual_ms_RESIDUAL` | 344,177 | 0 | 19.0729 ms |
| `s6_derived_edge_queue_wait_ms_RESIDUAL` | 344,177 | 0 | 0.0852 ms |

All four residuals are strictly positive over every sample, which is consistent
with the domain assignments above. It is corroboration, not proof.

The user's later analytical quantity, `model_front + feature_transport +
model_back + compact_feedback`, is preserved as
`analytic_model_front_transport_back_feedback_ms`. It is computed only where
every component is a same-frame compatible timestamp, and it is **not** labelled
capture-to-install latency: it excludes sensor preparation, the prepared-queue
wait and UE dispatch overhead, and its transport term is a round-trip residual.

## 8. Accuracy and preservation join

All 72 live actions were joined to their authoritative frozen validation
properties using existing registered fields and classifications only. **No
prediction was rescored and no threshold was tuned.** Model-validation metrics
are carried in `val_*` columns and are kept distinct from the live network
measurements; live per-route perception is separately namespaced `live_*`.

- Validation rows joined: 288/288 cells (72/72 actions).
- Source evidence, all hash-verified:
  - `experiments/splitfusion_fcos_ae_v1/20260903_phase10b_ae32_uint8_validation/phase10b_ae32_uint8_validation.json` (6 actions) sha256 `db4b944eb5992a82e9fe6b0befd2e3bcf583629a6f27ad2ad6cb4075f03a90ec`
  - `experiments/splitfusion_fcos_ae_v1/20260903_phase10b_ae64_uint8_validation/phase10b_ae64_uint8_validation.json` (6 actions) sha256 `eead1786a5d12294b9d61d9271431049aac28540a20f2c4608db33ab66de3aad`
  - `experiments/splitfusion_fcos_ae_v1/20260903_phase11d_lowbit_validation/phase11d_lowbit_validation.json` (48 actions) sha256 `2680e6dc21469e6fdcabe1ce79e9d2d333d9a137c639cd033338b9b1c01c2862`
  - `experiments/splitfusion_fcos_ae_v1/20260903_phase9d_ae128_uint8_validation/phase9d_ae128_uint8_validation.json` (6 actions) sha256 `89cc7c706fc3383106a5680d3d54d5fb514dcd5d808c13d9eaf1a2c380785963`
  - `experiments/splitfusion_fcos_hybrid_q_v1/20260902_223610_phase8b_uint8_validation/phase8b_uint8_validation.json` (6 actions) sha256 `a2779f5fb0a585b1c317dc755b5ab577fa7c34963ab7945cb704e0d4146bb029`
- Durable per-action settings hash-verified: 72/72.

Registered tiers across the 72 actions:

- `EMERGENCY_ONLY`: 28
- `FULL_PRESERVATION`: 32
- `emergency-mode anchor`: 2
- `primary anchor`: 4
- `primary profile`: 4
- `stress/emergency profile`: 2

Left deliberately null:

- `person_avo_recall_0_30m` and `person_avo_recall_30_40m_diagnostic` are
  registered only for the 48 Phase-11D UINT6/UINT4 actions. The 24 historical
  UINT8 actions (phase8b noAE, phase9d AE128, phase10b AE32/AE64) have no
  compatible per-range evidence, so those cells stay null. **0-30 m recall was
  not reconstructed from incompatible historical aggregates.**
- A pixel mask-accuracy field is not registered anywhere in the frozen evidence;
  `foreground_miou` and `person_box_mask_iou` are reported instead.
- No standalone segmentation gates-passed/total counter exists; segmentation is
  registered as `segmentation_installable` plus `segmentation_behavior`.

## 9. Outputs

| artifact | rows | sha256 |
| --- | --- | --- |
| `campaign_288_cell_table.csv` | 288 | `3f3067d4c9d0ef3c0d2661c3d19d04306bcbd18ef6d21fa426148e0ee24d2e0d` |
| `action_72x4_summary.csv` | 288 | `280b372ed8b6a52bb3e5f1f69e6cbdc02dc851a6a597565fcab9ca9652f5998e` |
| `action_72_summary.csv` | 72 | `250eb6d9391c0de5a343d692484647bbdb181153ba64f7f4a447b0076f580a88` |
| `latency_metric_dictionary.json` | n/a | `92439634dc648ef9aabe8b98ee45a0b430356bc068b1d11c90f195835e040c29` |
| `metric_availability.json` | n/a | `181dc89b860813f3dc4741540297e2247987c81a204e2564d155cc6a894c3f3e` |
| `analysis_summary.json` | n/a | `48358c2f27ffc2f3da8bf0f821c733914a03267f50da7adab4cbf70cbe239690` |

This report cannot carry its own digest. `REPORT.md` and every artifact above
are bound together by the compact terminal marker
`SPLITFUSION_288_OFFLINE_RL_DATASET_CONSOLIDATION_COMPLETE`.

`campaign_288_cell_table.csv` and `action_72x4_summary.csv` are both keyed by
the same 288 action/profile cells but are **not** structurally duplicated:

- `campaign_288_cell_table.csv` is the reconciled *cell record*: identity,
  provenance and continuation binding, workload/preparation, the separated
  payload components, and the delivery-stage counts and rates.
- `action_72x4_summary.csv` is the *scientific component table*: the full
  per-stage latency decomposition (field-level distributions with clock domain,
  valid-sample and exclusion accounting) joined to the frozen registered
  accuracy properties. It carries the Step-4 separation that later RL
  formulation consumes.
- `action_72_summary.csv` aggregates each action across the four profiles while
  retaining per-profile columns, so adverse and fade behaviour is never hidden
  inside a mean.

## 10. Scope

Delivered: Steps 1-4. Not performed, and deliberately left to the Step 5-6
reviewer: action pruning, Pareto decisions, feasibility masking, reward design,
RL-policy design, and training.
