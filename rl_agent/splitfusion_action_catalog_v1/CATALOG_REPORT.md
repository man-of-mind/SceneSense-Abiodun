# SplitFusion 72-profile action catalog

Frozen evidence base: `ba681dbd33514d80d96784ac4cd0d67f3151e800`.

The catalog locks 72 enabled SPLIT actions in family → quantizer → q order. All entries use CURRENT_CELL_MAJOR, mandatory zstd level 1, and the frozen p025 perception path. The builder reconciled 24 UINT8 and 48 low-bit durable setting records to their summaries without loading models or accessing frames.

FULL_PRESERVATION is only the relative 12-gate result; it does not imply 9/9 absolute service readiness. Localization and segmentation are independent capabilities. q=0.90 and q=0.98 remain stress/emergency actions. Perception degradation does not disable a technically valid action, and STATE_INFEASIBLE remains a dynamic runtime state.

When `segmentation_installable` is true, install current segmentation. Otherwise retain the prior segmentation layer with its original timestamp. The 0–30 m boundary is evaluation-only and never filters, suppresses, relabels, or rescores runtime detections.

## Inventory

- Profiles: 72 (all transport-valid and agent-enabled)
- Stress/emergency profiles: 24
- Durable settings reconciled: 72 (24 UINT8 + 48 UINT6/UINT4)
- Phase-11D frame-settings bound: 160,560

## By family

| family | profiles | relative | service | localization | segmentation | stress | zstd median range (bytes) |
|---|---:|---:|---:|---:|---:|---:|---:|
| noAE | 18 | 18 | 0 | 2 | 6 | 6 | 34811.0–3568326.0 |
| AE128 | 18 | 13 | 0 | 0 | 9 | 6 | 23346.0–2345384.0 |
| AE64 | 18 | 11 | 0 | 0 | 8 | 6 | 12095.0–1216049.0 |
| AE32 | 18 | 12 | 0 | 0 | 0 | 6 | 6229.0–602242.0 |

## By quantizer

| quantizer | profiles | relative | service | localization | segmentation | stress | zstd median range (bytes) |
|---|---:|---:|---:|---:|---:|---:|---:|
| UINT8 | 24 | 7 | 0 | 0 | 8 | 8 | 13246.0–3568326.0 |
| UINT6 | 24 | 24 | 0 | 1 | 8 | 8 | 10555.0–2734349.0 |
| UINT4 | 24 | 23 | 0 | 1 | 7 | 8 | 6229.0–1436103.0 |

## By q

| q_e4 | profiles | relative | service | localization | segmentation | stress | zstd median range (bytes) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 12 | 10 | 0 | 2 | 9 | 0 | 230087.0–3568326.0 |
| 3000 | 12 | 9 | 0 | 0 | 9 | 0 | 176700.0–2513732.0 |
| 5000 | 12 | 9 | 0 | 0 | 5 | 0 | 129707.0–1787658.0 |
| 7000 | 12 | 9 | 0 | 0 | 0 | 0 | 81087.0–1063190.0 |
| 9000 | 12 | 9 | 0 | 0 | 0 | 12 | 28109.0–349312.0 |
| 9800 | 12 | 8 | 0 | 0 | 0 | 12 | 6229.0–71129.0 |

## Evidence qualifications

The frozen noAE and AE128 UINT8 records predate per-bin 0–30/30–40 reporting. Their range fields are null, their all-eight localization boolean is false, and the catalog records seven evaluated gates plus explicit not-evaluable provenance. No range metric is inferred from the historical 20–40 m aggregate.

Two four-profile network definitions exist (`network_profile_design_v1.json` and `network_profile_design_v2.json`) with conflicting schema/route/target-SNR definitions. Their exact paths and hashes are recorded, but the binding is `UNRESOLVED_MULTIPLE_CONFLICTING_DEFINITIONS`; this does not alter the 72 actions.

The transport supports continuous q mechanically at 1e-4 wire resolution over the registered mechanical range, but this initial RL action catalog includes only the six measured anchors. No unmeasured continuous-q action was generated.

LOCAL_CPU, LOCAL_GPU, and SKIP are not action IDs here. A future top-level policy head may choose SPLIT (then one of these 72), LOCAL_CPU, LOCAL_GPU, or SKIP; SKIP sends no perception output and advances map AoI.

## Pareto/dominance diagnostic

Pareto dominance over lower zstd median bytes and higher relative-preservation count, absolute-service count, localization count, and segmentation-installability; at least one objective must be strict.

Nondominated: 10; dominated: 62. All 72 profiles are retained and enabled.

## Validation

The builder failed closed on fixed input hashes, source terminals/schemas, Cartesian coverage, IDs, q/keep/drop arithmetic, summary/durable equality, Phase-11D same-family UINT8 references, finite defined metrics, positive payloads, codec/layout/zstd bindings, stress flags, source gate counts, transport integrity, and cleanup ordering.
