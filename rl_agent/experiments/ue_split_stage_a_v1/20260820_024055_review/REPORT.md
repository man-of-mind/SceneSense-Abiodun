# UE SPLIT Stage-A evidence report

**Verdict:** `PASS_EVIDENCE_ASSEMBLED_REVIEW_REQUIRED_NO_RUN_AUTHORITY`

This is a reuse-only evidence assembly. It authorizes no CARLA/OAI run,
profile freeze, controller implementation, or training.

## What passed

- 72 unique measured profiles over 2,162 identical frames each.
- 155,664 profile-frame rows passed the frozen grid and count identities.
- All four registered integrated checkpoints and primary evidence files independently match their pinned SHA-256 values.
- The retained run record names those checkpoints, but the CSV rows do not embed a cryptographic checkpoint lineage field.
- Object detection/localization are primary; segmentation is reported only as a secondary diagnostic.

## Exploratory quality sensitivity

These ROI-incremental counts compare each profile with its same-model/
same-quantization q=0 baseline. They do not apply an absolute service floor,
are not final eligibility decisions, and do not certify missing small/far strata.

| Sensitivity floor | ROI-incremental profiles passing |
|---|---:|
| `prior_reference_exploratory` | 42 / 72 |
| `relaxed_exploratory` | 57 / 72 |
| `strict_exploratory` | 27 / 72 |

## Existing network capacity projections

The historical runs achieved about 5.8--8.0 sends/s and did not log
authoritative map update-done events. The 10-Hz values below are capacity
projections with a registered engineering uncertainty of plus/minus 30%,
not direct action certifications.

| Regime | Historical ID | SNR (dB) | MCS | Capacity (Mbps) | 10-Hz equivalent (KiB/frame) |
|---|---|---:|---:|---:|---:|
| clear | `clear` | 50.3 | 28 | 36.68 | 447.8 |
| mild | `mild` | 19.5 | 24 | 27.78 | 339.1 |
| mid | `mid15` | 15.6 | 19 | 19.71 | 240.6 |
| poor | `strong` | 8.2 | 9 | 10.39 | 126.8 |

## Existing staleness evidence

Direct loopback records do reach `map_update_done`; they provide capture-to-map
latency anchors, not profile-specific OAI AoI. The historical localization budget
is emitted separately as a fixed-floor latency-tolerance proxy and cannot select
`AoI_max` without later accepted-update events.

| Anchor | Evidence | P50 (ms) | P95 (ms) |
|---|---|---:|---:|
| `fresh_L_fast` | direct_capture_to_map_update_done | 67.9 | 100.4 |
| `fresh_L_normal` | direct_capture_to_map_update_done | 67.8 | 106.1 |
| `fresh_L_veryfast` | direct_capture_to_map_update_done | 66.5 | 94.7 |
| `fast rasterizer (conservative design anchor; 40-frame profile)` | registered_historical_latency_anchor | 93.3 | 136.1 |
| `legacy rasterizer (pre-optimization)` | registered_historical_latency_anchor | 180.7 | 247.5 |
| `core split->map (NOT the staleness lag)` | registered_historical_latency_anchor | 37.7 | 80.2 |

## Provisional payload-boundary candidates

These are mechanically nearest to each projected capacity center. They
are not approved measurements and may disappear once the quality floor
freezes the eligible catalog. The ratio includes estimated custom/UDP/IPv4
overhead but not GTP, PDCP, RLC, MAC, or scheduling overhead, so below/above
remains only a feature-payload proxy.

| Regime | Role | Profile | P95 feature payload (KiB) | UDP/IP proxy load ratio |
|---|---|---|---:|---:|
| clear | nearest_above_feature_payload_proxy | `noae/u6/q0.5` | 475.3 | 1.062 |
| clear | nearest_at_or_below_feature_payload_proxy | `noae/u8/q0.7` | 428.3 | 0.957 |
| mid | nearest_above_feature_payload_proxy | `ae128/u8/q0.3` | 258.8 | 1.076 |
| mid | nearest_at_or_below_feature_payload_proxy | `ae32/u8/q0` | 230.0 | 0.957 |
| mild | nearest_above_feature_payload_proxy | `noae/u6/q0.7` | 340.0 | 1.003 |
| mild | nearest_at_or_below_feature_payload_proxy | `noae/u4/q0.3` | 308.3 | 0.910 |
| poor | nearest_above_feature_payload_proxy | `ae32/u8/q0.5` | 126.9 | 1.001 |
| poor | nearest_at_or_below_feature_payload_proxy | `ae32/u6/q0.3` | 126.5 | 0.999 |

## Required decision before measurements

1. Review the object-quality sensitivity and missing difficult-object evidence.
2. Select and record one absolute `OBJECT_MAP_V1` quality floor.
3. Form N aggregate candidates, then resolve required difficult-object evidence for the survivors.
4. Freeze only fully supported eligible actions in a new immutable sibling bundle.
5. Recompute the exact 4N logical surface and authorize only remaining boundary cells.

No `COMPLETED.json` or action catalog is written in this review bundle.
