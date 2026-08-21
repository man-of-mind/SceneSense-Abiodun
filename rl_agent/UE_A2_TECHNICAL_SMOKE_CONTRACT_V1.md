# UE-A2 technical smoke contract v1

**Status:** LOCKED BEFORE IMPLEMENTATION (2026-08-20)

**Owners:** Abiodun and Codex

**Scope:** single-UE fixed-profile split inference only. This contract does not
authorize CARLA driving, OAI/SNR actuation, the 72 x 4 collection, continuous-q
promotion, perception retraining, or policy training.

## 1. Objective

UE-A1 proved that 72 measured action declarations exist. UE-A2 must now prove
that every declaration can traverse the technical path:

```text
registered row
  -> strict front checkpoint load
  -> backbone / q / integrated AE / quantization
  -> pickle + zstd-3 + chunking
  -> reassembly + fail-closed identity check
  -> dequantization + integrated AE decode
  -> strict tail decode
  -> OBJECT_MAP_V1-compatible packet construction
```

Perception quality is not an A2 pass gate. A profile may produce an empty or
poor prediction and still be technically valid. Quality remains an outcome for
the later controller data sheet.

## 2. Version and provenance rule

The create-only UE-A1 bundle and its pinned sources remain immutable. UE-A2
uses versioned successor sources rather than silently changing what UE-A1
certified:

- runtime: `carla_fusion_staleness_scenario_uplink_only_v2.py`;
- OAI launcher: `run_track1_oai_default106_ttracer_10fps_v2.sh`;
- wire helper: `rl_agent/ue_split_wire_contract.py`; and
- A2 runner/config/tests under the `ue_a2` namespace.

The eventual UE-A4 registry is a successor artifact. It must not overwrite
`registries/ue_split_profile_registry_v1/`.

## 3. Stable action identity

`row_fingerprint_sha256` remains A1 evidence provenance. It includes mutable
review/status fields and therefore is not the lasting on-wire identity.

UE-A2 derives `action_contract_sha256` only from immutable operational fields:

- registry and profile IDs;
- model family, integrated-AE contract, checkpoint SHA-256;
- input and feature schemas and expected shapes;
- quantizer, q, entropy coder/level, and chunk size;
- decoder settings; and
- map-output schema.

The front sends this exact mapping in `profile_identity`:

```text
schema, registry_id, profile_id, action_contract_sha256,
checkpoint_sha256, feature_schema_id, feature_wire_schema_id,
quantization_mode, roi_drop_fraction, entropy_coder, entropy_level,
udp_chunk_bytes
```

The edge derives its expected mapping from its own selected registry row. It
must never select a model or codec from incoming identity fields.

## 4. Fail-closed registered mode

Both roles receive:

```text
--ue-profile-registry-csv PATH
--ue-profile-id PROFILE_ID
--require-ue-profile-binding
```

Registered mode must, before model construction:

1. resolve exactly one CSV row and verify its A1 row fingerprint;
2. hash the actual checkpoint and match the registered full hash;
3. reject external `--ae-checkpoint` overrides;
4. require the registered quantizer, q, zstd-3, chunk size, input size, and
   decoder settings; and
5. load the checkpoint strictly, with zero missing or unexpected keys.

The payload identity and serialized feature headers are validated immediately
after reassembly and before insertion into the edge work queue. A rejected
payload may not evict a valid queued frame, run deserialization/tail inference,
or publish a map packet.

Required shapes are:

| Family | Wire low | Wire high | High after edge AE decode |
|---|---|---|---|
| no-AE | `1x40x54x96` | `1x960x27x48` | `1x960x27x48` |
| AE-32 | `1x40x54x96` | `1x32x27x48` | `1x960x27x48` |
| AE-64 | `1x40x54x96` | `1x64x27x48` | `1x960x27x48` |
| AE-128 | `1x40x54x96` | `1x128x27x48` | `1x960x27x48` |

Accepted identity is copied into edge metrics, returned diagnostic results,
and the constructed map packet.

## 5. Fixed input and factored 72-action smoke

Use one real retained moving-UE input copied into the A2 fixture namespace:

```text
source: data_collection/experiments/phase2_paired_causal_v1/
        20260817_181354_pilot/phase2_pilot_benign_001/recipient/
        retained_inputs/frame_00156944_inputs.npz
sha256: 7fcfad2255c6626b8b87ff3a1c85ec7d32e17c8c2b4eee2875f5f132be423b41
bytes:  2499548
```

Only RGB, radar tensor, camera matrices, capture ID, and timestamp are used.
No paired/helper, scenario label, GT, or recipient-sharing logic is consumed.

Avoid 72 redundant backbone loads while still exercising every action:

1. For each of four checkpoint families, load a strict front model and encode
   the same input once.
2. Exercise all six registered q values and the family's integrated AE.
3. Exercise all three quantizers for every q output.
4. Cross the real pickle/zstd/chunk/reassembly/dequantization path for all 72.
5. Load a separate strict edge model per family and run AE/tail/object decode
   for all 18 family rows.
6. Construct an OBJECT_MAP_V1 packet for every row. Empty detections are valid;
   a separate synthetic non-empty result verifies object-field normalization.

CUDA is the deployment-representative gate. CPU execution is diagnostic only.

## 6. Transport infrastructure guard

Large no-AE payloads previously failed localhost UDP when the OS receive
buffer was capped below the complete burst. Therefore:

- deterministic in-memory pickle/zstd/chunk/reassembly must pass for all 72;
- actual localhost UDP must log requested and actual `SO_SNDBUF`/`SO_RCVBUF`;
- UDP runs proceed only when the granted receive buffer exceeds the largest
  message plus the frozen margin; and
- insufficient buffer is `BLOCKED_INFRASTRUCTURE_BUFFER`, never a technical
  profile failure.

UE-A2 does not change sysctls automatically.

## 7. Negative tests

Before any model smoke, table-driven tests must reject:

- missing identity or strict-binding arguments;
- unknown/duplicate profile ID;
- corrupt A1 row fingerprint;
- wrong action-contract or checkpoint hash;
- wrong feature/wire schema;
- wrong quantizer, q, coder, level, or chunk size;
- wrong feature names, declared shapes, or serialized header shapes;
- cross-profile payloads; and
- any external-AE override.

Every rejection occurs before feature decode and map construction.

## 8. Evidence and terminal states

A create-only A2 output contains:

- `resolved_config.json`;
- `fixture_manifest.json`;
- `ue_a2_profile_smoke.csv` with exactly 72 rows;
- `negative_contract_tests.json`;
- `transport_preflight.json`;
- `REPORT.md`;
- `manifest.json`; and
- exactly one terminal: `UE_A2_PASSED.json`, `UE_A2_FAILED.json`, or
  `UE_A2_BLOCKED_INFRASTRUCTURE.json`.

Per-row status is one of:

- `TECHNICALLY_VALID`;
- `TECHNICALLY_INVALID` with a reproducible stage/reason; or
- `BLOCKED_INFRASTRUCTURE`, which does not count as profile invalidity.

## 9. Acceptance

UE-A2 passes only when:

- 72/72 registry rows resolve and their action contracts verify;
- four front and four edge checkpoint loads are strict;
- 72/72 in-memory wire round trips pass;
- 72/72 identities and pre/post-AE shapes pass;
- 72/72 tail decodes are finite and map packets match the fixed schema;
- every injected mismatch is rejected before decode/map publication;
- no quality-derived action mask is introduced; and
- actual UDP either passes under adequate buffers or is truthfully separated
  as an infrastructure block.

Only after these local gates pass may Abiodun authorize one short fixed-profile
live wire smoke. That later smoke is not SNR calibration or data collection.
