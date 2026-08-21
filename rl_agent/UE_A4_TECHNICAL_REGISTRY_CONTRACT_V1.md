# UE-A4 technical registry contract v1

## Purpose

UE-A4 freezes the 72 UE split actions that passed the authoritative UE-A2
local CUDA/wire smoke. It joins the immutable UE-A1 declarations to the
create-only UE-A2 `_02` evidence and changes only their evidence status from
pending to technically valid. It does not filter actions by perception quality,
payload size, or preference.

This successor is a certification artifact. The deployed v2 runtime and
launcher remain deliberately bound to the immutable UE-A1 CSV and wire
identity. UE-A4 must not retarget either source or overwrite UE-A1/UE-A2.

## Frozen inputs

- UE-A1 registry, manifest, terminal, and every manifest-declared output;
- UE-A2 `20260820_cuda_model_smoke_02` manifest, terminal, profile table, model
  summary, negative-test matrix, transport evidence, and every
  manifest-declared output; and
- the exact v2 runtime, v2 launcher, wire-contract helper, and production
  feature codec source seals recorded by UE-A2.

The earlier UE-A2 `_01` bundle is superseded and is rejected as authority.

## Join and promotion

The inputs must each contain exactly 72 unique rows. Their key sets must be
identical, with a one-to-one match on:

`action_index`, `profile_id`, `model_family`, `quantization_mode`,
`quantization_bits`, canonical `roi_drop_fraction`, and `checkpoint_sha256`.

For each row, UE-A4 recomputes the UE-A1 action contract and requires it to
equal the UE-A2 `action_contract_sha256`. All UE-A2 per-stage statuses must be
the exact passing values and the row must be `TECHNICALLY_VALID` with an empty
blocking reason.

The successor preserves the A1 operational `registry_schema`, `registry_id`,
`profile_id`, and `action_contract_sha256`. Separate successor fields identify
the UE-A4 evidence registry and its A1/A2 provenance. Current runtime use of
the A4 CSV is not authorized.

## Compression semantics

Two different hops intentionally use different codecs:

- UE to edge intermediate-feature wire: `zstd`, level 3;
- edge to spatial-map JSON packet: `zlib`, level 1.

The second binding is verified in the pinned v2 runtime. The per-fixture UE-A2
payload byte counts remain evidence for that fixture only and are not promoted
as general network-payload predictions.

## Acceptance and authority

Accept only if the A2 terminal reports 72 valid, zero invalid, zero blocked,
model inference and actual UDP executed, no quality gate, and no prior
successor registry. The model summary must report strict counts `4/4/4/24/72`,
no source drift, and the localhost transport must report 72/72. The negative
matrix must pass all 34 cases with zero decode/map calls after rejection.

The output is create-only and contains exactly 72 unfiltered rows, deterministic
row seals, a sealed manifest, and one terminal. It authorizes the next design
stage only; it does not authorize CARLA/OAI collection, policy training,
continuous ROI, local inference, or skip inference.
