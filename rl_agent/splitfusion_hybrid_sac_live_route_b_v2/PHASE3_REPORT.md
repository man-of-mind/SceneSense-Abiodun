# Run-4 live-qualification package v2 — Phase 3: continuous `(mode_id, q_e4)` execution

**Verdict:** `PHASE3_CPU_IDENTITY_PROTOCOL_PARITY_PASSED`. Offline only. This is CPU
protocol/identity parity. It makes **no** CUDA, model-output or live-performance claim.

## Implementation (`continuous_execution_v2.py`)

- **One authoritative identity seam.** `executed_identity_from_profile(profile, contract)`
  first requires `contract.verify_profile(profile)`, which is full equality with the
  contract's own resolution. It then copies mode, family, quantizer, exact `q_e4`, keep/drop,
  nullable `action_id`/`profile_id` and the catalog binding into `ExecutedActionIdentity`, and
  calls the catalog's `reconciled_against`. No float is quantized. `measurement_status` must
  agree with anchor presence.
- **Codec view.** `CodecProfileViewV2` has the attributes `ProductionSplitCodec` reads. Its
  `q` is derived from the exact `q_e4` (`q_e4 / 10000`); the legacy `profile.q` float is never
  passed. At construction, `continuous_q.quantize_q(view.q)` must reproduce `q_e4`,
  `keep_count` and `drop_count` without snapping. A sweep over all 9,801 values confirmed this:
  0 mismatches, `snapped` never set.
- **SFD3 envelope.** It carries:
  - the action: mode, exact `q_e4`, keep count, nullable anchor `action_id` (flagged),
    `reward_requested`;
  - the frame context: session UUID, controller-lineage SHA-256, decision, ticket, frame,
    tensor, capture timestamp;
  - the digests: execution-bundle SHA-256, inner-payload SHA-256, and a SHA-256 over the
    header.

  Any version, length, flag or digest inconsistency fails closed. SFD1 and its anchor-only
  runtimes are untouched: SFD1 still round-trips, SFD3 rejects SFD1, and SFD1 rejects SFD3.
- **Runtimes.** `ContinuousUERuntimeV2` drives the ranker, which is bypassed only at
  `q_e4 == 0`, and the registered per-family encoder through the codec. `ContinuousEdgeRuntimeV2`
  re-resolves the profile from `(mode_id, q_e4)`. It refuses any bundle, keep or anchor
  mismatch, re-derives the identity through the same seam, and requires
  `require_inner_agreement` between the inner header and the profile.

## Tests (`test_phase3_continuous_execution_v2`: 12 OK, 15.5 s)

- All 72 anchors keep their exact `(action_id, profile_id)`.
- **The seam equals the training identity path** (`catalog.resolve(mode, q_e4/10000)` →
  `from_executable_action`) for **all 12 × 9,801 = 117,612** `(mode_id, q_e4)` pairs, including
  canonical serialization.
- Off-anchor lower, middle and upper values inside each mode's registered actor support carry
  `action_id=None` and `UNMEASURED_OFF_ANCHOR`, with no snapping and exact keep/drop.
- UE→edge round trip for every mode at 3 off-anchor and 6 anchor values.
- Selective saliency drop: the registered `_selection` keeps exactly the top-`keep_count`
  ranker scores (checked at `q_e4` 1, 3000, 4321 and 9799). `q_e4 = 0` never calls the ranker.
- Serialization is deterministic, and the `reward_requested` flag is carried.
- The edge refuses changed mode, q, keep, fabricated anchor, dropped anchor, wrong bundle, and a
  codec whose inner header disagrees.
- Header or payload bit-flips, truncation and a foreign magic all fail closed.

Fakes: front, ranker (fixed CPU score map), AE objects, JSON framing codec, and tail. The real
zstd/UINT codecs, CUDA models and FCOS are not exercised here.

## Files

Created: `continuous_execution_v2.py`, `test_phase3_continuous_execution_v2.py`,
`PHASE3_REPORT.md`. Nothing outside this package is modified.
