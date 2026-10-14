# Correction 01 — response to independent review v1 (HOLD_FOR_OFFLINE_CORRECTION)

**Status: READY_FOR_INDEPENDENT_OFFLINE_REVIEW.** This was offline only. There was no live run, registration,
deployment, deletion or unrelated refactoring. The review evidence in `../splitfusion_gt_geometry_containment_review_v1/`
was not written; its result hash was checked before and after. All earlier evidence here (`*_01`, `MANIFEST.json`,
`REPORT.md`, `HANDOFF.md`) is preserved. Where they differ, this document and the `_02` runs govern.

## Blocker 1 — the candidate dropped the current V15/V16 prep stack (confirmed, fixed)

**What the review found.** The first candidate's worker entry composed only the avo_tilt-era layers. The review's
entry probe showed `payload_storage_bytes` was the ORIGINAL function and no `storage_accounting` was registered. That
means it lacked the corrected V3.v2 accounting session, its capability refusal, the bounded per-call timer and its
cleanup.

**Fix.**

- **`contract.py`** now star-imports the **V16** contract. `gc_registration` is V16's: `storage_accounting` is
  unchanged, and only `trampolines`, `implementations`, `seam` and the added `geometry_containment` differ. A unit
  test compares the two registrations field by field. `exact_sources` is V16's set plus this package's sources.
- **`worker_entry.py`** composes the prep stack explicitly, so that containment sits inside the lifetime repair:
  1. The V16 accounting layer, copied from `m3_route_b_v16.worker_entry`. It has the same capability check and
     REFUSAL, the same `switch.install(max_objects=32768)` session wrapped by V16's own `AccountingTimer`, and the same
     teardown and records (`storage_accounting_binding.json`, `storage_accounting_timing.json`).
  2. The V8 tilt-AVO layer.
  3. The V7 FREEZE lifetime layer.
  4. Containment.
  5. V4 → V3.

  The score role is V16's (V8 trampoline). `service_entry.py` and `gc_seams.py` were already code-identical to V16's;
  only their docstrings changed.

**Evidence that accounting is ACTIVE, not just inventoried.**

| Evidence | Result |
|---|---|
| `review_rerun_01` | The review's own `entry_probe`, unchanged: candidate `accounting_v3_active` and `registration_has_storage_accounting` both true; PROBES_PASS 5/5 |
| `tests_02` `WorkerEntryCompositionTest` | Inside the retained main, the owner global is the V16 `AccountingTimer` wrapping an `AccountingSession`, with tilt AVO and `PrepCoreContained` installed. The record set equals V16's plus `geometry_containment_binding.json`, and everything is restored. A missing capability still raises V16's REFUSED error |
| `service_proof_02` (real service, 27/27) | Accounting is active in every completed arm: installed, timer installed/removed, restored, fast path, session closed, V3.v2 version, `AccountingSession` wrapped, calls = recorded, 0 dropped/errors/foreign |
| `verify_engine_02` (12/12) | All in-process runs also hold the V3.v2 session: 778 calls, 0 failed, fast path |

`service_proof_02` per-caller detail:
- **Upright arms:** the V16 stack and the candidate stack have identical per-caller counts (13/13/13/13) and identical
  accounting result bytes (sha `4856e690…`).
- **Contained arm:** `compute_unbound` 34 = 20 valid + 14 refused, and `stage_products` / `bind_plan` / `set_ready`
  20 each. Their result-byte sequences equal the unchanged V16 control's, so refused frames left no accounting trace
  in later frames.

## Blocker 2 — contained refusals skipped plan validation (confirmed, fixed)

**What the review found.** The first `bind` for a refused frame only checked canonical bytes and the capture
template. A missing `action`, a missing `frame_context` or a mismatched action frame produced an ordinary geometry
refusal, where a normal frame fails fatally.

**Fix (`engine.py`).** A refused frame's `bind` now applies the same three layers, in the same order and with the same
failure semantics, as the inherited bind of a normal frame. Only a fully valid plan reaches the geometry refusal.

1. **Owner plan checks:** the checks of `DeferredProducerOwner.bind_plan`, verbatim. They cover canonical JSON,
   `capture` and `action` present, the capture schema, `frame_context` as a dict, run-4 identity, reward-requested,
   action/capture frame equality, POLICY_DECISION, and the reservation bound. A failure sets the owner failure
   (`invalid producer plan: …`) and the engine failure (`plan binding failed: …`), then re-raises the original
   exception. This is fatal.
2. **Identity/anchor:** a capture that differs from the template goes through `deferred.anchor_provenance`. An invalid
   one raises the existing identity `DeferralRefused` (`SOURCE_IDENTITY_MISMATCH` / `CATALOGUE_ANCHOR_MISMATCH`); this
   is non-fatal, exactly as for a normal frame. A valid anchor is accepted.
3. **Admission:** every check of `ready.admit_payload → _metadata` that does not read the never-produced payload:
   - `_action` and `_context_document`;
   - the registered prediction contract;
   - context ↔ capture/action identity and `sequence_id`;
   - the live-context/source-camera pose agreement of `_frame_geometry`, applied to the source camera retained at
     refusal time, using the comparison `_frame_geometry` uses in this process (the registered P3 rule when installed,
     else `PO._agree`).

   A failure sets the engine failure `bound admission failed: …`. This is fatal.

**Regression tests (`tests_02` `PlanValidationTest`).** For 17 mutations, a normal frame (case 6, UNCHANGED engine)
and a refused frame (case 7, S1 tilt, contained) must classify identically. That includes the exception type and the
engine and owner failure flags. The single exception is that a valid plan gives OK on the normal frame and
`GT_GEOMETRY_ENVELOPE_TARGET` on the refused one. Result:

| Mutation(s) | Normal = refused |
|---|---|
| **missing_action** (review) | FATAL KeyError, engine+owner failed |
| **missing_frame_context** (review) | FATAL OwnershipError, engine+owner failed |
| **action_frame_mismatch** (review) | FATAL OwnershipError, engine+owner failed |
| missing_capture / capture_schema_invalid / frame_context_not_dict / non_canonical_bytes / not_json | FATAL (KeyError / ContractError / OwnershipError / OwnershipError / ContractError) |
| not_reward_requested / hold_frame_kind / run4_identity_invalid | FATAL DirectMapProtocolError |
| foreign_stream_capture | REFUSED SOURCE_IDENTITY_MISMATCH, not failed |
| tensor_seq_disagrees_with_context / other_frame_context | FATAL ContractError, engine failed only (as normal) |
| context_pose_shifted_1m | FATAL PoseParityError (engine failed) |
| valid_plan / valid_anchor_plan | normal OK; refused → `GT_GEOMETRY_ENVELOPE_TARGET`, not failed |

The review's own `plan_probe`, unchanged (`review_rerun_01`), now gives `malformed_plans_remain_fatal` true.

**Narrow recovery and rollback are unchanged.** `classify.py`, the rollback invariant and the FREEZE handling are
unmodified (hash-checked), so `replay_s1_01` stays valid and was not rerun. The `_02` runs reconfirm:
- **Engine:** 49 upright frames exact; 7 mixed refusals with 42 accepted frames exact against the reference; 43
  successive refusals with no history advance and capacity back to 0; bbox corruption fatal; close semantics as before.
- **Service:** 14 refusals each with exactly one `NO_GT_DEFERRAL_REFUSED_GT_GEOMETRY_ENVELOPE_*` outcome and
  `GROUND_TRUTH_UNAVAILABLE` feedback; 20 valid frames exact and identical to the V16 control; prep fd count 4
  throughout; bbox corruption fatal with the original error and a hash-verified V7 snapshot.

## Handoff correction

The registration-bound offline assessment covers two populations:
- **Actor replay** covers **all policy decisions**, including GT-unavailable ones.
- **Exact rescoring** covers **SCORED** decisions only, as in V15, whose queue-full refusal was replayed but not
  rescored.

The first HANDOFF wrongly said that replay was limited to SCORED decisions. `HANDOFF_02.md` replaces it.

## Unchanged limitations

The limitations in REPORT §7 still apply:
- refused frames are never committed to GT history;
- map delivery is argued from code and V15 precedent;
- the fixtures are synthetic, and the POSSIBLE_COMPETITOR case needed a relocation;
- the first two refusals write about 11.6 MB each in the FREEZE reply path;
- the outcome label carries the `DEFERRAL_REFUSED_` transport prefix;
- the containment is not live-qualified.

New in this correction: the pose check for a refused frame runs on the source camera matrix rather than the producer
payload's. This is the same matrix the producer copies, and the valid plain and anchor plans passing the check
confirm the agreement. The other `_frame_geometry` checks are payload integrity checks (frame manifest, intrinsics,
AVO timestamp), so they have nothing to validate when no payload exists.

Evidence:
- **Runs:** `tests_02.log`, `dependency_tests_02.log` (86/86: avo_tilt, FREEZE lifetime, V16), `review_rerun_01/`,
  `verify_engine_02/`, `service_proof_02/`.
- **Hashes:** `MANIFEST_02.json` (package, unchanged dependencies including the review files, inputs, results, logs).
  Evidence is 325.5 MB in total, with 37.06 GB free.
- **Commands:** appended to `COMMANDS.txt`.
