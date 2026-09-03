# Phase 9D — selected AE128 UINT8 + mandatory-zstd validation (implementation)

> **Executed 2026-09-03.** This document describes the implementing commit
> (`eec52b0`) and the statement below refers to *that* commit, not to the
> project state today. The single authorized validation run has since been
> executed exactly once from `eec52b0`; its result is
> `experiments/splitfusion_fcos_ae_v1/20260903_phase9d_ae128_uint8_validation/`
> (`AE_PHASE9D_UINT8_VALIDATION_REPORT.md`,
> `phase9d_ae128_uint8_validation.json`, terminal
> `SPLITFUSION_AE128_UINT8_VALIDATION_COMPLETE`). The preregistered rule was
> applied verbatim and returned
> **`AE128_UINT8_ZSTD_DEPLOYMENT_NOT_ACCEPTED`**: q=0 met its condition
> (12/12 same-q preservation gates, 7/9 absolute service gates against the
> registered 7/9 baseline), but no primary q qualified — q=0.30 and q=0.50 each
> lost `person_avo_recall` and q=0.70 lost `person_avo_f1`,
> `person_avo_precision` and `vehicle_precision`. Nothing was retuned, removed
> or rerun in response.

Implementation only. Nothing was executed: no inference, no CUDA context, no
validation pass, no training, no CARLA. The dirty `OAI/openairinterface5g`
working tree was left untouched. Only `py_compile`, the two new focused CPU
tests, the three pre-existing AE test files and diff checks were run.

The runner is `ae_uint8_validation.py`; it is **not** invoked by this change.

## What this phase will measure

Exactly the six registered q anchors — `q = {0.00, 0.30, 0.50, 0.70, 0.90,
0.98}` — once each, on the registered 3,345 validation frames, through the real
deployment path and nothing shorter:

```text
original FP32 C2
  -> selected AE128 encoder (always, on the complete frame)
  -> per-channel UINT8 latent quantization, ranges from the complete latent
  -> sparse AE latent wire
  -> mandatory zstd level 1
  -> received raw bytes
  -> exactly one decompression
  -> header-driven preloaded AE128 decoder selection
  -> dequantization / zero scatter
  -> AE128 decoder
  -> unchanged frozen perception tail
```

One inference/evaluation pass per q, enforced three ways: `run_validation_pass`
refuses an unregistered q, each completed q is persisted atomically as
`settings/qNNNN.json` carrying `inference_passes_for_this_q = 1`, and a rerun
reuses a fully validated setting instead of measuring it again. Test data is
never opened and CARLA is never launched.

## Durability order per q

`DURABILITY_ORDER` is fixed and the loop follows it exactly:

1. atomically write `settings/<q>.json` — write-beside, fsync, rename, fsync the
   directory. **This JSON is the durable scientific completion record.**
2. only after that write succeeds: remove `working_predictions/<q>`;
3. atomically write `cleanup/<q>.json`, the cleanup-complete marker, last.

So an interruption can lose at most a scratch prediction directory, never a
completed measurement. The record is written *before* removal, so it no longer
claims removal: it carries a `prediction_artifacts` block with
`removed_before_this_record: false` and names the marker that will record the
removal, and `load_durable_setting` refuses a record that claims otherwise. The
marker binds the setting's SHA-256, so a marker can only ever vouch for the
exact record it was written for.

`reuse_or_complete` is the whole resume step for one q and measures nothing,
ever. It returns `None` only when no setting JSON exists — the single case in
which the caller may run a pass. Otherwise it validates the record in full via
`load_durable_setting`, and if the cleanup marker is absent or does not bind
that record it removes any surviving prediction directory and writes the marker.
Both cleanup halves are idempotent, so an interruption anywhere in the window is
resolved by finishing the cleanup rather than by remeasuring. A setting JSON
that fails validation raises rather than being reused *or* silently remeasured.

`load_durable_setting` validates, not spot-checks: schema and terminal, run
identity, both `q` and `q_e4`, 3,345 frames, exactly one inference pass, the
registered keep and drop cardinality, the exact analytical pre-zstd payload size
against every measured pre-zstd statistic, one payload sample per frame with all
statistics finite, `zstd_mandatory`, the expected ranker invocation count (0 at
q=0) and 3,345 zstd decompressions, every integrity flag that must be true and
every one that must be false, the finite-result and frozen-state flags, the
no-training / no-gate-change / no-test-access flags, the complete and finite
protected and canonical metric sets, nine self-consistent absolute service gates,
twelve self-consistent same-q preservation gates with the expected baseline
label, the full protected-metric delta set, and a same-q noAE reference for the
right q and keep count.

Freshly measured rows go through the same door: the loop writes the setting,
re-reads it through `load_durable_setting`, and appends *that*, so the in-memory
row is exactly the durable bytes and the writer's own output is required to
satisfy the resume validator. `finalize` additionally records every cleanup
marker's hash and refuses to emit a result unless every completed q has both a
durable setting and a marker that binds it.

## Bound artifacts

| role | path | sha256 |
| --- | --- | --- |
| selected AE128 | `experiments/splitfusion_fcos_ae_v1/20260902_220623_phase9c_ae128_training/checkpoints/ae128_epoch_08.pt` | `0c2ba3a4…db30f72` |
| holdout decision | `…/20260902_220623_phase9c_ae128_training/holdout_selection/holdout_selection.json` | `69e49dea…0f878194` |
| frozen noAE UINT8+zstd reference | `experiments/splitfusion_fcos_hybrid_q_v1/20260902_223610_phase8b_uint8_validation/phase8b_uint8_validation.json` | `a2779f5f…04146bb029` |

All three were verified against the files on disk. The frozen perception
checkpoint, the stable epoch-4 ranker, the p025 forward lock and the hybrid-q
locked configuration are bound by reusing `ae_gpu_qualification.bind_inputs()`
verbatim, which already fails closed unless all four equal the authorized
Phase-9B digests.

The chain from the decision to the checkpoint is enforced, not assumed:
`load_holdout_decision` requires the decision's schema and terminal, its
`selected_epoch == 8`, its recorded `ae128_epoch_08.pt` hash to equal the bound
checkpoint hash, `validation_or_test_accessed == false`, its FP32-latent
transport label, and its four frozen bindings to equal ours.

`load_noae_reference` requires the reference's schema and terminal, 3,345 frames
per q, one pass per q, no test/CARLA access, no training or gate change,
`zstd_mandatory`, the exact six-q ladder with correct keep counts, the exact
protected-metric set, nine service targets per row, and that its **q=0 row
reports the registered 7/9 absolute service baseline**. Nothing is recomputed
from it.

## Routing tag

`routing_tag()` is `ae_contract.routing_tag_from_sha256(SELECTED_CHECKPOINT_SHA256)`
— the leading 32 bits of the **full** selected-checkpoint digest — and is
`0x0c2ba3a4` (204,186,532), nonzero as the deployable paths require. Both facts
are recorded side by side by `routing_record()`, which states explicitly that
the tag is a per-frame decoder-routing discriminator and
`routing_tag_is_checkpoint_identity: false`, with the full SHA-256 named as the
identity authority. The report and the JSON never describe the 32-bit tag as
checkpoint identity.

## Per-frame integrity audit

`_transport_one` transmits with the unmodified `ae_uint8_transport.encode_frame`
and receives with the unmodified `PreloadedAeDecoders.receive` over the raw wire
bytes, then refuses the frame unless all of the following hold:

- the wire `q_e4` equals the pass q and the keep count equals the registered
  cardinality, on both the header and the reconstructed mask;
- the family is AE128, the transported latent is 128 channels, the routing tag is
  the bound tag, and the codec id is the AE latent UINT8 wire;
- the retained UINT8 value block is `[keep, 128]`, and its cell indices are
  **exactly** the indices selection chose (`arange` at q=0);
- every dropped cell is exactly zero across all 128 latent channels *before* the
  decoder runs, checked on the latent `receive` itself decoded;
- exactly one zstd decompression occurred, measured by a counting wire codec
  reset per frame, not asserted by construction;
- the decoder was discovered from the received header bytes — `receive` is handed
  raw bytes only, `expected_packet` is never passed, and the returned decoder
  object must be the one preloaded AE128;
- the reconstructed C2 is `[256,112,192]` FP32 and finite, and is **not**
  bit-identical to the original, so no q is silently an identity;
- at q=0 the ranker was invoked zero times while AE128 still ran; at q>0 it was
  invoked exactly once, on the original FP32 C2 tensor object (checked by data
  pointer), before any cell was dropped.

Per pass, the aggregate is re-checked: one pre-zstd size equal to
`ae_uint8_transport.analytical_size(q, 128)`, 3,345 compressed sizes, one keep
count, exactly 3,345 ranker invocations (0 at q=0), exactly 3,345
decompressions, and the exact registered frame order and coverage. Frozen
perception, the stable ranker and the selected AE128 are snapshotted
(parameters *and* buffers) and re-verified unchanged after every pass and again
at finalization; the AE128 per-tensor and aggregate state hashes are recorded.

One honest limit is recorded rather than glossed: the "ranges from the complete
latent before dropping" property is enforced structurally by the hash-bound,
unmodified `encode_frame` and its committed transport test, not re-derived per
frame — re-deriving it would require a second encode of every frame. The
integrity block says so in `ranges_evidence`.

## Scoring

Unchanged throughout: the p025 service policy, the AVO ≥ 0.65 person view, the
canonical v010 person view at the locked 0.25 output threshold, the frozen
vehicle scorer at 0.20, the frozen segmentation scorer, and the frozen geometry
evaluator, all reached through `phase6_validation.score_validation_pass` and
`phase5_common.load_frozen_scorers`. No threshold, NMS setting, calibration or
gate was touched, and the scorer hashes are recorded.

Per q the report carries: vehicle and person precision / recall / F1 / XY MAE in
both the canonical-p025 and AVO views, person 20–40 m recall, vehicle IoU,
person box-mask IoU, foreground mIoU, the nine absolute service-gate results,
the twelve same-q preservation gates against the frozen noAE UINT8+zstd result,
and the failed-gate names with their exact signed degradations and bounds.

## Payload reporting

Actual measured mean / median / p95 (plus min / max) bytes for both the pre-zstd
sparse payload and the zstd wire, with the exact analytical pre-zstd breakdown
(header / mask / ranges / values). Ratios against all three registered
references:

1. framed FP32 noAE q=0, `contract.FRAMED_Q0_PAYLOAD_BYTES`, asserted to be
   22,020,140 bytes;
2. the frozen noAE UINT8+zstd row at the **same** q — exact pre-zstd bytes and
   median/p95 zstd bytes; that artifact publishes no mean compressed size, so no
   mean-vs-mean ratio against it is claimed;
3. this run's own AE128 UINT8+zstd q=0 row.

Component latency is recorded as diagnostics from what the path already exposes:
a transmit stage (ranker/selection at q>0, AE128 encode, range preparation,
UINT8 framing, zstd compression), a receive stage (the one decompression, header
inspection and decoder selection, dequantize/scatter, AE128 decode), and the
frozen tail per batch, each with an explicit include/exclude list. It is labelled
current-host evidence only; **no Raspberry Pi and no OAI latency is claimed or
implied anywhere.**

## Preregistered interpretation

`ACCEPTANCE_RULE` and `evaluate_acceptance` are registered in source before any
Phase-9D number exists and are applied verbatim:

- q=0 must pass all 12 same-q preservation gates **and** retain at least the
  baseline 7/9 absolute service gates;
- **and** at least one of q = {0.30, 0.50, 0.70} must pass all 12 same-q
  preservation gates without reducing the absolute service-gate count below the
  frozen noAE UINT8+zstd count at that same q.

"Without reducing" is `>=`, and the baseline for q=0 is the registered
`contract.FROZEN_Q0_SERVICE_PASS_COUNT`, cross-checked against the frozen noAE
q=0 row at load time. q=0.90 and q=0.98 are recorded as stress/emergency
profiles with `influences_acceptance: false` and can neither create nor destroy
acceptance. Every q is reported independently, and the document records
`setting_tuned_after_observing_validation: false` and
`setting_removed_after_observing_validation: false`.

## Changes to the existing AE package

Deliberately small, and every one of them is reported as a source delta against
what the selected checkpoint recorded:

1. `ae_family_dispatch.py` — new `ReceiveDiagnostics` and an opt-in
   `receive(..., diagnostics=True)` that hands back the intermediates the call
   already produced (parsed header, decoded latent, keep mask, selected
   decoder). No decode step changed and no work was added; the default is off so
   deployment does not keep a latent alive per frame. Without this the
   validation pass could only audit keep indices and zero-scatter by
   decompressing the frame a second time or by restating `receive`'s body — both
   worse than returning what already exists.
2. `ae_training_common.py` — the registered same-q preservation-gate function
   moved here from the Phase-9C holdout runner and gained a keyword-only
   `baseline` label. Gate set, bounds, sign convention and pass rule are
   byte-for-byte the same logic. The move keeps a *validation* command from
   importing the *train-holdout selection* runner: `ae_holdout_selection` is
   verified absent from `sys.modules` after importing the Phase-9D runner.
3. `ae_holdout_selection.py` — re-exports that function under the same name, so
   its call site and its published artifact are unaffected.

New files: `ae_uint8_validation.py`, `tests/test_ae_uint8_validation.py`, and
this report. No model, loss, optimizer, architecture, q semantics, codec, ranker
or scorer changed.

### How the loader handles that

Phase 9D *is* an addition to the AE package, so the AE package source map the
checkpoint recorded cannot be bit-identical to the live one.
`require_selected_bindings` therefore iterates every field
`common.binding_fields()` writes — so no binding can be silently forgotten —
requires exact equality for all of them except `ae_package_source_sha256`, and
routes that one field through `ae_package_source_delta`, which fails closed
unless:

- every module defining what the saved tensors mean (`ae_contract.py`,
  `ae_model.py`) and every module defining the measured transport
  (`ae_composition.py`, `ae_loss.py`, `ae_uint8_transport.py`, `__init__.py`) is
  byte-identical to what the checkpoint recorded;
- no recorded file was removed;
- no file outside the explicit allowlist (`ae_family_dispatch.py`,
  `ae_holdout_selection.py`, `ae_training_common.py`) changed.

Every added and changed file is reported with its before/after hashes either
way. The checkpoint is additionally bound by its own full SHA-256 — a stronger
statement than any source map — and its embedded `configuration` must equal
`common.training_configuration()` exactly, so a change to the shared module that
moved the locked training configuration would be rejected by value.

The current delta is exactly: changed `ae_family_dispatch.py`,
`ae_holdout_selection.py`, `ae_training_common.py`; added
`ae_uint8_validation.py`, `tests/test_ae_uint8_validation.py`, this report; none
removed; all six semantics modules unchanged.

## Tests

Three focused CPU tests in the one Phase-9D test module,
`tests/test_ae_uint8_validation.py` — no new test file or suite was added. Each
records the process-global `torch.cuda.is_initialized()` flag on entry and
asserts it is exactly where it found it on exit, so no test creates a CUDA
context regardless of what ran before it in the same process.

1. **The acceptance rule, applied verbatim** over synthetic per-q rows: both
   conditions met accepts and names the qualifying q; q=0 losing one gate blocks
   acceptance even when a primary q qualifies; q=0 below 7/9 blocks it while
   matching or beating 7/9 does not; a ladder where every primary q either loses
   a gate or reduces the service count is refused, with
   `reduces_absolute_service_gate_count` set on the right row; "without
   reducing" is confirmed to be `>=`; perfect q=0.90/0.98 rows cannot rescue a
   failing ladder and collapsed ones cannot spoil a passing one; and a missing
   q, a duplicated q, or a noAE q=0 row that is not 7/9 is refused.
2. **The exact selected-checkpoint and routing binding** against the real
   artifacts: the three bound SHA-256 constants equal the files on disk; the
   decision selects epoch 8 and records exactly this checkpoint hash;
   `load_holdout_decision` accepts the real decision and is refused when any one
   of its four frozen bindings drifts; the routing tag equals the derivation from
   the full digest, is nonzero, fits in 32 bits, and differs from the tags of
   both unselected candidates; `routing_record()` reports the full digest and
   `routing_tag_is_checkpoint_identity: false`; the real recorded-vs-live source
   delta passes with all six semantics modules byte-identical and only
   allowlisted files changed; and the delta is refused for a changed semantics
   module, a removed file, or an unallowlisted change. `require_selected_bindings`
   is checked to enforce every other binding field, both when drifted and when
   absent. The checkpoint is opened only to read its metadata; its tensors are
   never built into a model or moved to a device.

3. **The interruption window**, reproduced exactly: a durable setting written
   for q=0.30 by the real `_setting_document`, a populated scratch prediction
   directory, and no cleanup marker. `reuse_or_complete` then returns the
   durable record byte-for-byte, and with `_atomic_json` recorded the **only**
   write is the cleanup marker — the setting is not rewritten — while
   `run_validation_pass` is patched to raise, so remeasuring the q would fail
   the test. The prediction directory is gone, the marker carries the cleanup
   terminal and binds the setting's SHA-256, and a second resume writes nothing
   further. A record with a wrong frame count is refused rather than reused or
   remeasured, and a q with no record at all returns `None`, the one case that
   permits a pass. Because the fixture is built by the real writer, this also
   pins that `_setting_document`'s output satisfies `load_durable_setting`.

Result: 3 tests in the Phase-9D module pass, and the three pre-existing AE test
files still pass (15 tests) unchanged — 18 together.
