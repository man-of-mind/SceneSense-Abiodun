# Phase 10B — AE64/AE32 UINT8 + mandatory-zstd validation runner (implementation only)

Implementation report for the one shared, family-aware deployment-path validation
runner for the two selected Phase-10A checkpoints. **Nothing was executed.** No
checkpoint was loaded into a model, no CUDA context was created, no dataset,
teacher shard, validation or test data was opened, nothing was trained, tuned,
selected or scored, CARLA was not launched, and no completed artifact was
modified.

## What was added

Exactly three files, all new:

| file | role |
| --- | --- |
| `ae_phase10b_uint8_validation.py` | the one shared AE64/AE32 validation runner |
| `tests/test_ae_phase10b_validation.py` | two focused CPU test areas, four cases |
| `AE_PHASE10B_UINT8_VALIDATION_IMPLEMENTATION_REPORT.md` | this report |

No existing AE, hybrid-q, scoring, training, selection or Phase-9D file was
modified. `git show --stat` on the commit is the check.

These three paths are also the *only* additions the runner's own source-map
guard permits (`AE_PHASE10B_ADDED_SOURCES`), so the allowlist and the commit are
the same list.

## Public commands

One runner, one family per process, family selected by `--bottleneck`:

```
python3 -m ...ae_v1.ae_phase10b_uint8_validation \
  --execute SPLITFUSION_AE64_PHASE10B_UINT8_VALIDATION --bottleneck 64 --output <run>

python3 -m ...ae_v1.ae_phase10b_uint8_validation \
  --execute SPLITFUSION_AE32_PHASE10B_UINT8_VALIDATION --bottleneck 32 --output <run>
```

Surface: `--execute --bottleneck --output --workers --resume`. There is
deliberately no bounded, smoke or frame-limiting option: every pass measures all
3,345 registered validation frames.

`main` resolves `require_token_agrees_with_bottleneck` **first**, before
`torch.cuda.is_available()` and before any directory is created, so a
token/bottleneck mismatch costs nothing and leaves nothing behind.
`--bottleneck 128` is refused by argparse `choices` and again by
`require_phase10_bottleneck`; `AE128` is not constructible anywhere in this
module, and the Phase-9D and Phase-10A tokens are refused as firmly as a family
mismatch.

## Everything family-dependent is derived from `--bottleneck`

There is one implementation, not an AE64 one and an AE32 one. Every
family-dependent quantity is a function of the single `--bottleneck` argument,
and every emitted string is passed through
`ae_phase10_common.require_family_labelled`, which fails closed unless the text
names its own family and no other AE family:

| derived from `--bottleneck` | AE64 | AE32 |
| --- | --- | --- |
| family id | 2 | 3 |
| transported latent channels | 64 | 32 |
| range bytes per frame | 512 | 256 |
| analytical pre-zstd bytes, q=0 | 1,376,818 | 688,434 |
| analytical pre-zstd bytes, q=0.70 | 416,114 | 209,426 |
| routing tag | `0xdd7c5124` | `0xe2f86775` |
| execute token | `SPLITFUSION_AE64_PHASE10B_UINT8_VALIDATION` | `…AE32…` |
| terminal | `SPLITFUSION_AE64_PHASE10B_UINT8_VALIDATION_COMPLETE` | `…AE32…` |
| schema | `splitfusion_fcos_ae64_phase10b_uint8_validation_v1` | `…ae32…` |
| acceptance terminals | `AE64_UINT8_ZSTD_DEPLOYMENT_{ACCEPTED,NOT_ACCEPTED}` | `AE32_…` |
| result / report / manifest names | `phase10b_ae64_…`, `AE_PHASE10B_AE64_…` | `…ae32…`, `…AE32…` |

Payload accounting is asked of the committed transport rather than tabulated:
`analytical_size` and `range_byte_count` are `ae_uint8_transport`'s own functions
called with the family's bottleneck. Routing tags come from
`ae_contract.routing_tag_from_sha256` applied to the **full** selected-checkpoint
digest; the record states explicitly that the 32-bit tag is a decoder-routing
discriminator and authenticates nothing.

`ae_phase10_common.bind_process_family` enforces one family per process.

## Nothing scientific is restated

The runner composes the completed objects rather than re-expressing them, which
is what makes "the same gates and the same rule as AE128" checkable:

- six registered q, three primary q, two stress q, the twelve preservation
  gates, the nine absolute service gates, the registered 7/9 service baseline,
  the frozen noAE reference loader, the per-frame ranker counter, the byte
  statistics and `acceptance_inputs` are all imported from
  `ae_uint8_validation` (Phase 9D);
- the acceptance **rule text** is built by substituting the family name into
  `ae_uint8_validation.ACCEPTANCE_RULE` and is checked to carry this family's
  label, to carry no `AE128` label, and to differ from the AE128 string — so it
  cannot have been reworded here;
- the acceptance **evaluation** calls `ae_uint8_validation.evaluate_acceptance`
  and relabels only the decision terminal and the family identity;
- the preservation gates are scored by `ae_training_common.evaluate_same_q_gates`
  against the frozen noAE Phase-8B UINT8+zstd row at the same q, at the same
  bound digest Phase 9D used;
- no codec, decoder-selection or metric step is reimplemented: transmit is
  `ae_uint8_transport.encode_frame`, receive is
  `PreloadedAeDecoders.receive`, scoring is `score_validation_pass`.

## Binding and source-map policy

`bind_inputs(bottleneck)` binds the frozen perception checkpoint, the stable
epoch-4 ranker, the p025 forward lock and the hybrid-q locked configuration
through the existing authorized AE binding, unchanged, plus this family's three
Phase-10B artifacts by exact SHA-256:

| family | selected checkpoint | epoch | holdout decision |
| --- | --- | --- | --- |
| AE64 | `…/20260903_phase10_ae64_training/checkpoints/ae64_epoch_12.pt`<br>`dd7c5124…c838aa8e0` | 12 | `…/holdout_selection_ae64/ae64_holdout_selection.json`<br>`0d2fe444…f5ff87c87e` |
| AE32 | `…/20260903_phase10_ae32_training/checkpoints/ae32_epoch_08.pt`<br>`e2f86775…40693d1f271` | 8 | `…/holdout_selection_ae32/ae32_holdout_selection.json`<br>`e3dfbfb7…4b0d271afac34` |

All four digests were verified against the artifacts on disk during
implementation. Both checkpoint and decision paths are *derived* from the
family's own registered filename helpers (`candidate_filename`,
`holdout_selection_dirname`, `holdout_report_filename`) rather than spelled out
a second time.

`load_holdout_decision` verifies the decision selected that exact
epoch/checkpoint: its schema and terminal are this family's, its scope declares
this family and reports no validation/test access, no training and no deployment
validation, its transport is the FP32 holdout quantizer over 3,284 frames, its
`selected_epoch` equals the bound epoch, its recorded candidate hash for that
epoch equals the bound checkpoint digest, it disclaims being a service-ready
decision, and it bound the same four frozen artifacts. Its own completion marker
holds the sha256 of the report the selector had just written, and that digest
must equal the bound one — so the selector signed the document Phase 10B reads.

`require_selected_bindings` enforces every field of
`ae_training_common.binding_fields` bit-identically, iterated from that function
so none can be silently forgotten. The one exception is the AE package's own
source map, enforced as a declared delta by `ae_package_source_delta`, which is
**stricter than Phase 9D**:

- every recorded file that defines what the saved tensors mean
  (`ae_contract.py`, `ae_model.py`, `ae_phase10_common.py`) or what transport is
  being measured (`ae_composition.py`, `ae_loss.py`, `ae_uint8_transport.py`,
  `ae_family_dispatch.py`, `__init__.py`) must be byte-identical;
- **any** changed recorded file fails (`changed_files_allowed: 0`) — Phase 9D
  allowlisted three, Phase 10B allowlists none;
- **any** removed recorded file fails;
- additions are restricted to exactly the three registered Phase-10B files.

Both selected checkpoints recorded a 26-file source map, and both match the live
tree exactly today, so the only delta a run will see is this phase's own
additions.

## Deployment path measured

Exactly the established path, per frame, with no shortcut:

```
original FP32 C2
  -> selected family encoder (always, complete frame)
  -> ranges from the complete latent -> per-channel UINT8
  -> stable per-frame top-K for q>0 -> family-labelled sparse wire
  -> mandatory zstd level 1 -> received raw bytes
  -> exactly one decompression
  -> decoder selected from header family/bottleneck/routing tag
  -> dequantization and zero scatter -> selected family decoder
  -> unchanged perception tail and p025 service policy
```

Only the received bytes are handed to `receive`; the local packet object is
never passed, so the decoder can only be discovered from the header.

## Per-frame integrity

`_transport_one` refuses the frame unless all of the following hold, so a
violation produces no durable record at all rather than a recorded bad row:

- the received header declares this family id, this latent width and the bound
  routing tag, and the `ae_latent_uint8` codec — each read off the wire and
  compared against the value derived from `--bottleneck`;
- the header q and keep count match the plan, and the retained UINT8 block is
  `[keep, B]`;
- the selected indices equal the transmitted cells exactly (`torch.equal`);
- every dropped latent cell is exactly zero across all B channels before the
  decoder runs, and the drop cardinality matches;
- exactly one zstd decompression per frame, counted by `CountingWireCodec`;
- q=0 invokes the ranker zero times but still invokes the AE, and no q produces
  an identity reconstruction;
- for q>0 the ranker was handed the *original FP32 C2 tensor object*, checked by
  `data_ptr` identity, not a latent, copy or quantized tensor;
- pre-zstd payload equals the family's analytical size, and the range-byte
  accounting matches;
- all model, postprocess and p025 outputs are finite;
- frozen perception, stable ranker and AE state are unchanged, snapshotted and
  re-checked after every pass and again before finalization.

The pass then aggregates: one family id, one latent width, one routing tag, one
keep count and one payload size across all 3,345 frames, and exactly one
decompression per frame.

## Evaluation and the primary result

3,345 registered validation frames, exactly once per q, for
q ∈ {0, 0.30, 0.50, 0.70, 0.90, 0.98}. Each row is compared with the frozen noAE
UINT8+zstd result at the **same** q, so the reported degradation isolates the AE
latent transport. Reported per q: canonical-p025 person metrics (diagnostic),
AVO≥0.65 person metrics, vehicle detection and localization, all three
segmentation metrics, the twelve preservation gates with exact degradations and
bounds, and the nine absolute service gates.

The preregistered acceptance rule is applied verbatim: q=0 must pass 12/12 and
retain ≥ 7/9 service gates; at least one of q ∈ {0.30, 0.50, 0.70} must pass
12/12 without reducing the noAE same-q service count; q=0.90 and q=0.98 are
stress profiles and cannot make or break acceptance.

**A failed family acceptance suppresses no measured q row.** Every completed q
gets its durable record, appears in the curve, the CSV and the report tables, and
`finalize` asserts `rows_suppressed_by_failed_acceptance == 0` and that all six
registered q are reported regardless of the decision.

## Durability

The proven Phase-9D semantics, unchanged in order and rationale:

1. atomically write (fsync + rename + directory fsync) `settings/<q>.json` — the
   scientific completion record;
2. remove `working_predictions/<q>`;
3. atomically write `cleanup/<q>.json`, binding that setting's sha256.

An interruption can therefore lose at most a scratch prediction directory, never
a completed measurement. After writing, the record is immediately re-read through
the *same* validator a resume uses, so the in-memory row is exactly the durable
bytes.

`--resume` requires an existing run directory and a manifest whose run identity
is bit-identical; it refuses to rewrite a completed run. A valid durable q is
never rerun — at most its interrupted cleanup is finished, with no inference.
An invalid record is **refused, not overwritten and not re-measured**: the test
asserts the damaged bytes are still on disk after the refusal. A fresh run
without `--resume` refuses to start in a directory that already holds files or
completed records. Finalization requires all six settings *and* all six cleanup
markers.

The run identity covers the family, the scope, the selected checkpoint and
decision digests, the noAE reference digest, the routing tag, the primary rule,
the gate counts and baseline, the **whole secondary registration** (both
threshold sets, the provenance wording, the tier list, the install rule, the
service-ready rule and the masking policy), every frozen binding, the named
runner sources and the runner's own sha256. Because the secondary thresholds sit
inside the identity, they cannot be moved between an interrupted run and its
resume.

## Secondary prospective classification

Registered in source before any Phase-10B number exists, and applied verbatim.
It is computed **from** the already-evaluated primary result, never recomputed
alongside it, so it cannot disagree with the primary rule about whether the
primary rule passed.

Tiers, evaluated as a cascade per profile:

1. `FULL_PRESERVATION` — the existing primary rule passes at this profile;
2. `LOCALIZATION_PRIORITY` — the absolute AVO/object requirements all pass;
3. `EMERGENCY_ONLY` — transport/output integrity passes but the object
   requirements fail;
4. `INVALID` — transport, routing, numerical or execution failure only.

q=0.90 and q=0.98 are `EMERGENCY_ONLY` regardless of measured metrics; the test
pins this with a stress row that passes 12/12 and still does not promote.

Absolute object requirements (segmentation deliberately excluded):

`vehicle_precision ≥ 0.80`, `vehicle_recall ≥ 0.85`, `vehicle_xy_mae_m ≤ 1.00`,
`person_avo_precision ≥ 0.70`, `person_avo_recall ≥ 0.70`,
`person_avo_f1 ≥ 0.70`, `person_avo_xy_mae_m ≤ 1.20`,
`person_avo_recall_20_40m ≥ 0.70`.

Canonical-p025 person metrics are recorded as diagnostics with
`used_for_localization_priority_classification: false`; the classification uses
the AVO≥0.65 visible-object person view.

Threshold provenance is recorded verbatim as:

> Holdout-informed thresholds frozen before AE64/AE32 held-out deployment
> validation. The validation frames were not used for AE training or checkpoint
> selection.

with `independent_test_set_confirmation: false` and
`untouched_test_set_confirmation: false`. This is **not** described as an
untouched or independent test-set confirmation anywhere.

### Segmentation installability — separate from 12/12

```
segmentation_installable =
    vehicle_iou >= 0.85 and person_box_mask_iou >= 0.50 and foreground_miou >= 0.675
```

The three segmentation outputs do not enter the localization-priority
classification but are measured and reported in every artifact. A 12/12 relative
preservation result does **not** authorize replacing the spatial-map segmentation
layer: each profile records
`twelve_of_twelve_preservation_authorizes_install: false` and an explicit action,
either `install_new_segmentation` or
`retain_previous_segmentation_layer_with_original_timestamp`. Nothing is
installed by this runner
(`segmentation_layer_installed_here: false`).

### `SERVICE_READY` — separate again

`SERVICE_READY` is reported only when all nine absolute service gates pass, with
`derived_from_relative_preservation: false`. It is never produced from a 12/12
relative preservation result.

### `STATE_INFEASIBLE` and masking

`STATE_INFEASIBLE` is registered and defined as a *runtime availability* verdict
for an action a hard state-dependent resource constraint makes unavailable in the
current state. It is therefore not assignable by this offline validation — every
registered q is measured — and the runner records
`state_infeasible.assignable_by_this_validation: false` with a tier count of 0.
Every profile records `masked: false` and
`masked_for_perception_degradation: false` under the policy: perception
degradation changes tier and reward, never availability; only technical
invalidity or a hard state-dependent resource constraint may mask an action.

### Scope guarantees

The secondary block records, and `finalize` asserts the separation of, the
following: it changed no checkpoint selection, no primary acceptance terminal, no
threshold, NMS setting, model or scorer, and it neither erases nor reinterprets
any original preservation failure. The primary twelve-gate result and the
family-level acceptance rule are untouched by it.

### Registration-time feasibility of the absolute bars

`reference_feasibility` applies the registered classification to the **frozen
noAE** rows, from an already-published document, so it contains no Phase-10B
measurement. It is included so the bars' difficulty is visible at registration
rather than discovered afterwards. On the frozen noAE UINT8+zstd path:

| q | object reqs passed | failing | segmentation installable |
| ---: | ---: | --- | ---: |
| 0.00 | 7/8 | `person_avo_recall_20_40m` | true |
| 0.30 | 5/8 | + `person_avo_f1`, `person_avo_precision` | true |
| 0.50 | 5/8 | same three | false |
| 0.70 | 5/8 | same three | false |
| 0.90 | 3/8 | + `person_avo_recall`, `vehicle_recall` | false |
| 0.98 | 3/8 | same five | false |

**Flag for review:** `person_avo_recall_20_40m ≥ 0.70` is missed at every q on
the reference path itself (0.5777 at q=0 against a 0.70 bar). Since the AE
transport is lossy relative to that path, `LOCALIZATION_PRIORITY` is in practice
unreachable for AE64 and AE32 at any q as registered. The bar was implemented
exactly as specified and is not adjusted here; this is a preregistration
observation about frozen published numbers, not a Phase-10B result.

## Testing

One new test file, two required areas, four CPU cases. No CUDA context is created
(each case asserts the process-global flag is where it found it), no dataset,
teacher shard or CARLA process is touched, and the two selected checkpoints are
opened only to read identity and source-map fields — their tensors are never
built into a model or moved to a device.

**Area 1 — family/token/artifact/routing separation.** Thirteen string emitters
are each checked to produce two distinct, self-labelled strings that survive
their own family-labelling guard; latent width, family id, range bytes and every
analytical payload size are checked as derived and as differing between AE64 and
AE32; the routing tag is checked to be the leading 32 bits of the full digest,
nonzero, and disclaimed as identity; AE128 is refused by every public emitter;
token/bottleneck mismatch, the Phase-10A tokens and the Phase-9D token are all
refused; the parser surface is pinned; one-family-per-process is checked. Then
per family: the bound checkpoint and decision exist at the bound digest, the
decision's marker records that digest, the decision selected the bound epoch and
recorded the bound hash, neither artifact is readable as the other family's, and
the source-map delta admits the registered additions while refusing a changed
semantics module, a changed ordinary recorded file, a removed file and an
unregistered addition.

**Area 2 — shared acceptance, secondary classification and durable resume.** The
primary scope is pinned to the reused AE128 objects; the rule text is checked to
be the AE128 rule with only the family name substituted; the accept, q=0-blocks
and stress-cannot-decide cases run identically for both families with
family-labelled terminals. The secondary registration is pinned in source
(including that the tier is `FULL_PRESERVATION` and not `FULL_PERCEPTION`, that
`STATE_INFEASIBLE` exists and is never assigned, and that no segmentation metric
enters the object set), and the classifier is exercised on six synthetic rows
built by the real writer from frozen reference numbers: stress q stay
`EMERGENCY_ONLY` while passing 12/12; a `FULL_PRESERVATION` row authorizes
neither a segmentation install nor `SERVICE_READY`; segmentation installability
flips on its own bound alone; a row that fails the relative rule but passes all
eight absolute requirements is `LOCALIZATION_PRIORITY`; the same row missing a
requirement is `EMERGENCY_ONLY` and is not masked; and a broken transport
declaration — not a quality shortfall — is the only thing that yields `INVALID`.
The durable-resume case reproduces the interruption window for both families: the
record is reused byte-for-byte, the only write is the cleanup marker, the scratch
predictions are gone, a second resume is a no-op, the marker belongs to one
family only, the record is unreadable by the other family, by a different
identity and for a different q, and five distinct kinds of damaged record are
each refused with the damaged bytes still on disk.

### What was run

- `python3 -m py_compile ae_phase10b_uint8_validation.py` — clean.
- The two new test areas — 4 cases, pass.
- The full existing AE regression suite together with them — **27 tests, OK**
  (23 pre-existing, 4 new).
- `git diff --check` — clean.
- Out-of-band (not committed as a test): the CSV writer produces 60 columns × 6
  rows and the report writer 117 lines on a synthetic six-q document, confirming
  every field name in the writers resolves. This mattered because a field-name
  error there would otherwise surface only at the end of a real six-pass run.

Two real defects were found by the tests during implementation and fixed: the
Phase-10B records were being validated against `ae_phase10_common`'s
`phase: "phase10a"` label (now a Phase-10B-specific `family_fields` /
`require_family_identity` pair, with the Phase-10A validator retained for the
Phase-10A artifacts this phase *consumes*), and a promotion fixture built from
the q=0.30 reference row could not reach `LOCALIZATION_PRIORITY` because that row
misses three object requirements, not one.

## Explicitly not done

Not executed: validation, inference, evaluation, training, tuning, selection,
checkpoint loading into a model, CUDA initialization, validation/test data
access, CARLA. Not changed: any threshold, NMS setting, calibration, scorer,
model, q value, codec, gate, the primary acceptance rule, checkpoint selection,
or any completed Phase-9C/9D/10A artifact. Not created: any Phase-10B run
directory or result.

---

## Amendment 001 — pedestrian operating range correction

Applied before Phase-10B validation ran; nothing above was remeasured, and no
Phase-10B number exists yet. Everything above this line records the contract as
originally registered and is left intact.

### Why

The "Flag for review" above stands: `person_avo_recall_20_40m >= 0.70` is missed
at every q on the frozen noAE reference path itself (0.5777 at q=0), so
`LOCALIZATION_PRIORITY` was unreachable as registered. The completed range-aware
person semantic-support feasibility study
(`..._person_p025_calibration_v1/RANGE_AWARE_SUPPORT_FEASIBILITY_V1.md`) then
tested whether long-range recall could be recovered instead of the bar being
moved. It did not corroborate on train-holdout episode 04, so **policies A/B/C
are not implemented and the frozen p025 perception path is unchanged.** What is
corrected here is only the range the tier gate is read on.

### The correction

Frozen pedestrian operating ranges:

- primary: `0 <= gt_distance_m < 30`
- extended diagnostic: `30 <= gt_distance_m <= 40`

In `LOCALIZATION_OBJECT_REQUIREMENTS`, `person_avo_recall_20_40m >= 0.70` is
replaced by `person_avo_recall_0_30m >= 0.70`. **The other seven object
requirements are unchanged**, as are the twelve preservation gates, the nine
service gates, the segmentation-installation rule, every checkpoint, codec,
threshold and q value, and the `EMERGENCY_ONLY` status of q=0.90/0.98.

`person_avo_recall_0_30m` is a plain sum of the 0-10, 10-20 and 20-30 m
`tp` / `fn` / `eligible_gt` counts the frozen AVO scorer already produces: no new
matching, scoring or inference logic. Those per-bin slices were previously
computed and discarded, so the durable per-q record now carries a
`person_range_stratified` block, validated on the resume path.

### The 30 m boundary is evaluation-only

It does not filter, suppress, relabel, rescore or otherwise change any runtime
detection. Deployment continues to emit every detection accepted by the frozen
p025 pipeline throughout its existing range. The boundary is expressed on
ground-truth distance, which exists only in the evaluator, so it is not
runtime-computable even in principle. This is recorded as a set of explicit
declarations that travel with every classification and are pinned by the test.
It is also what distinguishes this correction from the rejected policies A/B/C,
which would have gated on *predicted* radial distance at runtime.

### Still reported, never gated

Per-band recall for 0-10, 10-20, 20-30 and 30-40 m; the 20-30 m boundary band
separately, so the cumulative 0-30 m result cannot hide boundary behaviour;
30-40 m as extended-range stress; and the original 20-40 m recall for historical
comparison, which remains one of the twelve protected metrics and is
cross-checked to reproduce exactly.

**Per-band precision is not reported, because it is not derivable.** The frozen
AVO scorer publishes each distance bin as a recall slice (`eligible_gt` / `tp` /
`fn`) only; a false positive is not attributed to a range, and attributing one
would mean binning predictions by predicted distance — new matching logic this
correction does not introduce. The block records
`precision_by_range.available: false` with that reason rather than omitting it
silently. Aggregate AVO precision remains the precision gate.

### Registration-time feasibility

The frozen noAE reference document publishes the twelve protected metrics
without per-bin slices, so `person_avo_recall_0_30m` **cannot be evaluated on
those rows**. `reference_feasibility` records it as not evaluable and fails
closed, rather than reporting a fabricated miss: per q it now reports 7 evaluated
of 8 registered, with `not_evaluable_object_requirements:
["person_avo_recall_0_30m"]`. q=0 clears all seven evaluable requirements — the
superseded 20-40 m bar was the only one it missed.

### Provenance

> The 0-30 m primary operating range was selected from frozen noAE
> range-stratified analysis and literature context before Phase-10B AE64/AE32
> validation. The 30-40 m results remain reported as extended-range stress.
> Independent test-set confirmation has not been performed.

### What was run

- `python3 -m py_compile` on the runner and the test — clean.
- The existing Phase-10B test file, updated in place, no new test file — **4
  tests, OK**. No broader suite was run.
- `git diff --check` — clean.
- Out-of-band (not committed as a test): the CSV writer produces 66 columns × 6
  rows and the new report section renders on six synthetic rows, confirming every
  new field name resolves.

Validation was not run.
