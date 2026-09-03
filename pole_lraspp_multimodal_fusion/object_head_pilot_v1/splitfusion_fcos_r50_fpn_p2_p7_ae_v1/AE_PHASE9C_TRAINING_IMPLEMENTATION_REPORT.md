# SplitFusion AE128 — Phase 9C training and holdout-selection implementation

Implementation only, for review. **No checkpoint was loaded, no CUDA was used,
no dataset or teacher-cache shard was read, and nothing was trained, inferred or
evaluated.** Neither runner is launched here, and neither launches the other.

The AE architecture, the AE loss, the AE wire format, the ranker, the perception
model, the p025 scorer and every frozen hybrid-q file are imported and reused,
never edited (`git diff HEAD` over the hybrid-q package is empty).

> **Revision, 2026-09-02 — four review corrections.** Split-pure teacher
> loading, exact-resume bookkeeping, full source-binding enforcement and a
> batching-independent holdout tie-breaker. See
> [Review corrections](#review-corrections-2026-09-02). Nothing in the AE
> architecture, objective, optimizer, schedule, q cycle, checkpoint epochs,
> scoring, thresholds or ranking criteria changed.

## New files

| File | Role |
| --- | --- |
| `ae_training_common.py` | locked configuration, exact schedule, sample-id-keyed teacher store, atomic checkpoint format, frozen-noAE reference loader |
| `ae_training.py` | the scientific AE128 trainer (fit frames only) |
| `ae_holdout_selection.py` | holdout evaluation and the preregistered checkpoint ranking |
| `tests/test_ae_training_schedule.py` | the focused CPU tests (one file, extended by the review corrections) |

## Two separate commands

```
python3 -m ...ae_v1.ae_training \
  --execute SPLITFUSION_AE128_PHASE9C_TRAINING --output <run_dir> [--resume]

python3 -m ...ae_v1.ae_holdout_selection \
  --execute SPLITFUSION_AE128_PHASE9C_HOLDOUT_SELECTION --training <run_dir>
```

Distinct modules, distinct execute tokens, no automatic launch. The trainer
additionally calls `require_holdout_unopened()` at start-up, at every epoch and
before writing its terminal: if `ae_holdout_selection` is so much as present in
`sys.modules`, training refuses to run. The selection command refuses to start
unless `<run_dir>/TRAINING_COMPLETE` exists, the training report's terminal is
`SPLITFUSION_AE128_TRAINING_COMPLETE`, and the training configuration is
byte-equal to the locked configuration.

## Scientific training configuration

AE128 only; AE64 and AE32 are out of scope. The AE is built by the committed
deterministic `build_split_feature_ae(128)` (init seed 20260957 = 20260829+128).
The frozen perception model and the stable epoch-4 ranker are loaded, put in
eval mode with `requires_grad=False`, and their gradients cleared; the optimizer
is built through the frozen `training.build_ranker_optimizer` and then re-checked
by `ae_loss.require_ae_only_optimizer`, so it owns exactly the eight named AE
tensors and nothing else.

| Setting | Value |
| --- | --- |
| Optimization split | 13,543 registered fit frames |
| Selection split | 3,284 reserved train-holdout frames (selection command only) |
| Validation / test | never opened |
| Augmentation | off |
| Batch | 16, `drop_last=False`, final short batch retained |
| Batches per epoch | **847** (the last carries 7 frames) |
| Epoch order | `torch.randperm` at seed **20260829 + epoch**, every fit frame exactly once |
| Epochs 1–4 | Stage A, q=0 only, LR **1e-3** |
| Epochs 5–12 | Stage B, continuous round robin over q = 0.00/0.30/0.50/0.70, LR **3e-4** |
| Optimizer | AdamW, weight decay 1e-4, state preserved across the stage transition (only `group["lr"]` changes) |
| Gradient clip | global norm 5.0 (the frozen `clip_ranker_gradients`) |
| Candidate checkpoints | epochs 4, 8, 12 |

Augmentation is off because the Phase-4 importance maps were produced on the
unaugmented frames; an augmented frame would be supervised by the map of a
different image. The trainer asserts `dataset.augment is False`.

### The Stage-B cycle carries across epoch boundaries

847 is not a multiple of four, so restarting the cycle each epoch would start
every Stage-B epoch at q=0.00 and unbalance the totals. A single
`stage_b_position` counter is incremented per Stage-B update and carried through
epoch boundaries and through resume, so:

- Stage-B updates = 8 × 847 = **6,776**
- per-q updates = 6,776 / 4 = **1,694** for each of 0.00, 0.30, 0.50, 0.70

`require_balanced_stage_b()` recomputes this from the schedule before the run
starts and the trainer re-checks the *realized* counts against it after epoch 12.
q=0.90 and q=0.98 are never scheduled and `ae_loss.require_optimization_q`
refuses them.

### Objective and data join

The committed `ae_loss.task_aware_reconstruction_loss` is used unchanged:
`total = plain reconstruction + combined-importance reconstruction`. Only the
cached combined importance map and the `valid_groups`/`excluded_groups` metadata
are consumed — no per-group maps, no invented D/G/S/A losses, no fake
quantization and no zstd anywhere in training.

`AeTeacherStore` holds one split and **deserializes only that split's shards**
(see [Review corrections](#review-corrections-2026-09-02)). The trainer loads it
with `split="fit"`, so a holdout map is never read, let alone held. Batches are
joined to cached records by exact sample ID (`store.batch(sample_ids)`), never by
position, and each epoch additionally re-checks the observed sample-id sequence
against the seeded order, full fit coverage, and the empty intersection with the
holdout ids.

### Runtime safeguards, per update and per epoch

Per update: finite total loss before backward; finite per-tensor gradients and
finite pre/post-clip global norms; `require_post_step_health` (finite AE
parameters and finite tensor-valued AdamW state); optimizer ownership re-checked;
and `.grad is None` asserted on every frozen perception and ranker parameter.

Per epoch: `GradientQualification` with a window of 847 requires every named AE
tensor to receive a finite, **nonzero** gradient at least once in the epoch — not
on every batch; isolated zero-gradient batches are counted and reported, never
treated as failures. Both frozen modules are then compared against their
snapshots *and* against their per-tensor and aggregate sha256 hashes recorded
before epoch 1.

Losses are recorded separately by stage and by q (`per_q` with per-q update
count, mean total, mean plain, mean combined-importance and worst batch total).
Each epoch summary carries an explicit note that a Stage-A aggregate and a
Stage-B aggregate are not comparable, and `clipping_is_not_a_failure: true` is
recorded alongside the clip count, clip fraction and max pre-clip norm.

### Checkpoints

Every epoch writes `recovery/epoch_NN.pt` through a write-beside → `fsync` →
`os.replace` → directory-`fsync` sequence. The payload carries everything an
exact resume needs: AE `state_dict`, AdamW `state_dict`, epoch and next epoch,
global update index, **Stage-B cycle position** and the next q it implies, torch
/ CUDA / Python / NumPy RNG, the sampler/order identity (seed, batch size,
drop_last, batch count, sample-id digest), the full locked configuration, and
the source bindings. Resume refuses on any binding or configuration drift.

Candidate checkpoints are exactly `ae128_epoch_04.pt`, `ae128_epoch_08.pt`,
`ae128_epoch_12.pt`. After epoch 12 the trainer verifies the total update count
(12 × 847 = 10,164), the Stage-B position (6,776), the realized per-q counts and
the candidate set, writes `training_report.json` plus `TRAINING_COMPLETE`, and
**stops without opening the holdout**.

## Holdout-selection runner (implemented, not executed)

Exactly 3 candidate epochs × 4 q values = **12 passes**, one holdout
inference/evaluation pass per checkpoint/q pair, over the 3,284 reserved frames.

Transport is **FP32 AE latent reconstruction only**: the ranker scores the
original FP32 C2, the AE encoder runs on the complete frame, the keep mask drops
cells, the AE decoder reconstructs, and the complete unchanged frozen perception
tail plus the unchanged p025 service policy and AVO scoring run on the result.
No UINT8, zstd, UINT6 or UINT4 at this phase. No threshold, calibration, NMS or
visibility setting is touched: `score_pass` and the frozen scorers are imported
from the Phase-5 holdout module and used verbatim.

Reported per pass: all vehicle and person-AVO precision/recall/F1, vehicle and
person XY MAE, vehicle IoU, person box-mask IoU, foreground mIoU, the p025
service outputs, person AVO recall in 20–40 m, and the reconstruction losses
(the global task-aware loss with its exact numerators and denominators, the
per-batch means kept as diagnostics, plus per-frame plain and
importance-weighted error median/p95/max over all 3,284 frames).

### Comparison baseline

Each AE result is compared with the **frozen noAE result at the same q**, read
from the completed Phase-5 holdout evaluation and bound by hash:

```
experiments/splitfusion_fcos_hybrid_q_v1/20260901_185725_phase5_ranker_training/
  holdout/holdout_evaluation.json
sha256 b86df9aeea6a9d5bb269f7a9d1f185f6bd4a93ffe87d3dd04a0ade1b15922717
```

q=0 uses that run's ranker-free q=0 baseline; q=0.30/0.50/0.70 use its rows for
ranker epoch 4 — exactly the stable epoch-4 ranker every AE pass uses. Nothing
is recomputed, and the loader fails closed on schema, frame-count, q-coverage or
protected-metric-set drift. Same-q comparison isolates the AE's contribution
instead of re-charging it for the ROI drop the noAE path already pays.

### Preregistered ranking

Applied verbatim, in order, over the three candidates:

1. maximize the **minimum** number of same-q preservation gates passed across
   the four q settings;
2. then maximize the **total** gates passed across all four;
3. then minimize the **worst normalized** protected-metric degradation —
   normalized means the signed degradation divided by that metric's registered
   gate bound, so the 0.01, 0.015, 0.03 and 0.05 bounds are comparable;
4. then minimize the **mean holdout task-aware reconstruction loss** over the
   four q, where each q's value is the *global* loss — a total numerator over a
   total denominator accumulated across every holdout frame, not an unweighted
   mean of per-batch ratios;
5. then prefer the **earlier** epoch.

There are 12 registered preservation gates, so each q contributes 0–12. The rule
is a total order by construction (criterion 5 is unique), so it always names one
checkpoint; the report also records `decided_at_criterion`, the full ordering,
and `all_gates_passed_at_every_q` per candidate.

**Selecting a checkpoint is not a service-ready claim.** It chooses the one
AE128 checkpoint that later deployment-path UINT8+zstd validation will use.
`selection_is_a_service_ready_claim` is recorded as `false` in the report.

## Focused CPU tests

1. `test_stage_schedule_and_balanced_stage_b_q_counts` — stage boundaries and
   both learning rates, out-of-range epochs refused, batch 16 with the final
   7-frame batch retained, 847 batches/epoch, then a full walk of all 10,164
   updates asserting Stage A is q=0 only, 6,776 Stage-B updates, exactly 1,694
   per q, that the cycle demonstrably carries across epoch boundaries
   (epoch 6 starts at `cycle[847 % 4]`, not at q=0), and that 0.90/0.98 are never
   scheduled and are refused by `require_optimization_q`.
2. `test_preregistered_checkpoint_ranking_is_deterministic` — synthetic records
   that isolate each of the five criteria in turn, assert the winner and the
   reported `decided_at_criterion` at every level, assert input order does not
   matter, and assert an incomplete or duplicated q sweep fails closed.

The review corrections add nine more checks to the same file; they are listed
with the correction they cover below.

<a id="review-corrections-2026-09-02"></a>

## Review corrections (2026-09-02)

Four focused corrections, applied before any training is authorized. No
architecture, objective, optimizer, schedule, q-cycle, checkpoint-epoch,
scoring, threshold or ranking-criterion change; no checkpoint or real data was
loaded, no CUDA context was created, and neither runner was launched.

### 1. Split-pure teacher loading

`load_ae_teacher_store` previously walked all 66 shards and `torch.load`ed every
one of them, discarding the maps of the split it was not for. Reading the tensor
payload of a holdout shard inside the trainer is exactly what the split is meant
to prevent, so the discard came too late.

The split decision now happens before anything is deserialized, in the new pure
`plan_teacher_shards(cache_root, partition, split)`:

- the complete manifest is read for provenance and its sha256 is checked against
  `contract.TEACHER_CACHE_MANIFEST_SHA256`, along with its schema, terminal,
  perception-checkpoint binding, `validation_or_test_frames == 0`, entry count
  and total frame count;
- every entry must be split-pure by its own manifest counts
  (`fit_frames`/`holdout_frames`), and only entries whose manifest `split` equals
  the requested split are admitted;
- `other_split_ids` is derived from the registered `SplitPartition`, never by
  opening the opposite split, and both splits' exact sample-ID coverage is
  reconciled against the manifest's `fit_sample_id_sha256` /
  `holdout_sample_id_sha256` digests and frame counts.

`load_ae_teacher_store` then verifies each admitted shard's byte size and sha256
**before** calling the loader, and re-checks split purity in the payload itself
(`split` and every entry of `splits` must equal the requested split, and no
registered opposite-split id may appear). Manifest and hash inspection is
provenance and is retained in full; deserializing an opposite-split tensor
payload is what can no longer happen. Training with `split="fit"` never opens a
holdout shard, and holdout selection with `split="holdout"` never opens a fit
shard.

Reporting changed accordingly: the trainer now prints *"N holdout IDs excluded,
M holdout shards never opened"* rather than "holdout maps withheld", and both
run records carry `store.provenance()` with
`holdout_shards_deserialized: 0` / `holdout_maps_loaded: 0` (and the mirrored
`fit_teacher_shards_deserialized: 0` on the selection side).

### 2. Exact-resume bookkeeping

`--resume` previously restored only the newest recovery checkpoint and started
with empty `recovery_hashes` and `candidate_hashes`, so a run resumed after
epoch 4 or 8 would fail the final candidate-set check even though the epoch-4
and epoch-8 candidates existed on disk.

`restore_completed_epochs()` now rebuilds the whole record:

- `--resume` refuses a missing output directory, a missing `recovery/`, an empty
  `recovery/`, a missing `epoch_summaries.json` and a non-contiguous recovery
  sequence;
- `completed_recovery_epochs()` parses each filename strictly and `read_recovery`
  requires the filename epoch to equal the embedded epoch;
- each completed epoch's checkpoint is verified for schema, stage, bottleneck,
  every binding, the locked configuration, `global_update_index == epoch × 847`,
  the implied Stage-B cycle position, the sampler/order identity and its own
  epoch summary;
- `epoch_summaries.json` must hold exactly epochs 1..completed with no duplicate
  or missing epoch, and each entry's order digest must match its recovery
  checkpoint;
- `recovery_hashes` and `candidate_hashes` are reconstructed from every already
  completed file after SHA-256 and metadata verification, with each candidate
  additionally re-loaded through `load_candidate`;
- only the last recovery checkpoint restores weights, optimizer state, counters
  and RNG — no completed epoch is replayed.

### 3. Every saved source binding is enforced

`load_candidate` skipped any field ending in `_source_sha256`, and
`load_recovery` compared only four hard-coded names. Both now call the new
`common.require_bindings()`, which enforces every field
`common.binding_fields()` writes — including the per-file
`hybrid_q_source_sha256` and `ae_package_source_sha256` maps. Any mapping, file
or configuration drift between writing a checkpoint and loading it fails closed.

### 4. Batching-independent holdout tie-breaker

The committed task-aware loss is a ratio of sums, so an unweighted mean of
per-batch ratios depends on how the frames were grouped and over-weights the
short final batch. Criterion 4 was ranking on exactly that number.

`HoldoutReconstructionTotals` now accumulates, per frame and in float64, the
plain squared-error numerator, the plain reference-energy denominator, the
combined-importance numerator and the combined-importance reference-energy
denominator, and the pass reports

```
global_plain    = plain_numerator    / plain_denominator
global_combined = weighted_numerator / weighted_denominator
global_total    = global_plain + global_combined
```

with all four exact sums recorded. Criterion 4 — unchanged as a criterion, and
still fourth — now ranks on `global_total_loss`. The per-batch means survive as
`reconstruction.batch_diagnostics` with an explicit note that they are not used
for selection, and the per-frame summaries are unchanged.

### Added coverage

| Check | Correction |
| --- | --- |
| `test_fit_never_invokes_the_shard_loader_on_a_holdout_entry` | 1 |
| `test_a_shard_that_is_not_split_pure_is_refused_before_any_load` | 1 |
| `test_manifest_sample_id_coverage_must_match_the_registered_partition` | 1 |
| `test_resume_after_epoch_eight_reconstructs_recovery_and_candidate_hashes` | 2 |
| `test_resume_refuses_an_empty_or_new_directory_and_a_broken_record` | 2 |
| `test_every_saved_source_binding_is_enforced_on_load` | 2, 3 |
| `test_global_loss_is_unchanged_under_regrouping` | 4 |
| `test_a_single_batch_reproduces_the_committed_loss_exactly` | 4 |
| `test_the_unweighted_mean_of_batch_ratios_is_the_number_that_moves` | 4 |

The split-purity check runs the production call path — a spy replaces
`teacher_cache.load_shard` on the module and no loader is injected — over a
synthetic 66-entry manifest whose two fit shards exist on disk and whose 64
holdout shard files are deliberately never created, so "no holdout shard was
opened" is enforced by the filesystem as well as by the spy. The resume check
writes epochs 1..8 with the real `save_recovery`/`save_candidate`, resumes with a
fresh AE and optimizer, and asserts the reconstructed hashes equal the written
ones and that adding epoch 12 satisfies the trainer's final candidate-set check.
The tie-breaker check regroups the same 12 synthetic frames into `(12)`,
`(8,4)`, `(5,5,2)`, `(1)×12` and `(7,1,3,1)` and asserts the four sums and three
ratios are bit-identical, and separately that the discarded per-batch mean does
move. `torch.cuda.is_initialized()` is `False` after the whole file runs.

## What was run

`py_compile` over the package, the focused Phase-9C tests, the existing test
suites as a regression check, and `git status` / `git diff` checks.

```
Ran 93 tests   OK      # 82 pre-existing + 11 focused Phase-9C checks
```

The only file this implementation phase read is the frozen noAE holdout summary
JSON above, whose sha256 had to be computed to be bound as a constant. It is an
aggregate result document, not a dataset, cache shard or checkpoint.

## Terminal

```
SPLITFUSION_AE128_TRAINING_IMPLEMENTATION_READY_FOR_REVIEW
```

Running either command remains separately authorized.
