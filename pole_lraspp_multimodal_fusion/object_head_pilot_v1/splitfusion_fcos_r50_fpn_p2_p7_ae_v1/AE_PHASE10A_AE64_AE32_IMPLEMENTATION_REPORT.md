# SplitFusion AE64 / AE32 — Phase 10A training and holdout-selection implementation

Implementation only, for review. **Neither runner was executed.** No checkpoint
was loaded, no CUDA context was created, no dataset, teacher-cache shard,
validation frame or test frame was read, nothing was trained, inferred, selected
or evaluated, and CARLA was not launched.

The completed AE128 Phase-9C/9D work is imported and reused, never edited. `git
diff HEAD` over every protected file is empty:
`ae_contract.py`, `ae_model.py`, `ae_composition.py`, `ae_loss.py`,
`ae_uint8_transport.py`, `ae_family_dispatch.py`, `ae_training_common.py`,
`ae_training.py`, `ae_holdout_selection.py`, `ae_uint8_validation.py` and every
AE128 artifact under `experiments/splitfusion_fcos_ae_v1/`.

## New files

| File | Role |
| --- | --- |
| `ae_phase10_common.py` | the one shared family-aware layer: tokens, terminals, schemas, filenames, per-family seed, inherited configuration, family AE/optimizer, atomic family checkpoints |
| `ae_phase10_training.py` | the AE64/AE32 trainer (fit frames only), one family per command and per process |
| `ae_phase10_holdout_selection.py` | the separate AE64/AE32 train-holdout evaluation and preregistered checkpoint selection |
| `tests/test_ae_phase10_family.py` | the three focused CPU tests |
| `AE_PHASE10A_AE64_AE32_IMPLEMENTATION_REPORT.md` | this report |

There is **one** implementation, not two copies. Every family-dependent quantity
is derived from the single `bottleneck` argument, so AE64 and AE32 cannot drift
apart and neither can borrow the other's artifact. AE128 is not constructible by
this phase at all: `require_phase10_bottleneck` admits 64 and 32 only.

## Four commands

```
python3 -m pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_ae_v1.ae_phase10_training \
  --execute SPLITFUSION_AE64_PHASE10_TRAINING --bottleneck 64 --output <new-empty-dir> [--resume]

python3 -m ...ae_v1.ae_phase10_training \
  --execute SPLITFUSION_AE32_PHASE10_TRAINING --bottleneck 32 --output <new-empty-dir> [--resume]

python3 -m ...ae_v1.ae_phase10_holdout_selection \
  --execute SPLITFUSION_AE64_PHASE10_HOLDOUT_SELECTION --bottleneck 64 --training <completed-ae64-run>

python3 -m ...ae_v1.ae_phase10_holdout_selection \
  --execute SPLITFUSION_AE32_PHASE10_HOLDOUT_SELECTION --bottleneck 32 --training <completed-ae32-run>
```

Suggested run directories, following the existing convention:
`experiments/splitfusion_fcos_ae_v1/<UTC-stamp>_phase10a_ae64_training` and
`..._phase10a_ae32_training`.

### Fail-closed selection of the family and the command

* `--execute` and `--bottleneck` are both required, both constrained by
  `argparse` choices, and then cross-checked: the token's family and the
  `--bottleneck` family must be identical or the command raises before anything
  else happens (verified live: `SPLITFUSION_AE64_PHASE10_TRAINING --bottleneck 32`
  exits on `execute token … names AE64 but --bottleneck 32 names AE32`, having
  created no directory and touched no CUDA).
* A selection token passed to the trainer, a training token passed to the
  selection runner, and either AE128 Phase-9C token passed to either command are
  all refused.
* `bind_process_family` binds one family per process for its whole life, so
  **AE64 and AE32 cannot be trained in the same process**.

## Nothing AE128-labelled is emitted for these families

Every emitted schema, execute token, terminal, checkpoint filename, report
filename and selection directory passes `require_family_labelled`: it must
contain its own family slug and must not contain any other registered AE
family's slug. The two families share no emitted name, and none of them collides
with an AE128 schema or with `ae128_epoch_NN.pt`.

| Artifact | AE64 | AE32 |
| --- | --- | --- |
| training token | `SPLITFUSION_AE64_PHASE10_TRAINING` | `SPLITFUSION_AE32_PHASE10_TRAINING` |
| selection token | `SPLITFUSION_AE64_PHASE10_HOLDOUT_SELECTION` | `SPLITFUSION_AE32_PHASE10_HOLDOUT_SELECTION` |
| training terminal | `SPLITFUSION_AE64_PHASE10_TRAINING_COMPLETE` | `SPLITFUSION_AE32_PHASE10_TRAINING_COMPLETE` |
| selection terminal | `SPLITFUSION_AE64_PHASE10_HOLDOUT_CHECKPOINT_SELECTED` | `SPLITFUSION_AE32_PHASE10_HOLDOUT_CHECKPOINT_SELECTED` |
| training schema | `splitfusion_fcos_ae64_phase10a_training_v1` | `splitfusion_fcos_ae32_phase10a_training_v1` |
| recovery schema | `splitfusion_fcos_ae64_phase10a_recovery_v1` | `splitfusion_fcos_ae32_phase10a_recovery_v1` |
| candidate schema | `splitfusion_fcos_ae64_phase10a_candidate_v1` | `splitfusion_fcos_ae32_phase10a_candidate_v1` |
| selection schema | `splitfusion_fcos_ae64_phase10a_holdout_selection_v1` | `splitfusion_fcos_ae32_phase10a_holdout_selection_v1` |
| candidate file | `checkpoints/ae64_epoch_{04,08,12}.pt` | `checkpoints/ae32_epoch_{04,08,12}.pt` |
| recovery file | `recovery/ae64_recovery_epoch_NN.pt` | `recovery/ae32_recovery_epoch_NN.pt` |
| epoch summaries | `ae64_epoch_summaries.json` | `ae32_epoch_summaries.json` |
| training report | `ae64_training_report.json` | `ae32_training_report.json` |
| selection output | `holdout_selection_ae64/ae64_holdout_selection.json` | `holdout_selection_ae32/ae32_holdout_selection.json` |

Every checkpoint, summary, report and routing record additionally carries the
explicit family block `{phase, family, family_id, bottleneck, init_seed}`, and
`require_family_fields` refuses any artifact whose block is not this run's.

**Interpretation note.** Two AE128 mentions remain in Phase-10A output *by
design*, and neither labels an AE64/AE32 artifact: the frozen binding's
`ae_package_source_sha256` map lists every AE package file by path (provenance,
and it must stay complete), and `configuration_delta_from_ae128` records exactly
which locked-configuration keys this family re-derived from the completed AE128
configuration. Both are declarations of inheritance, not family labels.

## Scientific configuration: inherited, not restated

`training_configuration(B)` **copies** `ae_training_common.training_configuration()`
and overrides only three keys — `family`, `bottleneck`, `init_seed` — then
verifies that exactly those three differ, raising otherwise. A scientific
setting therefore cannot silently differ between the completed AE128 run and
these two families.

* AE64 seed `ae_contract.ae_init_seed(64)` = 20260893; AE32 seed
  `ae_contract.ae_init_seed(32)` = 20260861. The seed also seeds the process
  RNG (`torch`, `torch.cuda`, `random`, `numpy`), so the two families never share
  a draw. Sampler order is unaffected by this: `contract.epoch_shuffle_seed(epoch)`
  drives the per-epoch order and is family independent, so AE64 and AE32 see the
  identical frame order.
* 12 epochs; epochs 1–4 Stage A at q=0 and LR 1e-3; epochs 5–12 Stage B at
  LR 3e-4.
* Stage-B cycle `0, 0.30, 0.50, 0.70`, one q per batch, **cycle position carried
  across epoch boundaries**; 8 × 847 = 6,776 Stage-B updates, exactly
  **1,694 per q**, checked before the run starts (`require_balanced_stage_b`) and
  again against the realized per-q counts at the end.
* AdamW state preserved across the Stage-A→Stage-B transition; only the learning
  rate changes. Weight decay 1e-4, global gradient clipping 5.0, batch 16,
  `drop_last=False`, augmentation disabled.
* Objective: plain reconstruction plus cached combined-importance reconstruction
  (`ae_loss.task_aware_reconstruction_loss`, unchanged). No fake quantization and
  no zstd anywhere in training.
* Candidate epochs 4, 8, 12. q=0.90 and q=0.98 are never optimized;
  `ae_loss.require_optimization_q` guards every scheduled q.

## Data and isolation

* Exactly the registered 13,543 fit frames, every frame once per epoch, order
  seeded and hash-checked per epoch.
* The existing Phase-4 teacher cache is reused, never rebuilt: the manifest hash
  is bound, shard admission is decided **from manifest metadata before any
  deserialization**, and only fit shards are opened. Teacher records join by
  exact sample ID.
* A holdout sample ID entering an optimizer batch, or a holdout shard being
  opened, raises. Validation and test are never reachable
  (`require_no_validation_or_test`).
* Perception and the stable epoch-4 ranker are loaded frozen and in eval mode;
  their per-tensor and aggregate hashes are compared at every epoch boundary, and
  any gradient reaching a frozen parameter raises.
* The optimizer owns exactly the selected family's **eight** trainable tensors —
  checked by count, by identity, and by `ae_loss.require_ae_only_optimizer`
  before and after every step.
* Training refuses to start if either selection runner is in `sys.modules`
  (re-checked at every epoch and before the terminal) or if the run directory
  already holds any `holdout_selection*` output.

### Reuse rather than re-implementation

The family-independent epoch mechanics are imported from the proven AE128
trainer and are *not* copied: `optimizer_update` (the full per-update safety
contract), `EpochAccumulator`/`QAccumulator`, `TrainingState`, `order_identity`
and `expected_stage_b_position`. None of them emits a family label. The teacher
store, atomic write primitives, RNG capture/restore, schedule functions, frozen
loaders, registered preservation gates and the frozen noAE reference loader come
from `ae_training_common`. Importing `ae_training` is what makes "the same
procedure" checkable rather than merely claimed; it is a trainer, not a selection
runner, and it launches nothing on import.

## Recovery

The proven Phase-9C durability behaviour is retained exactly, per epoch and all
atomic: **1)** candidate checkpoint, **2)** the family epoch-summary file,
**3)** the recovery checkpoint last. `recovery/ae<B>_recovery_epoch_E.pt` is
therefore the sole durable declaration that epoch E completed, and it embeds that
epoch's final summary including its candidate name and hash.

`--resume` verifies every recovery checkpoint in full (schema, family block,
epoch, stage, all six source bindings, configuration, both counters, sampler
identity, embedded summary, candidate naming), restores weights, optimizer
state, all four RNG streams, the counters and the Stage-B cycle position from the
last one only, rebuilds the candidate and recovery inventories with hash checks,
rewrites the summary file from the canonical record, and **replays no completed
epoch**. A recovery directory holding another family's checkpoints is refused
outright rather than parsed.

## Holdout selection

A separate module and command. It refuses to start unless the family's
`SPLITFUSION_AE<B>_PHASE10_TRAINING_COMPLETE` marker, `ae<B>_training_report.json`,
its terminal, its family block, its locked configuration and its three candidate
hashes all check out, and its output directory is create-only.

* Candidates 4/8/12 at q = {0, 0.30, 0.50, 0.70} — 12 passes, one per
  checkpoint/q pair, over the 3,284 reserved train-holdout frames.
* **FP32 latent reconstruction only.** No UINT8, no zstd, no UINT6/UINT4. The
  single-pass encode/drop/decode/serve routine is `ae_holdout_selection.run_pass`
  itself, reused unchanged.
* The same frozen noAE same-q holdout reference
  (`…/20260901_185725_phase5_ranker_training/holdout/holdout_evaluation.json`,
  sha256-bound) and the same `evaluate_same_q_gates` scorer/gate definitions
  AE128 used.
* The same preregistered ranking: `aggregate_by_checkpoint` and `ranking_key`
  are the AE128 objects, so the five criteria, their order, the normalized
  degradation definition and the batching-independent **float64 frame-summed**
  reconstruction tie-breaker cannot drift. Only the reported family identity and
  purpose text are this phase's own.
* No fit teacher shard is deserialized; validation and test are never opened.
* The report states explicitly that selecting a checkpoint is **not** a
  service-readiness claim.

**Phase-9D deployment validation is deliberately not implemented**, because the
selected AE64/AE32 checkpoint hashes do not exist yet. Nothing here derives a
routing tag, writes a wire frame or measures a byte count.

## Focused CPU tests

`tests/test_ae_phase10_family.py`, three tests, no CUDA / checkpoint / shard /
dataset access. AE128 fixtures (`registered_shape_partition`, `synthetic_binding`,
`cpu_rng_state`) are imported from the Phase-9C test file rather than copied.

1. **Token / bottleneck / family / schema / seed separation.** 128 and every
   malformed bottleneck refused; every emitted name family-labelled, mutually
   disjoint and disjoint from the AE128 schemas and filename; token↔bottleneck
   agreement in both directions and both commands; AE128 tokens refused; per-family
   seeds; the inherited configuration equal to AE128's on every key except the
   three family keys; one family per process.
2. **Family-specific checkpoint naming and optimizer ownership.** Eight trainable
   tensors owned by identity and nothing else, not even the other family's
   tensors (supplied as a frozen companion); candidate save/load round trip with
   the family schema and block; saving under another family's filename, loading
   as the other family, a source-binding edit and a non-candidate epoch all fail
   closed.
3. **Recovery and holdout isolation through the shared family-aware path.** An
   eight-epoch AE64 run written in the trainer's commit order resumes with both
   counters continued, all summaries and both candidate hashes rebuilt and no
   epoch replayed; the other family cannot resume from it; a present
   `holdout_selection*` directory stops training; importing either selection
   runner stops training (observed firing for real in this process); and the
   trainer's own source is parsed to prove it imports no selection runner.

## What was run

```
python3 -m py_compile   (the three new modules and the new test file)   OK
Ran 21 tests   OK    # the whole ae_v1 suite: 18 existing + 3 new
Ran 79 tests   OK    # the frozen hybrid-q suite, unchanged
git diff --check                                                        clean
```

Both command-line interfaces were exercised with `--help` and with one
deliberately mismatched token/bottleneck pair, which failed closed before CUDA,
before any file was created and before any data path was touched. Nothing else
was executed.

## Launch risk

1. **The AE package source map moves when files are added.** Every checkpoint
   binds `ae_package_source_sha256`, and `require_bindings` enforces it exactly.
   These five new files therefore change the live map. Consequences, in order of
   importance:
   * A Phase-10A run must not have any AE package file added, edited or removed
     between its start and its completion or resume, or `--resume` fails closed.
     Freeze the package for the duration of each run; the two family runs are
     independent of each other in this respect.
   * Re-running the completed **AE128 Phase-9C holdout selection** would now fail
     on this binding. It is complete and is not scheduled to re-run; its
     artifacts are unchanged.
   * **AE128 Phase 9D still re-runs cleanly**: it allowlists added files and
     requires only that the checkpoint- and transport-semantics modules be
     byte-identical, and this phase changed none of them.
2. **Selection output blocks a later resume.** Once `holdout_selection_ae<B>/`
   exists, the trainer refuses to run in that directory. This is intended; if a
   run genuinely has to be resumed after selection, the selection output must be
   moved aside deliberately.
3. **Resource sequencing.** Each command requires `cuda:0`, and one family per
   process means four sequential runs. On the RTX 5090 the AE128 equivalents took
   ~22 min (training) and ~54 min (12 selection passes); AE64 and AE32 are
   smaller, so budget roughly that or less per family. Host memory holds the
   ~1.08 GiB fit teacher store during training (~0.26 GiB holdout during
   selection); AE128 peaked at ~4.2 GiB allocated / ~4.6 GiB reserved VRAM, and
   the two smaller families sit below that.
4. **Unproven at runtime.** Nothing in this phase has been executed against real
   data or a GPU. The AE128 procedure it extends is proven, and the reused code
   paths are the proven ones, but the first real launch is still the first launch:
   expect to validate epoch 1 wall time, VRAM peak and the per-q loss report
   before letting a full 12-epoch run proceed unattended.

## Terminal

```
SPLITFUSION_AE64_AE32_PHASE10A_IMPLEMENTATION_READY_FOR_REVIEW
```

Running any of the four commands remains separately authorized.
