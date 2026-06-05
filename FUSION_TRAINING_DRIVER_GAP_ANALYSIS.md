# SceneSense Fusion Training Driver Gap Analysis

Last updated: 2026-06-04

Purpose: record what is already available for parked-ego fusion fine-tuning and
what is still missing before we can run an actual training job.

## Current Finding

The local `neu_collab` checkout contains the fusion model package, object-target
helpers, inference runtime, saved-data collector, dataset validator, and target
dry-run validator. It does not appear to contain a standalone SceneSense
RGB+radar fusion training driver or a concrete `FusionPoleMultiTaskDataset`
class.

Searches performed locally included:

- filename scan for `*train*.py`, `*training*.py`, `*finetune*.py`,
  `*dataset*.py`, and matching shell scripts
- source search for `FusionPoleMultiTaskDataset`, `MultiTaskFusionLRASPP`,
  `multitask_object_loss`, `build_object_targets`, `DataLoader`, `torch.optim`,
  `train_loader`, `object_heads`, and `learned_object`

The scan found V2Xverse/OpenCOOD training scripts, but those belong to the
V2Xverse/OpenCOOD stack and are not the current LR-ASPP RGB+radar fusion training
driver used by `checkpoints/fusion_object_best.pt`.

## Proven Parked-Ego Data Path

Smoke dataset:

```text
fusion_training_data/parked_ego_fusion_training_smoke_20260604
```

Schema validation passed:

- 30 manifest rows
- 474 actor-derived object rows
- 370 vehicle rows and 104 person rows
- RGB shape `(480, 854, 3)`
- mask shape `(480, 854)` with classes `0/1/2`
- radar tensor shape `(4, 432, 768)`

Training-target dry run passed:

- 30/30 samples build feature tensors with shape `(7, 432, 768)`
- 30/30 samples build segmentation targets with shape `(432, 768)`
- 30/30 samples contain positive vehicle object-head targets
- 369 valid vehicle objects become training targets
- object heatmap shape `(1, 432, 768)`
- object regression shape `(10, 432, 768)`
- GT object tensor shape `(64, 9)`
- 65 vehicle targets have radar-support evidence
- 5 vehicle targets are marked parked

Important caveat: the current object-head target helper consumes vehicle actor
rows only. Person rows are present and useful for segmentation/class balance
checks, but they are not object-head positives unless the target helper is
extended.

## Present Reusable Pieces

- `pole_lraspp_multimodal_fusion/model.py`
  - `MultiTaskFusionLRASPP`
  - LR-ASPP builder/adapters
  - checkpoint-compatible model structure
- `pole_lraspp_multimodal_fusion/object_targets.py`
  - `load_object_boxes`
  - `valid_vehicle_objects`
  - `build_object_targets`
  - `multitask_object_loss`
  - object decoding helpers
- `pole_lraspp_multimodal_fusion/common.py`
  - manifest/object schema fields
  - split/config helpers
- `carla_collect_parked_ego_fusion_training_data.py`
  - saved parked-ego RGB/mask/radar/object-label collection
- `scripts/validate_fusion_training_dataset.py`
  - file/schema/data-shape validator
- `scripts/dry_run_fusion_training_targets.py`
  - no-training target-construction validator

## Missing Pieces To Recreate Training

1. Dataset class:
   - read `manifest.csv`
   - load RGB, mask, radar tensor, and object rows
   - resize/normalize RGB and radar into 7-channel model input
   - resize mask into segmentation target
   - call `valid_vehicle_objects` and `build_object_targets`

2. DataLoader wiring:
   - train/val split filtering
   - batch collation for images, masks, object targets, and sample metadata
   - deterministic seed and worker settings

3. Model/checkpoint loading policy:
   - build the same `MultiTaskFusionLRASPP` architecture used by inference
   - load `checkpoints/fusion_object_best.pt`
   - decide whether to freeze the backbone, train only heads, or fine-tune all
     layers

4. Loss function:
   - segmentation loss, likely cross entropy over background/vehicle/person
   - object loss via `multitask_object_loss`
   - configurable loss weights matching or approximating the original run

5. Metrics:
   - segmentation mIoU, foreground IoU, vehicle IoU, person IoU
   - object positive count and center heatmap sanity
   - first-pass object recall/localization on validation rows if feasible

6. Training loop:
   - epoch/batch loop
   - optimizer and scheduler
   - AMP or plain FP32 option
   - checkpoint save/resume
   - JSON/CSV logs

7. Output checkpoint format:
   - must save model state and metadata in a format accepted by the existing
     inference loader
   - include model input size, class count, object-head settings, and selection
     metric

## Recommended Next Engineering Step

If no remote-only original trainer is found, create a minimal smoke trainer
rather than a full retraining pipeline:

- train on the 23 train rows
- validate on the 4 val rows
- run 1 epoch or a fixed small number of batches
- default to loading `checkpoints/fusion_object_best.pt`
- start with low learning rate and a head-only option
- write outputs under `experiments/parked_ego_fusion_finetune_smoke_*`

The goal of that smoke trainer is only to prove that backprop, losses,
checkpoint loading, and checkpoint saving work on the parked-ego dataset. It is
not yet a model-quality experiment.
