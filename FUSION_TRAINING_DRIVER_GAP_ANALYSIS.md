# SceneSense Fusion Training Driver Gap Analysis

Last updated: 2026-06-11

Purpose: record what is already available for parked-ego fusion fine-tuning and
what is still missing before we can run an actual training job.

## Current Finding

Superseded by the 2026-06-11 local inspection. The earlier 2026-06-04 scan was
done before the supervisor training folders were copied locally.

The local `abiodun/` checkout now contains two relevant training workflows:

- `pole_lraspp_training/`: RGB-only CARLA LR-ASPP segmentation training and
  evaluation.
- `pole_lraspp_multimodal_fusion/`: RGB+radar early-fusion LR-ASPP
  segmentation plus learned localization/object-head training and evaluation.

The multimodal workflow includes:

- `pole_lraspp_multimodal_fusion/run_pipeline.py`
- `pole_lraspp_multimodal_fusion/collect_dataset.py`
- `pole_lraspp_multimodal_fusion/train_fusion.py`
- `pole_lraspp_multimodal_fusion/evaluate_fusion.py`
- `pole_lraspp_multimodal_fusion/model.py`
- `pole_lraspp_multimodal_fusion/object_targets.py`
- `pole_lraspp_multimodal_fusion/radar_fusion.py`
- `pole_lraspp_multimodal_fusion/split_runtime.py`

Important caveat: this is a fusion SEG/localization workflow, not yet a clean
Faster-R-CNN-style true OD training workflow. The object head regresses
center/XYZ/dimensions/yaw/parked/radar-support targets from actor rows. A
separate true RGB+radar OD model family still needs to be located from the
supervisor or designed explicitly.

Operational caveat: the copied shell launchers in
`abiodun/pole_lraspp_multimodal_fusion/` still hardcode workflow paths under the
`neu_collab/` root. Before using the `abiodun/` copy for a long run, normalize
the launch/status/stop scripts to resolve paths relative to their own location,
or run the supervisor's root-level workflow if that complete tree is restored.

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

Original vehicle-only training-target dry run passed:

- 30/30 samples build feature tensors with shape `(7, 432, 768)`
- 30/30 samples build segmentation targets with shape `(432, 768)`
- 30/30 samples contain positive vehicle object-head targets
- 369 valid vehicle objects become training targets
- object heatmap shape `(1, 432, 768)`
- object regression shape `(10, 432, 768)`
- GT object tensor shape `(64, 9)`
- 65 vehicle targets have radar-support evidence
- 5 vehicle targets are marked parked

Current class-aware update: the localization target path now supports both
`vehicle` and `person` center heatmaps. New training uses a 12-channel object
head by default: 2 class-aware heatmap channels plus 10 shared regression
channels. The legacy `valid_vehicle_objects()` helper remains as a compatibility
wrapper, but the trainer/evaluator/dry-run now use `valid_localization_objects()`.

Class-aware target dry runs passed on the TL16 parked-ego pilots:

- `parked_ego_tl16_spawn80_rightm3_seed7_pilot300`: 4,126 valid vehicle targets
  and 4,562 valid person targets.
- `parked_ego_tl16_spawn80_right0_seed11_pilot300`: 3,936 valid vehicle targets
  and 4,216 valid person targets.
- `parked_ego_tl16_spawn85_right0_seed17_pilot300`: 4,230 valid vehicle targets
  and 4,164 valid person targets.

## Present Reusable Pieces

- `pole_lraspp_multimodal_fusion/model.py`
  - `MultiTaskFusionLRASPP`
  - LR-ASPP builder/adapters
  - checkpoint-compatible model structure
- `pole_lraspp_multimodal_fusion/object_targets.py`
  - `load_object_boxes`
  - `valid_localization_objects` for vehicle/person class-aware localization
  - `valid_vehicle_objects` compatibility wrapper
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

## Remaining Gaps Before Parked-Ego Training

1. Parked-ego dataset adapter/compatibility:
   - confirm `carla_collect_parked_ego_fusion_training_data.py` writes every
     manifest/object field expected by `train_fusion.py` and
     `evaluate_fusion.py`
   - align folder names if needed (`radar_tensors` vs `radar_tensor`,
     `masks` vs `mask_3class`)
   - class-aware trainer loader now accepts parked `.npy` radar tensors as well
     as original `.npz` tensors with a `radar` key

2. Training launch path:
   - fix the copied shell launchers to use the `abiodun/` workflow path, or
     restore/use the complete root-level supervisor workflow
   - dry-run the launcher before any overnight job

3. Parked-ego collection scale and scene choice:
   - select a dense intersection viewpoint with enough visible vehicles and
     pedestrians
   - use `scenesense_scenarios/scout_parked_ego_training_views.py` to rank
     real parked-ego spawn candidates before committing to a collection view
   - collect pilot data and check foreground coverage before the full overnight
     dataset
   - supervisor feedback: prefer a parked ego on the right side of the road so
     the camera sees multiple motion profiles: side-passing traffic, oncoming
     traffic, crossing traffic, and crosswalk pedestrians
   - selected visual candidate: TL16 spawn `80`, forward offset `4.0 m`, right
     offset `7.0 m`, yaw offset `-28.414 deg`; right-offset `7 m` moved the ego
     toward the parking/curb lane while preserving the useful intersection view
   - first full parked-ego training collection should target roughly
     12k-18k saved frames/samples; older pole/fusion runs locally used 12k,
     36k, 76.8k, and 92k sample scales depending on the experiment
   - include density variation where practical: no cars, few cars, and crowded
     traffic profiles, rather than collecting only one traffic condition
   - preferred first full dataset: low/medium/crowded density folders with
     4,000 saved samples each at `--sample-stride 2`, then merge by symlink
     before training
   - do not over-optimize radar/person support before first training; train the
     baseline first, then inspect pedestrian performance and radar support
     failure modes

4. True RGB+radar OD:
   - current fusion object head is localization-style, not a complete OD
     boxes/classes/AP route
   - ask supervisor for the true OD trainer if it exists; otherwise design a
     separate OD model family or extend the current object-head targets/evaluator

## Recommended Next Engineering Step

Start with compatibility, not overnight training:

1. Pick/visualize the parked-ego intersection view.
   - Rerun the scout with right-side-road preference and inspect the top one or
     two candidates visually before full collection.
2. Collect a small parked-ego RGB+radar pilot dataset.
3. Validate the schema and target dry run.
4. Run `train_fusion.py` for one tiny smoke epoch on that dataset.
   - Done on `2026-06-11` with
     `experiments/parked_ego_classaware_train_smoke_20260611`.
   - The smoke used `parked_ego_training_tl16_spawn80_60samp`, completed one
     CPU epoch, and wrote both `best.pt` and `last.pt`.
   - Checkpoint metadata confirms the class-aware head:
     `object_channels=12`, `object_class_names=['vehicle', 'person']`.
5. Only after the smoke epoch writes a usable checkpoint, launch the full
   overnight training job.
