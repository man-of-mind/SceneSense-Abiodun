# SplitFusion FCOS person instance consolidation v1

This package preregisters one bounded, parameter-free, train-only feasibility study over the frozen recovered epoch-26 detector (`da14d21edbd374c1c3abce02ca4674b9f4097becfba9759aba945cea160a297f`). It loads no candidate-quality or ROI-verifier head and trains no model.

For each frame, semantic channel 2 wins only where the full-resolution semantic argmax equals person. OpenCV SAUF computes deterministic row-major 8-connected components without morphology. A continuous candidate box is rasterized by pixel-center membership; the component with maximum box/component mask IoU is assigned, with the lower row-major component ID breaking an exact IoU tie. No intersection is component `-1`, support `0`.

Candidates first pass the unchanged person score threshold `0.20`. Semantic support is either off or one of `0.01, 0.025, 0.05, 0.10, 0.20`. Duplicate grouping is either off or uses box IoU `0.05, 0.10, 0.20, 0.30, 0.40`. For an enabled grouping threshold, valid pairwise relations require the same nonnegative semantic component, world-XY distance at most 3 m, and box IoU at least the threshold. Groups are connected components of that fixed relation graph. Each group retains the highest original FCOS score, with original post-NMS order breaking exact ties. Retained fields are selected unchanged; vehicles are never filtered.

Every configuration is rematched after filtering with the reviewed canonical match-before-ignore labeler. Original candidate labels are not cached. All 36 configurations are evaluated on the eight fit episodes. Selection among recall-at-least-0.80 configurations is maximum precision, then higher recall, fewer retained predictions, and fixed grid order. Holdout is evaluated exactly once only when the selected fit result has both precision and recall at least 0.80.

`runtime_wrapper.canonical_records` emits retained predictions directly from the frozen detection mapping using their original post-NMS indices. This keeps every vehicle record and every retained-person score, box, geometry, identity, and prediction index unchanged.

Future commands, from the repository root:

```bash
python3 -m pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_person_instance_consolidation_v1.smoke \
  --device cuda:0

python3 -m pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_person_instance_consolidation_v1.build_train_cache \
  --output experiments/person_instance_consolidation_v1/train_cache --device cuda:0

python3 -m pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_person_instance_consolidation_v1.run_feasibility \
  --cache experiments/person_instance_consolidation_v1/train_cache \
  --output experiments/person_instance_consolidation_v1/feasibility_result.json
```

Compact CPU synthetic checks:

```bash
python3 -m unittest pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_person_instance_consolidation_v1.tests.test_synthetic -v
```
