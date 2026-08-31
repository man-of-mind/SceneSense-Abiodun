# SplitFusion FCOS candidate quality v1

This package freezes the recovered epoch-26 SplitFusion FCOS model and learns only a shared `263 -> 64 -> 1` quality MLP. Its input is the frozen 256-value FPN vector followed by base score, scalar class identity, normalized FPN level, semantic probability at the candidate point, maximum same-class semantic probability in the box, maximum depth-bin probability, and normalized depth-bin entropy. The zero-initialized output layer makes the initial refined score numerically equal to the base score.

Candidates are only the base detector's post-original-NMS detections (at most 100 per frame). Refinement cannot create or recover a box. Inference re-ranks them and applies one stable class-aware cross-level NMS, default IoU `0.60`, before writing the unchanged canonical detection fields. The evaluator threshold remains `0.20`.

Do not run the following until separately authorized. These are the exact future commands, from the repository root:

```bash
python3 -m pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_candidate_quality_v1.build_train_cache --output experiments/route_b_v3_1_splitfusion_fcos_r50_fpn_p2_p7_candidate_quality_v1/train_cache_epoch026 --device cuda:0

python3 -m pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_candidate_quality_v1.train_quality_head --cache experiments/route_b_v3_1_splitfusion_fcos_r50_fpn_p2_p7_candidate_quality_v1/train_cache_epoch026 --output experiments/route_b_v3_1_splitfusion_fcos_r50_fpn_p2_p7_candidate_quality_v1/quality_head_epoch005.pt --device cuda:0

python3 -m pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_candidate_quality_v1.infer_refined --quality-checkpoint experiments/route_b_v3_1_splitfusion_fcos_r50_fpn_p2_p7_candidate_quality_v1/quality_head_epoch005.pt --output experiments/route_b_v3_1_splitfusion_fcos_r50_fpn_p2_p7_candidate_quality_v1/predictions/quality_head_epoch005 --device cuda:0 --nms-iou 0.60

python3 -m pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_candidate_quality_v1.evaluate_refined --prediction-dir experiments/route_b_v3_1_splitfusion_fcos_r50_fpn_p2_p7_candidate_quality_v1/predictions/quality_head_epoch005
```

The cache builder opens only the training split and stores float16 feature vectors (after a representability check), class, float32 base score, label, sample ID, and the original `(image, FPN level, flattened point, class)` identity. Matching precedes ignore neutralization, as in the canonical evaluator: matched ignore-centred candidates remain positive, while only unmatched ignore-centred candidates receive label `-1` and no loss. Training is fixed to five epochs with sigmoid focal loss (`alpha=0.25`, `gamma=2`) on `logit(base_score) + quality_delta`, checks loss/quality gradients/quality parameters for finiteness, records final train-cache precision and recall by class at `0.20`, and writes one final head checkpoint. Refined inference checks logits and scores for finiteness. `evaluate_refined.py` delegates one completed prediction directory to the existing frozen v0.10 scorer and nine service gates; it does not redefine their equations.

The permitted CPU synthetic checks are:

```bash
python3 -m unittest pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_candidate_quality_v1.tests.test_synthetic -v
```
