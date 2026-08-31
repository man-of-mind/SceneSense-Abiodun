# Frozen-base person relational selector v1

This implementation-only package defines the final bounded person post-head attempt for the recovered epoch-26 SplitFusion FCOS model. It does not rebuild either locked train cache or execute the frozen model.

Each frame is one complete, untruncated set of post-NMS person candidates. The 1,044-dimensional input contains the existing 1,034-dimensional FP16-round-tripped ROI/FP32-scalar representation plus normalized box and predicted-world coordinates, semantic component information, component occupancy, and the locked consolidation decision as a feature. Labels, ground-truth coordinates, ignore flags, visibility, sample IDs, and experiment IDs never enter the selector.

`PersonRelationalSelector` is exactly a LayerNorm/linear projection to 128 dimensions, two position-free transformer encoder layers with four heads, 256-dimensional feed-forward blocks and zero dropout, followed by one zero-initialized residual-logit output. Its padding mask supports variable frame sizes through the locked maximum of 97 without truncation.

`cache_join.py` streams paired shards and fails closed on the locked hashes, corpus counts, checkpoint provenance, episode partition, frame identities, candidate counts/order, and FP32 base scores. `train_selector.py` derives all five deterministic 1:3 loss-sampling plans from one metadata scan, then keeps all candidates in attention context during the five training passes. Cached labels are used only by the training loss. Holdout scoring processes tied scores together and canonically rematches retained candidates per frame at every boundary; its one threshold must achieve precision and recall of at least 0.80 jointly and in each untouched holdout episode, then reproduce the same rematched counts at deployed score 0.20. Checkpoint loading binds both manifest hashes, the fixed training contract, and every calibration gate.

`runtime.py` replaces only person scoring/selection. The previous consolidation is never a hard filter. Vehicles retain the exact locked service calibration, original ordering, and all non-score outputs; segmentation and geometry are not changed.

Run the five synthetic CPU checks only:

```bash
CUDA_VISIBLE_DEVICES='' python3 -m unittest -v \
  pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_person_relational_selector_v1.tests.test_synthetic
```
