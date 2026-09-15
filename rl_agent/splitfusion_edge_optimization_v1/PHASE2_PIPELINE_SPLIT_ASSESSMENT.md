# Can the edge compute worker be split into inference / post-processing / publication?

**Decision: retain the current boundary.** The one genuinely CPU-only
independent operation is already overlapped. The rest of the proposed split
targets at most ~2.4 ms per frame against binding constraints that are 40x
larger, and it requires exactly the correctness proof that the existing, far
smaller two-stream overlap failed to satisfy in production.

This assessment records the measurement and the reasoning so the question does
not have to be re-opened from scratch.

## The proposal

1. one serialized FCOS/model-inference owner;
2. one bounded latest-only post-processing stage;
3. one bounded latest-only publication/map stage;

with frame *n+1* allowed to start inference once frame *n*'s required CUDA
event completes, without waiting for frame *n*'s CPU-only work.

## Correction to the premise

Two premises do not hold as stated.

**"Publication is already separate."** True of the freshness-scheduler edge
(`splitfusion_edge_freshness_scheduler_v1/pipeline.py`, `BoundedTwoStagePipeline`),
which owns a compute worker and a publication worker. It is **not** true of the
direct edge-to-map runtime, which is the architecture the object-record path
actually uses: `live_pilot_runtime_direct_v1.process_loop` performs decode, the
frozen tail, dense-segmentation evidence, update construction and
`publisher.publish` on one thread.

**"frame n's independent post-processing / CPU-only work."** Camera-aware
post-processing is not CPU work. `postprocess_geometry_after_nms_v2` runs
`batched_nms`, `clip_boxes_to_image`, `index_select`, `argsort`, a float64 GEMM
per FPN level and the whole numerical-recovery decoder — all CUDA kernels on
the same device as the model.

## Measurement

Repaired v3 tail, real action-15 (noAE/UINT4/q0.70) reconstructions, 50 timed
frames on the RTX 5090, medians (`probe`: `begin_frame`/`resolve_frame` CUDA
event pairs plus wall stages):

| stage | wall ms | CUDA ms |
|---|---:|---:|
| `decode_tail` (the frozen FCOS model) | 6.20 launch | **15.01** |
| `camera_aware_postprocess` | 11.56 | **4.08** |
| `p025_service_filter` | 2.01 | 2.01 |
| `finite_check_outputs` | 1.39 | 0.05 |
| `finite_check_postprocess` | 0.41 | 0.40 |
| `finite_check_p025` | 0.41 | 0.40 |
| `segmentation_upsample_argmax` | 0.04 | 0.06 |
| `camera_pose_reconstruct` | 0.09 | 0.08 |
| **tail total** | **22.26** | **22.27** |

`camera_aware_postprocess` shows 11.56 ms of wall against 4.08 ms of CUDA
because its data-dependent `torch.where`/`nonzero` calls block the host until
the *model's* kernels drain. That 7.5 ms is already the model's own GPU time
being waited on, not work the split could remove.

## Why the split is not implemented

**1. The addressable saving is ~2.4 ms per frame, and only on an idle GPU.**
Everything after inference is 4.08 ms of GPU work plus ~2.4 ms of host work.
Overlapping frame *n+1*'s inference with frame *n*'s post-processing cannot
create GPU capacity: both stages are GPU-bound on one device that also runs
CARLA at `Epic`. The only genuinely hideable component is the ~2.4 ms of host
work, an upper bound of ~11% of the tail.

**2. The binding constraints are elsewhere, and are 40x larger.** The
direct edge-to-map service tail measured in this task is p99 58-107 ms in
publisher-start -> first-datagram-at-map, whose entire work sequence
microbenchmarks at p99 1.06 ms; that is a scheduling problem, addressed in
Phase 3. Phase-15 separately places ~90 ms of a ~110-114 ms worker period in
UE-side CARLA sensor-window assembly and radar rasterisation, against a model
path of only 18-23 ms. A 2.4 ms tail optimisation does not move either.

**3. The required tensor-lifetime proof is the one v3 already failed.** The
repair in this task established, with a direct reproduction (34 false
non-finite verdicts in 400 trials, 0 after the fix), that the PyTorch caching
allocator is stream-scoped: a tensor allocated on one stream and read from
another without `record_stream()` can be handed to an unrelated allocation
before the reading kernel runs, and the reader then silently observes recycled
memory. That failure came from a *small* overlap — one `isfinite` reduction on
one side stream, inside a single frame, with the checked tensors still alive.

The proposed split is the same hazard, larger in every dimension:

* every tensor post-processing consumes (`outputs["detection"]["per_level"]`,
  `outputs["geometry"]`, `outputs["anchors"]`, `semantic_logits` — 76 tensors
  in the `outputs` tree alone) is allocated by `decode_tail` on the inference
  stream and would be read on the post-processing stream;
* the overlap window grows from microseconds to a whole frame period;
* the inference owner would be concurrently allocating and freeing
  *identically shaped* blocks for frame *n+1*, which is exactly the condition
  that makes recycling likely rather than theoretical;
* the failure mode is silent wrong numbers, not a crash. The action-15 symptom
  was a fail-closed stop; a corrupted geometry tensor read on the
  post-processing stream would instead install a wrong object into the map.

Output identity could only be claimed by `record_stream`-ing every one of
those tensors and retaining them across the frame boundary — which pins frame
*n*'s full activation tree in GPU memory while frame *n+1* runs, on a GPU
shared with CARLA.

**4. Map ordering and terminal accounting each gain a new supersession
point.** A second in-flight frame means a frame can be computed and then never
published. That is representable — the map already enforces capture-monotonic
supersession and the pipeline already emits `SUPERSEDED_PENDING` — but it adds
a terminal class to reconcile for a saving of 2.4 ms.

## What is already overlapped

The tail's one genuinely independent, genuinely CPU-only operation — the
deterministic OpenCV connected-component labelling of the person mask — is
already run on a dedicated CPU worker on a dedicated CUDA stream while the
compute owner continues camera-aware post-processing, and is joined before
p025 consumes the labels. After the Phase-1 repair the semantic logits are
registered with the component stream, the reused pinned mask buffer is guarded
against reuse while a copy is in flight, and the deferred finite verdicts are
allocator-correct in both directions. Measured saving: 11.6 ms of tail median
against v2 (33.66 -> 22.08 ms), bit-identical on every registered output.

## Stage 3 alone (publication) — safe, deferred

Of the three proposed stages, the publication split is the one that *can* be
proven today: the detached v3 product materializes `perception_cpu` and the
service rows on the compute owner before hand-off, so the publication owner
touches no CUDA tensor, and output identity is trivial. It is deferred because
it is not the measured cause of the publication tail: the stall is the owning
thread not being scheduled, and moving the same work to a second thread that
competes for the same run queue does not fix that. Phase 3 addresses the
scheduling directly and instruments the boundary, so a later decision can be
made on measurement rather than on argument.

## What would re-open this

* A measured post-inference GPU stage that is large relative to the model
  (currently 4.08 vs 15.01 ms), or a model that no longer saturates the device.
* The GPU no longer being shared with CARLA during a cell.
* The Phase-3 and Phase-15 bottlenecks closing, so that a 2.4 ms tail saving is
  the binding constraint rather than noise.
* A tensor-lifetime construction that does not depend on `record_stream`
  discipline across a frame boundary — for example a double-buffered,
  explicitly owned activation arena handed between the two owners.
