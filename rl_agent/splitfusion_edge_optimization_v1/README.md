# SplitFusion edge optimization candidate v1

This package is an unpromoted, output-preserving optimization candidate for
the deployed FCOS edge service. It leaves the frozen model and production tail
untouched until strict equivalence and timing qualification pass.

The candidate defers camera/world geometry reconstruction until after NMS and
copies retained output tensors to CPU once before constructing service rows.
It does not change actions, payloads, thresholds, NMS, p025 policy, model
weights, segmentation, or wire schemas.

Required promotion gates:

1. bit-identical post-NMS tensors and ordering;
2. bit-identical p025 indices and output tensors;
3. bit-identical segmentation labels;
4. byte-identical serialized service records;
5. unchanged feature payload, datagram, reassembly, and radio accounting;
6. a measured service-time improvement on the four live diagnostic actions.

## Qualification status

All six gates passed in the four-action live CARLA/OAI qualification recorded
under
`experiments/splitfusion_edge_optimization_v1/20260909_live_actions30_15_50_71`.
Median deployed edge-service savings were 44.3, 38.4, 45.0 and 41.1 ms for
actions 30, 15, 50 and 71. Perception tensors, p025 indices, segmentation
labels and serialized service records remained exact.

The candidate remains explicit rather than silently replacing the frozen
production tail. Promotion must update the runtime binding and its hashes. The
historical 288-cell measurements remain valid as the pre-optimization radio
and action surface; their AoI values must not be adjusted by subtracting a
constant because the shorter service time also changes latest-frame queueing.

## Detached publication candidate

`detached_tail.py` removes the optimized adapter's cross-frame singleton
handoff without changing its model or scientific outputs. The sole compute
owner runs the existing optimized tail and performs the consolidated CUDA-to-
CPU transfer. It then emits a frame-scoped work product. A distinct publication
owner constructs service records and JSON exclusively from those CPU tensors,
so the next model call may overlap publication without concurrent calls into
the frozen model.

`qualify_detached_pipeline.py` compares this handoff against the established
optimized adapter for bit-identical perception, p025 indices, segmentation and
serialized bytes. It also exercises both selected scheduling policies with a
single CUDA owner, a single publication owner, terminal reconciliation and
reports whether stage overlap is actually observed; overlap is not forced to
make the candidate pass. This is a microbenchmark, not a live CARLA/OAI result.
