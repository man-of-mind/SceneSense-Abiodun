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

## Synchronization-light v2 candidate

`optimized_tail_v2.py` targets the remaining non-model edge overhead without
changing FCOS inference. The numerical-recovery decoder historically performs
and then discards per-tensor min/max/mean audits for every FPN level. Those
scalar reads repeatedly synchronize CUDA. V2 retains fail-closed finite and
positive-dimension checks at the adapter boundary, but batches the reductions
to one host observation per device. It also moves redundant p025
`index_select` self-equality assertions from every live frame into the strict
v1/v2 promotion qualification.

In two 20-frame paired deterministic runs on the RTX 5090, all perception
tensors, p025 indices, segmentation labels and serialized service bytes were
exact. Excluding the first frame, the additional median tail-adapter saving
was 11.88--13.20 ms. In the final run, median tail-adapter time changed from
43.01 ms to 29.80 ms; camera-aware post-processing changed from 13.70 ms to
3.47 ms. The remaining total is already close to the independently measured
20--21 ms FCOS inference plus the classical p025 work, so this candidate does
not attempt to alter or parallelize the frozen FCOS model itself.
The detached two-owner pipeline separately passed exact parity, terminal
accounting, frozen-state and real compute/publication-overlap gates.

These are bounded CUDA microbenchmarks, not a live CARLA/OAI latency claim.
`DetachedOptimizedTailAdapterV2` and its preload are additive; the qualified v1
runtime remains untouched pending a prospective live comparison.

## Overlapped v3 candidate

V3 preserves the v2 model and geometry path while overlapping two independent
operations after `decode_tail`: semantic person-mask connected components on a
single CPU worker and camera-aware detection post-processing on the sole CUDA
owner. It also removes a NumPy-to-Torch-to-NumPy component-label round trip,
retains one authoritative full reconstructed-C2 finite scan instead of two
consecutive identical scans, and carries the already-constructed service rows
across the detached publication boundary rather than serializing JSON and
immediately parsing it back.

All asynchronous finite checks are resolved before an output can be published;
the candidate remains fail closed. No model kernel is interrupted, no second
model worker is introduced, and the latest-only scheduler is unchanged.
