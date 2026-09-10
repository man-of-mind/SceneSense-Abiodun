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
