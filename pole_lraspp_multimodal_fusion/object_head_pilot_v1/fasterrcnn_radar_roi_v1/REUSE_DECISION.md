# Repository reuse decision

The final Route B package does not own a codec or transport stack.

Reused directly from `carla_split_inference_udp_data_collect.py`:

- per-level uint8/uint6/uint4 serialization and deserialization;
- zlib/zstd/no-op entropy coders;
- UDP chunking and reassembly;
- transport configuration and payload accounting;
- generic feature AE implementation;
- server-side dummy `ImageList` reconstruction;
- the standard Faster R-CNN RPN/ROI/postprocess tail helper.

Reused from `carla_split_inference_udp_segmentation_demo.py` for future work:

- rank-based per-level saliency drop;
- heterogeneous per-level AE construction.

The Route B adapter only namespaces and flattens the complete five-level RGB
FPN plus five-level radar pyramid for those existing generic functions. The
current clean qualification does not execute q, quantization, AE or UDP.

New package-owned work is limited to ResNet50-FPN-v2 Route B class mapping and
fine-tuning, the aligned four-channel radar encoder, radar-conditioned ROI
localization, the semantic decoder, the ROI-record schema adapter, training,
fixed validation and reporting.

No legacy AE checkpoint is loaded. The new ResNet50/radar boundary shapes are
different, so only the existing implementation framework is reusable.

