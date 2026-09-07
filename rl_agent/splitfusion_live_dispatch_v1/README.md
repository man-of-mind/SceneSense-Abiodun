# Preloaded SplitFusion live dispatcher v1

Status: `IMPLEMENTATION_READY_NOT_GPU_OR_LIVE_QUALIFIED`.

This package implements only the `SPLIT` branch of the locked 72-action
catalog. It does not implement `LOCAL_CPU`, `LOCAL_GPU`, or `SKIP`; those
remain future top-level agent modes. It contains no model loader, dataset
reader, inference runner, socket, or campaign launcher.

## Immutable startup binding

`SplitActionRegistry.from_runtime_binding()` reads `runtime_binding.json`, the
locked action catalog, both lock terminals, and the Phase-12B campaign binding
once during startup. It verifies their SHA-256 bindings and validates exactly
72 unique, contiguous action IDs with the complete family × quantizer × q
Cartesian product. The lookup is exposed as immutable `ActionProfile` values
through a read-only mapping. Invalid, disabled, non-SPLIT, or unregistered
entries fail closed; perception quality metrics are never used as a runtime
rejection rule.

The default startup verification also hashes the selected perception, ranker,
and AE checkpoints and the bound implementation sources. This is verification,
not loading. A deployment loader may construct and load each object once before
passing it to the runtime. The focused CPU test disables only that optional
artifact pass so it never opens a real checkpoint; the catalog and campaign
locks are still verified.

The campaign supervisor's pending runtime/radio blocker remains in place.

## Resident objects and action switching

`PreloadedSplitUERuntime` receives exactly these already-loaded objects:

- one frozen 7-channel front/C2 callable;
- the stable epoch-4 ranker;
- resident AE128, AE64, and AE32 encoders.

For every frame, the caller supplies only `action_id`, the 7-channel input,
sequence ID, and capture timestamp. The local registry resolves family,
quantizer, and q. q=0 passes `None` to the selection path and never invokes the
ranker. q>0 selects through the one resident ranker. noAE passes C2 directly to
the chosen existing UINT codec; an AE family selects its resident encoder by
family and then uses the existing AE UINT8 or shared UINT6/UINT4 codec. The one
startup-created zstd context applies the frozen level-1 configuration.

`PreloadedSplitEdgeRuntime` receives exactly these already-loaded objects:

- one callable adapter around the unchanged frozen `decode_tail`, existing
  postprocessing, and p025 service path;
- resident AE128, AE64, and AE32 decoders.

The edge accepts the reassembled raw frame bytes plus the transmitted action
identity. It compares the control-plane action with the outer envelope,
resolves the local catalog entry, decompresses once, and strictly inspects the
existing inner UINT8/UINT6/UINT4 frame. Before dequantization, decoder choice,
or tail execution, it requires catalog agreement for family/noAE status,
family ID, routing tag, transported and latent widths, quantizer and bit width,
q_e4, keep count, wire magic, codec ID, and wire version. It then selects only
the already-resident matching decoder, reconstructs C2 on the configured
`torch.device`, verifies finiteness and exact tail-device placement, and calls
the frozen tail adapter. The adapter's existing p025 representation is returned
unchanged.

Switching action IDs therefore changes only immutable profile lookup and
resident-object selection. It never unloads or reloads an AE. Runtime counters
separate caller-reported startup load/construction counts from fixed zero
hot-path load/construction counts and record frame/ranker/AE/tail dispatches.
Startup device moves, evaluation transitions, and parameter freezes are also
counted; no such state mutation occurs in `prepare()` or `process()`.

## Outer envelope

The existing `!IHH` UDP header in `phase2_map_sharing/transport.py` and
`rl_agent/multiue_oai/endpoint.py` was inspected. It represents only message
ID/chunk index/chunk count, so it cannot carry or validate the SplitFusion
action, frame sequence, capture time, or inner length.

The package consequently adds one minimal versioned 36-byte `SFD1` application
envelope, intended to be the future UDP transport's reassembled message body.
It carries protocol version, header size, action ID, 64-bit sequence ID,
64-bit capture timestamp in nanoseconds, and 64-bit inner length, followed by
the unchanged zstd-compressed scientific inner bytes. No inner wire or codec is
altered. Results report scientific inner bytes, framing/control overhead, and
total bytes separately. This phase opens no socket.

## Output and timing contract

The dispatcher neither copies nor changes thresholds, NMS, grouping,
visibility, detection/localization calculations, or segmentation tensors. It
adds no 30 m runtime filter. Immutable metadata exposes the catalog's
`segmentation_installable` and `segmentation_behavior`; downstream spatial-map
policy remains responsible for installation or retention.

Raw `time.monotonic_ns()` start/finish boundaries are recorded for:

- UE: front/backbone, ranker/selection, AE encode, quantize/pack, zstd
  compression, and total UE preparation;
- edge: zstd decompression, unpack/dequantize, AE decode, frozen tail, output
  serialization, and total edge processing.

The records publish no latency statistic. End-to-end/network timing and any
future CUDA-event measurements are deferred to GPU/live qualification.

## Review boundary

The two focused CPU cases use only lightweight stubs. They cover representative
noAE/AE128/AE64/AE32 and UINT8/UINT6/UINT4 switches, q=0 ranker bypass,
preloaded-object reuse, byte accounting and timing-stage presence. They also
mutate every inner identity dimension and prove rejection before decoder or
tail invocation, plus outer action, protocol-version, and unregistered-action
rejection. They do not claim numerical, GPU, latency, networking, or live
campaign qualification.
