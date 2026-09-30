# Run-4 split-host GT transport handoff

Status: **offline seam implemented; live lifecycle integration not performed**.
This does not authorize a live run.

## What is preserved

`gt_transport.py` transports the exact three files already produced by
`splitfusion_quality_feedback_probe_v1.gt_evidence`:

1. `<stem>.objects.json`
2. `<stem>.semantic.npy`
3. `<stem>.semantic.json`

It never recreates those files from Python objects. Before transmission and
again after storage it checks their existing schemas, seven-field GT identity,
snapshot/frame equality, NPY dtype/shape/content digest, raw-file lengths and
SHA-256 digests. The unchanged `read_ground_truth()` is the final check before
the ingress returns an ACK, so `phase6_edge_runtime_v2.Run4EvaluatorV2` can keep
reading the same filenames and bytes without modification.

The transport identity additionally binds the reward ticket's canonical
`session_uuid`, controller-lineage digest, decision/ticket/tensor sequences,
run, cell, stream, frame, capture timestamp and real-or-null anchor identity.
Filenames are derived only through the existing `_stem(stream_id, frame_id)`;
the protocol accepts no path or filename from its peer.

## Exact integration points

### W10275 sender

In the additive split-host child, after the existing
`write_object_ground_truth()` and `write_semantic_ground_truth()` calls have
both returned:

1. build `GtTransportIdentityV1` with `identity_from_phase6()` using the
   already-verified SFD4 envelope, context and evaluator GT identity;
2. call `bundle_from_phase6_paths()` with the three returned paths;
3. send it through one caller-owned, persistent, timed connection using
   `PersistentGtSenderV1`;
4. accept only an ACK whose identity, bundle and receipt digests match.

Do not remove the local writer or change the GT construction/evaluator rules.

### L10319 ingress and edge

Create the per-attempt `segmentation_evidence` directory before constructing
`GtIngressStoreV1`. As soon as `Run4EdgeProcessorV2` has verified a
reward-requested SFD4 frame and formed its `EvaluationTicketV2`, derive the
same transport identity and call `ExpectedTicketRegistryV1.authorize()`.
Only then can the ingress install that ticket's files. If GT reaches L10319
slightly before complete frame reassembly, `accept()` waits for this exact
authorization for a bounded interval; it never stores a future ticket.

The persistent listener owns bind/accept/stop and calls `serve_one()` for each
framed request. `serve_one()` intentionally does not create a socket or a
thread. Point the unchanged edge evaluator at the same evidence directory.

## Failure and replay rules

- Foreign run/cell, unknown/future ticket, unsafe identifiers, extra fields,
  oversized/truncated messages and digest drift fail closed.
- Writes are create-only and atomically linked from an fsynced temporary file.
- A logical ticket authorized with different fields is a conflict.
- A committed ticket replayed with different bytes is a conflict.
- An identical retry (for example, after a lost ACK) performs no rewrite and
  returns `DUPLICATE_IDENTICAL` with the original receipt digest.
- Receipt files are create-only under `.gt_transport_receipts/` in the same
  per-attempt evidence root. No pickle is used.

## Clock boundary

Never compare W10275 `CLOCK_MONOTONIC`/`CLOCK_MONOTONIC_RAW` values with
L10319 monotonic values and never derive one-way latency from them. Preserve
timestamps only for ordering and durations on their originating host. The
170-ms reward deadline remains the W10275 action-open-to-feedback measurement
on one clock. Wall-clock correlation, if recorded after chrony/NTP audit, is
diagnostic only and must not enter reward, timeout or qualification decisions.

## Remaining gate

The separate remote lifecycle implementation must wire these explicit points,
then pass an offline/synthetic exact-byte round trip before any live handshake.
The seam itself starts no network, CARLA, OAI, Docker or CUDA activity.
