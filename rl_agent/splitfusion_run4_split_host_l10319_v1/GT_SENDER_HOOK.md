# W10275 GT sender hook

This is an additive integration seam. It does not replace either frozen
Phase-6 writer and it does not run on the UE action, codec, or datagram path.

## Exact installation point

Install immediately after `GtWriteRecorderV2.wrap` in the additive split-host
child. The order must remain:

1. the original Phase-6 writers create the exact GT bytes;
2. `GtWriteRecorderV2` records all three returned paths and digests;
3. `HighWorkerGtSenderV1` joins those exact paths and sends the bundle.

The concrete resolver uses the existing `GtTicketLogV3` row and the two frozen
runtime identity maps. It must not derive a ticket from a filename or invent an
anchor action:

```python
runtime_ref = {"value": None}

# In the already-additive LivePilotCellRuntime factory:
runtime = factory(**factory_kwargs)
runtime.gt_log = ticket_log
runtime_ref["value"] = runtime

def resolve_high(gt_identity):
    live = runtime_ref["value"]
    frame_id = int(gt_identity["frame_id"])
    with ticket_log._lock:
        row = dict(ticket_log.tickets.get(frame_id) or {})
    if row.get("queue_class") != "HIGH":
        return None
    if row.get("output_identity") != dict(gt_identity):
        raise RuntimeError("HIGH ticket-log/output identity drift")
    return identity_from_ue_maps(
        gt_identity=live._gt_identity[frame_id],
        run4_identity=live._run4_identity[frame_id],
    )

def is_existing_high_worker():
    live = runtime_ref["value"]
    return (live is not None
            and threading.current_thread() is live.evaluation_worker)
```

The ticket-log `HIGH` check and exact `evaluation_worker` object comparison are
both required. A thread-name comparison alone is not sufficient.

Construct the sender with the registered endpoint only, then connect after the
remote edge/GT readiness ACK and before route admission:

```python
sender = HighWorkerGtSenderV1(
    resolve_high_identity=resolve_high,
    is_high_worker=is_existing_high_worker,
)
sender.connect()

recorded_objects, recorded_semantic = recorder.wrap(
    Q.write_object_ground_truth, Q.write_semantic_ground_truth)
Q.write_object_ground_truth, Q.write_semantic_ground_truth = (
    sender.wrap_after_recorder(recorded_objects, recorded_semantic))
```

Close the sender during additive child cleanup, before the remote GT listener
is stopped, and retain `sender.close()` in the attempt evidence. A delivery
failure is an infrastructure failure; it is never converted into policy reward
or timeout.

## Invariants

- The semantic wrapper only publishes paths in memory. The object wrapper is
  the sole network-send site, and only the existing HIGH worker may enter it.
- One caller-owned persistent TCP connection is active. A lost ACK permits one
  reconnect and an identical retry; the edge returns `DUPLICATE_IDENTICAL`.
- Pending identities are bounded at 4096, equal to the edge authorization
  registry. The 4097th distinct identity is refused before insertion.
- No pickle, filename-derived identity, path traversal, future/foreign ticket,
  or unregistered endpoint is admitted.
- W10275 and L10319 monotonic clocks are never subtracted. Local timestamps are
  for same-host ordering only. The reward deadline remains W10275
  action-open-to-feedback; any wall-clock correlation is diagnostic only.
