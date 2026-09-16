# Exact quality-feedback timing probe

This package answers a narrow testbed question: how long after one CARLA
capture can the UE receive the **exact segmentation and localization score for
that same frame**, and can scoring overlap safely with the remaining tail
work?

The score is privileged training/evaluation feedback. It uses frozen CARLA
ground truth for the exact source frame and is therefore **not deployable
ground-truth feedback**. A deployed controller would need labels, delayed
supervision, a learned quality estimator, or another explicit proxy.

## Measured candidate

The production FCOS output and direct edge-to-map publication remain
authoritative. The additive candidate:

1. binds run, cell, stream, frame, action, profile and capture timestamp;
2. freezes semantic and object ground truth at the same CARLA frame;
3. starts an owned, event-synchronized semantic-label branch as soon as model
   logits are ready, while production object post-processing continues;
4. computes localization only after final p025 records exist;
5. emits one compact (at most 1,200 bytes), record-free
   `QUALITY_EVALUATED` progress datagram from the edge container to the UE;
6. leaves the map-terminal ledger and direct map publication unchanged.

The acceleration is currently **safe overlap plus CPU-exact scoring**. It is
not described as a GPU quality scorer. Eight bounded frames also compare the
early semantic mask byte-for-byte with the production label map and recompute
the serial reference score after ACK emission. Any mismatch fails the cell.

## Timing boundaries

All cross-process intervals use wall-clock timestamps. Process-monotonic
timestamps are retained for within-process stage validation and are never
subtracted across processes. The durable evidence separates:

- source RGB capture to UE quality-ACK receipt;
- synchronized RGB/radar availability to receipt;
- seven-channel/action-path start to receipt;
- first feature datagram send to receipt;
- model ready, final p025/serialized prediction ready, GT ready, evaluation
  enqueue/start/complete, edge socket send and UE receipt.

The report must give P50/P95/P99, the fraction at or below 140 ms, and the
fraction available before the next 8, 9 and 10 FPS decision. Unavailable
boundaries remain unavailable; no timestamp is fabricated.

## Population accounting

The strict one-outcome quality gate applies to frames that reached a final
prediction. Every sent frame is still reconciled. Frames that never reached a
final prediction are closed as `NOT_ELIGIBLE_NO_FINAL_PREDICTION` using the
existing transport/edge terminal; they are not silently counted as quality
failures. Reports keep distinct denominators for sent, reassembled,
final-prediction eligible, successful quality ACKs and failed quality ACKs.

See [README_LIVE_PROBE.md](README_LIVE_PROBE.md) for the guarded two-cell,
300-transmission live command. No live run is launched by importing or testing
this package.
