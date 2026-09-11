# Final SplitFusion edge-optimization assessment

Status: **QUALIFIED OFFLINE; THREE-ACTION LIVE COMPARISON COMPLETE**

The final v3 pass combines three output-preserving changes:

1. semantic connected-component work overlaps independent camera-aware
   post-processing on one CPU worker and one dedicated CUDA stream;
2. reconstructed C2 receives one authoritative full finite-value scan instead
   of two consecutive identical scans; and
3. publication carries the constructed compact records forward instead of
   immediately parsing the JSON bytes it just serialized.

No model, checkpoint, action, q value, quantizer, feature payload, radio
configuration, p025 threshold, record schema or latest-only scheduling rule was
changed.

## Qualification

- The 24-frame CUDA microbenchmark measured a median tail saving of 8.9 ms.
- All compared perception tensors, p025 selections, segmentation labels and
  serialized service bytes were exact.
- Reconstructed C2 was bitwise exact, and frozen model hashes remained equal.
- Injected non-finite tensors were rejected fail-closed.
- Detached compute and publication retained one owner each; no model call was
  made concurrently and no running CUDA kernel was preempted.

## Live CARLA/OAI result

Each successful cell transmitted 300 frames under `FAVORABLE_STABLE` with a
fresh radio and CARLA lifecycle. Both variants used the same predicted-install
latest-only scheduler.

| Action | v2 edge service | v3 edge service | Incremental saving | v2 install AoI | v3 install AoI |
|---:|---:|---:|---:|---:|---:|
| 30 | 79.1 ms | 73.0 ms | 6.1 ms (7.7%) | 288.6 ms | 286.1 ms |
| 50 | 64.6 ms | 58.3 ms | 6.3 ms (9.8%) | 211.9 ms | 202.1 ms |
| 71 | 57.8 ms | 49.2 ms | 8.6 ms (14.8%) | 170.3 ms | 163.2 ms |

The incremental live saving is therefore 6–9 ms, below the hoped-for 10–20
ms. It is consistent in direction with the microbenchmark and is reported
without rounding it up to the target.

Relative to the original pre-optimization edge service, the complete
optimization sequence reduced the median from 160.4 to 73.0 ms for action 30,
144.5 to 58.3 ms for action 50, and 136.3 to 49.2 ms for action 71. These are
86–87 ms reductions, but the endpoints come from independent live scenes and
do not characterize run-to-run variance.

End-to-end AoI does not fall by exactly the edge-service saving. Faster service
changes latest-only replacement and installation nonlinearly, while live
sensor and radio conditions vary across fresh runs. The causal simulator must
replay the optimized service distribution rather than subtract a constant from
all historical samples.

## Fail-closed action-15 finding

Action 15 was not included in the successful latency table. Two v3 live
attempts encountered a non-finite camera-aware geometry output and stopped
fail-closed; the diagnostic reproduction recorded one `PROCESSING_FAILED`
terminal after 66 published results. This is a numerical reliability finding,
not a latency sample. The validity gate was not weakened and the failed output
was preserved.

## Safety interpretation

One of 297 installed action-71 frames met the 100 ms reference; that single
event is not service qualification. The successful medians remain 163–286 ms
capture-to-install. The current system can support cooperative map awareness
and earlier warning, but it should not be presented as the sole hard real-time
emergency-braking authority.
