# Edge-pipeline optimization and validation results

Live evidence: `experiments/splitfusion_direct_edge_map_v1/20260914_edge_optimization_validation_v1_retry1`
(4 cells, 300 transmitted frames per action, FAVORABLE_STABLE, live CARLA + OAI/RFsim,
RTX 5090, git `3bf4331`, config sha256 `d9a3e4b5…`).
Preserved first attempt: `…/20260914_edge_optimization_validation_v1` (a15 FAILED, see §5).
Baseline for comparison: `…/20260914_live_validation_retry1`.

---

## 1. The action-15 non-finite failure was not a non-finite output

The v3 cells stopped fail-closed on
`v3 camera-aware postprocess output contains non-finite values`, twice, only for
action 15 (noAE/UINT4/q0.70), while the *identical* v2 geometry path never did.

**The arithmetic cannot reach a non-finite value.** Across 500 real unbiased
train-only reconstructions (two families), the geometry tree is always finite and
nowhere near overflow:

| quantity | measured max | overflow at | headroom |
|---|---:|---:|---:|
| `log_dimensions` (feeds `exp` in float64) | 2.81 | 709.78 | 2.5×10² |
| `yaw_raw_norm` (float32) | 28.3 | 3.40×10³⁸ | 1.2×10³⁷ |

**The defect is in the check, not the output.** `_DeferredFiniteValidator`
evaluated `torch.isfinite` on a second CUDA stream, reading tensors the compute
stream allocated, and published a verdict tensor the compute stream later read
back — in *both* directions without `record_stream()`. The PyTorch caching
allocator is stream-scoped: it may hand either block to an unrelated allocation
before the other stream's kernel has run, and the verdict then describes
recycled memory rather than the checked output.

Reproduced directly, checking an all-zero tensor so a non-finite verdict can
only come from recycled storage:

| | false "non-finite" verdicts |
|---|---:|
| checked tensor released before the validation stream reads it | **34 / 400** |
| checked tensor kept alive | 0 / 3000 |
| after the repair | **0 / 400** |

### Repair

* register both directions with the allocator (`record_stream` on every checked
  tensor and on the verdict);
* retain every checked tensor in `_pending` until `resolve()`, so the verdict no
  longer depends on caller reference lifetimes it does not own;
* take the same ownership for the component stream's read of the semantic logits;
* guard the reused pinned mask buffer against reuse while a copy is in flight;
* release an abandoned frame's verdicts explicitly;
* confirm a *negative* deferred verdict with the synchronous scan the qualified
  v2 adapter performs before failing a frame. An overturned verdict is counted
  and gated at zero. Injected non-finite values still stop the frame.

### Equivalence

`qualify_v3_nonfinite_repair.py` — repaired v3 is **bit-identical** to both the
frozen reference tail and v2 on real reconstructions for actions 15/30/50/71:
perception tree, serialized service-record bytes, p025 indices, segmentation
labels. `qualify_direct_v3_parity.py` repeats this through real SFD1 v2
envelopes against the production edge, adding service rows and transport byte
accounting. Both pass exactly, with zero overturned verdicts.

---

## 2. Compute-boundary assessment

See `rl_agent/splitfusion_edge_optimization_v1/PHASE2_PIPELINE_SPLIT_ASSESSMENT.md`.
**Decision: retain the current boundary.** Camera-aware post-processing is not
CPU-only — it is 4.08 ms of CUDA against the model's 15.01 ms — so the proposed
split can hide at most ~2.4 ms per frame on a GPU already shared with CARLA,
while requiring exactly the tensor-lifetime proof §1 shows v3 failed, at 76
tensors across a whole frame period. The one genuinely CPU-only independent
operation (OpenCV connected components) is already overlapped.

---

## 3. Map-service decomposition

The path is now stamped on one wall clock from tail completion to ACK
transmission. **Every stage that is work is small; the tail is entirely in
stages where the owning thread is not running.**

Medians and p99 across the four cells (ms):

| stage | p50 | p99 | verdict |
|---|---:|---:|---|
| publisher validate | 0.87–1.17 | 2.0–2.7 | work |
| serialization_start → first datagram received | 1.29–1.48 | **116–130** | **stall** |
| first → last datagram received | 0.00–0.06 | 6.1–10.6 | work |
| last datagram → reassembly complete | 0.02–0.05 | 0.07–0.30 | work |
| reassembly complete → map worker start | 0.05–0.09 | **116–238** | **stall** |
| map worker start → association start | 2.22–2.95 | **92–158** | **stall** |
| association (normalise) | 0.17–0.22 | 8.4–20.8 | work |
| map lock wait | 0.001 | **0.002–0.006** | not contended |
| map lock hold | 0.005 | **0.13–0.68** | already minimal |
| lock released → ACK emit | 0.008–0.010 | 0.69–0.87 | work |
| ACK emit → ACK sent | 0.32–0.42 | 18.8–24.1 | work |

This **rules out by measurement** two of the candidate changes: the map lock is
not a contention point (p99 wait 6 µs, hold 0.68 ms) and the ingest hand-off
never backed up (queue depth 0, zero back-pressure, zero expiries). Preallocated
buffers are likewise not indicated — the whole publication work sequence
microbenchmarks at p50 0.96 / p99 1.06 / max 2.30 ms on an idle host.

---

## 4. Live result: large gains, and one regression

| | a15 (noAE) | a30 (AE128) | a50 (AE64) | a71 (AE32) |
|---|---|---|---|---|
| edge compute p50 (ms) | 168.7 → **67.0** | 156.3 → **66.8** | 139.6 → **53.1** | 134.1 → **46.5** |
| installed / transmitted | 42.9% → **82.9%** | 34.2% → **68.7%** | 58.5% → **89.7%** | 61.1% → **92.4%** |
| capture→install AoI p50 (ms) | 387 → **265** | 410 → **308** | 329 → **211** | 291 → **169** |
| superseded before compute | 1354 → **233** | 941 → **120** | 1167 → **86** | 1091 → **111** |
| publish→install p95 (ms) | 44.3 → *107.3* | 35.2 → *117.2* | 42.6 → *119.5* | 46.8 → *121.7* |
| publish→install p99 (ms) | 119.9 → *225.9* | 78.4 → *215.7* | 113.1 → *332.4* | 121.0 → *281.9* |
| ACK timeouts | 281 → 248 | 889 → 735 | 58 → *240* | 44 → *122* |

The edge is 2.5–2.9× faster, so far fewer frames are displaced before they can
be computed and **31–40 percentage points more of the transmitted frames reach
the map**, with capture-to-install freshness improving by 102–122 ms at p50.

**The post-publication service tail regressed** at p95/p99, and for a50/a71 that
surfaces as more ACK timeouts. Two things changed at once and are not separable
in this run: the publication rate roughly doubled, and the latency-critical
threads were pinned. The decomposition points at the pin: the two worst stall
stages both run on the map ingest thread pinned to exactly **one** core, with an
empty queue and a 5 µs lock.

**Recommended next change (not validated live — the single authorised retry is
spent):** widen `map_receive_cpus` / `map_ingest_cpus` from one core each to a
shared small set, or drop map-side pinning and keep only the receive/ingest
split. Because the SHA-pinned OAI launcher forbids repinning the softmodem, a
one-core reservation removes the thread's ability to migrate away from
contention it cannot displace, which is the opposite of what a reservation is
for.

---

## 5. Gates, accounting and lifecycle

**19 of 20 registered gates pass.** Every cell PASSED.

The one failure is `every_installed_update_has_a_complete_stage_decomposition`
— a gate added with this instrumentation, which correctly caught its own
evidence gap: the edge publication ledger was written only from the edge's
shutdown path into a mount deleted with the cell, so it was never collected
(`joined: 0`, no negative intervals). The map-side decomposition and the two
boundaries carried inside `edge_timing` survived, so the tail is still
localised; the edge-side send split is missing for this run. Fixed in a separate
commit (incremental flushed writes plus copy-out), **not live-validated**.

Exact accounting, all four cells:

* terminals == captures sent (2935 / 2899 / 2943 / 2952); 0 missing, 0
  unexpected, 0 with multiple terminals;
* reassembled == installed + stale-before-map, exactly;
* every superseded frame carries an explicit `SUPERSEDED_PENDING` terminal —
  never attributed to radio loss;
* 0 duplicate installations, 0 identity mismatches, 0 map rejections;
* install preceded ACK for every installed update; **ack-before-install = 0**;
* 0 record-bearing UE control messages — object records never traversed the
  radio, and the dense label map never left the edge mount;
* 0 incomplete reassemblies expired, 0 ingest back-pressure events.

Lifecycle: all four cells stopped the edge, the live dispatch and the map
process, and restored the RFsim noise power with read-back verification. Final
host state cold — no containers, no CARLA or softmodem processes, no UE tunnel,
no runtime scratch, GPU back to its 988 MiB baseline.

### First attempt (preserved)

`…/20260914_edge_optimization_validation_v1` failed at a15: the three
CPU-reservation options were declared on the map server's wrapper parser but not
added to its argument-split list, so they were forwarded to the baseline parser,
which rejected them and took the cell down. Repaired by deriving the split list
from the parser so the two cannot diverge, plus a regression that drives the
exact argv the adapter builds. a30 in that attempt failed on stale radio scratch
left by an earlier 13:43 run, not by this implementation; a50 was interrupted
during teardown of the diagnosis.
