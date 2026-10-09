# Evaluator-submit-before-map on the Phase-6 edge: control (C2) vs treatment (E)

Registration `experiment_early_submit_v1.json` (sealed `796f2179…` before any arm); code
524be1a at both launches; both arms retain the UE-only `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`
and the production CPU radar path. Result record with every evidence hash:
`EARLY_SUBMIT_C2E_RESULT.json`.

## The intervention (one ordering change, edge container only)

Control: tail compute → build map update → publish → `evaluator.submit`.
Treatment: tail compute → **`evaluator.submit`** → build map update → publish.

Implemented as process-local seams in new modules (`edge_ordering_v1.py`,
`edge_profiled_v1_ordering.py`); no pinned file edited; instrumentation still bound to
59f40d1. Both arms run the same edge entry and differ only in the ordering file the UE child
writes into the edge scratch. The ticket is built after the profiled compute returns
(records list and `.numpy()` labels already materialized, identities already verified by
`verify`), with bounded record copies; the returned frame carries `evaluation=None` so the
unchanged process loop cannot submit twice; a live gate stays closed until READY so prewarm
frames never gain evaluations; submit stays `put_nowait`, and any exception on the early path
is counted and never stops the map update.

Offline evidence before any capture (logs bound in the registration): 13 new tests on the
real Phase-6 composition — ticket envelope/context/records/labels/`gt_identity` equal between
orderings; map update content and encoded bytes identical; exactly-once including holds and
pre-READY frames; no mutation in either completion order; Q_perc/kind/reason identical;
queue-full synchronous infrastructure emission with the map update intact; send error on
overflow counted; map-build failure after early submission still raises with one evaluation;
E5 evidence; sampler edge-rule match. Full suites 92 + 250 OK. Preflight C2 and E PASS.

## Captures

| Arm | Run | Gates | G1 | E1–E5 | Ordering evidence |
|---|---|---|---|---|---|
| C2 control | `20261002T042846Z` | 9/9 `CAPTURE_VALID` | 999, decision-cycle boundary | pass | serial; early 0, loop 460 = evaluator 460 |
| E treatment | `20261002T043420Z` | 9/9 `CAPTURE_VALID` | 1,000 at budget | pass | early; early 466 = evaluator 466, loop 0, suppressed 466, errors 0, live-gate rejections 0 |

Same route, seed, trace, prewarm and safety bounds; C2 first, then E from a settled host
(load₁ 0.85 after 70 s; load₅ was still 3.19 from the C2 teardown — disclosed). No retry, no
setup correction. Profiler coverage of `edge.evaluator_submit`: 460/460 and 466/466; in E all
466 submissions precede the map build end for their frame; map `map.install` joined by
identity for 459/460 and 466/466 tickets.

## Primary (C2 → E; received-feedback percentiles; profiler cost included)

| Metric | C2 | E | E − C2 |
|---|---|---|---|
| Policy decisions / holds / fallbacks | 471 / 512 / 16 | 470 / 520 / 10 | −1 / +8 / −6 |
| **Timely feedback ≤ 170 ms / all decisions** | 312/471 (**66.2%**) | 337/470 (**71.7%**) | +5.5 pt |
| **Action-open → feedback P50 / P95 / P99 / max** | 158.5 / 228.4 / 390.8 / 676.9 | 156.5 / 204.0 / 248.1 / 293.5 | −2.0 / −24.4 / −142.6 / −383 |
| Missing replies | 11 | 4 | −7 |
| SUCCESS / excluded / TIMEOUT / service failure | 274 / 38 / 149 / 10 | 293 / 44 / 129 / 4 | +19 / +6 / −20 / −6 |

Same-configuration cap-only captures now span 66.2% (C2), 71.4% (C), 75.2% (B). The +5.5 pt
lies inside that spread; two captures do not define a noise band.

## Mechanism (per reward ticket, exact identity, one raw clock; branches reported separately)

| ms | C2 P50 / P95 | E P50 / P95 |
|---|---|---|
| action-open → compute end (upstream of the change) | 143.4 / 210.7 | 142.4 / 184.6 |
| compute end → evaluator submit | **3.8 / 5.3** | **0.1 / 0.1** |
| submit → Q_perc start | 2.3 / 5.4 | 5.7 / 9.0 |
| Q_perc evaluate | 3.4 / 5.6 | 3.6 / 6.0 |
| emit end → UE receipt | 5.2 / 10.0 | 5.1 / 8.9 |
| **compute end → UE receipt (whole evaluator branch)** | **15.2 / 24.4** | **14.8 / 22.7** |
| compute end → map built / published / installed | 1.9 / 3.8 / 6.3 | 1.9 / 4.0 / 6.2 |
| late tickets, excess over 170 ms P25 / P50 / P75; ≤ 5 ms | 5.0 / 17.8 / 39.8; 37 | 5.2 / 14.9 / 29.3; 32 |

Evaluator-internal wall-clock decomposition (edge report, both arms): the GT bundle was ready
~100 ms **before** the ticket in 459/460 (C2) and 466/466 (E) evaluations, so the evaluator
is submission-bound, never GT-bound. The submission did move 3.7 ms earlier for every ticket.
But submit → GT first observed rose 2.07 → 3.43 ms and submit → emit rose 7.65 → 9.45 ms:
the evaluator thread's probe/read/evaluate phase now overlaps the compute thread's map
build + publish inside one CPython process and loses most of the head start (GIL sharing is
the obvious explanation; it is a post-hoc observation, not a registered result). Realized
gain on the branch the change touches: **0.4 ms P50 / 1.7 ms P95**.

The primary-metric improvement therefore comes from upstream of the change: action-open →
compute end P95 −26 ms, edge compute P99 241.6 → 109.7 ms, edge queue P99 76.1 → 46.6 ms —
the frozen-tail CUDA stalls were simply milder in the E run. An ordering of two 2–4 ms steps
cannot produce a −143 ms P99.

**"23 recovered tickets" hypothesis: not supported.** Late-by-≤ 5 ms tickets went 37 → 32;
C2's late-excess P25 (4.97 ms) exceeds the realized median saving and even the gross 3.7 ms
shift once the evaluator-thread slowdown is counted.

## Regressions (none found)

Map branch unchanged: compute end → install 6.3 → 6.2 ms P50, 11.6 → 10.7 P95; outcomes
459 installed / 11 superseded / 1 stale → 466 / 4 / 0; map faults none; terminals exact;
edge processing failures 0/0. Quality parity: SUCCESS Q_perc mean 0.308 vs 0.307.
UE CPU 106.6 → 107.5 %, host busy 31.4 → 31.6 %, GPU util 57.5 → 58.7 %, power 195 → 192 W,
profiler hot path 1.10 → 1.05 ms/frame. Fallbacks 16 → 10 (scene-stale 9 → 5, MCS-stale
5 → 3, startup 2 → 2). The loop counter `evaluations_submitted` reads 0 in E by
construction; the evaluator's own counters are authoritative and E5 checks them.

## Recommendation: **inconclusive — not retained as a timeliness improvement.**

The change is correct, exactly-once, byte-identical for the map and regression-free, but the
measured per-ticket effect on the branch it touches is 0.4 ms at the median, below the late
floor; the headline +5.5 pt is explained by upstream run-to-run variation. The production
(serial) ordering remains the live path. Per the approved scope: no further optimization,
retraining, Run-5 integration or captures; nothing pushed.

Analysis note: the registration-bound `compare_ordering_mechanism_v1.py` mis-joined the map
install branch and mislabeled the nested emit span; the post-seal correction is the separate
`compare_ordering_mechanism_v2.py` (analysis only; v1 untouched so its bind still verifies).
