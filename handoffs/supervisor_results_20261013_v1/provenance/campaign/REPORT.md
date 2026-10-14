# Static-mode v13 — batch 3 result and complete 13-arm campaign table (2026-10-09)

## Batch 3

Authorized one attempt each for the remaining seven modes, in the frozen order 10, 7, 0, 6, 4, 9, 5
(`../AUTHORIZATION_BATCH3.json`, approval message "disk space is resolved, proceed with the rest of the mdoes").
Storage plan: `../storage_plan_03/STORAGE_PLAN.json` (sha `3b53bbc5…5cf5`). The batch-1 and batch-2 plans and
authorizations are preserved unchanged.

**Run:**
- Window: 2026-10-09 19:00:39Z – 19:34:30Z; about 4.8 min per arm.
- 41.35 GB free at the start.
- Launched with the existing campaign command. Before every arm the driver enforced the 16 GB gate plus the 2.5 GB
  per-arm allowance; the gate never triggered.
- Driver rc 0. No retries, no limit changes, no deletions.

**All seven arms are VALID_MEASUREMENT** (mode 0 with the pre-registered ZERO_COMPLETION subtype):
- 1,000/1,000 frames transmitted, each run ending with the registered stop signature
  `RouteBudgetReached: TRANSMITTED_BUDGET_REACHED`.
- 500 decisions, 500 holds, 0 fallbacks, 0 actor calls per arm.
- Raw host rc = 1, the pre-registered value for actor-free arms: G8 was the only false retained gate. G8, P0 and P5 are
  NOT_APPLICABLE without an actor. For mode 0, P7 (feedback over downlink) is NOT_EXERCISED because no prediction
  completed.
- The host was clean before and after every arm.

| Arm | Completed / 500 | ≤170 ms | Later | No completed prediction (subcategories) | GT-producer capacity refusals |
|---|---:|---:|---:|---|---:|
| Mode 10 · AE32/UINT6 | 500 | 292 | 208 | 0 | 0 |
| Mode 7 · AE64/UINT6 | 413 | 2 | 411 | 87: no compute span (GT capacity 41; STALE before edge 29; no result by horizon 14); failed compute span 3 (STALE before map; not counted) | 41 |
| Mode 0 · noAE/UINT8 | 0 | 0 | 0 | 500: no compute span (GT capacity 306; no result by horizon 191; STALE before edge 2); failed compute span 1 (STALE before map; not counted) | 306 |
| Mode 6 · AE64/UINT8 | 316 | 0 | 316 | 184: no compute span (GT capacity 95; STALE before edge 56; no result by horizon 31; STALE before map 1); failed compute span 1 (STALE before map) | 95 |
| Mode 4 · AE128/UINT6 | 27 | 0 | 27 | 473: no compute span (GT capacity 304; no result by horizon 159; STALE before edge 10) | 304 |
| Mode 9 · AE32/UINT8 | 476 | 123 | 353 | 24: no compute span (STALE before edge 21; GT capacity 3) | 3 |
| Mode 5 · AE128/UINT4 | 424 | 0 | 424 | 76: no compute span (GT capacity 37; STALE before edge 27; no result by horizon 11); failed compute span 1 (GT capacity) | 38 |

The no-completion outcomes are overload outcomes. They are not infrastructure failures and not packet loss.

Mode 0 (noAE/UINT8, about 3.56 MB per frame) reproduces the M1 noAE result: zero completions in 500 decisions.

## Complete cumulative table (13 arms)

Batch-1 rows were recomputed from the same evidence and asserted equal to the sealed batch-1 report (agent, mode08 and
mode11 all `true`). Rows are in run order.

| Arm | rc | Verdict | Completed / dec. | ≤170 | Later | None | P50/P99 ms | Payload P50/P99 kB | Feedback ≤170 |
|---|---:|---|---:|---:|---:|---:|---|---|---:|
| Agent (deployed actor) | 0 | VALID_MEASUREMENT | 493/493 | 479 | 14 | 0 | 134.0 / 204.0 | 117.3 / 165.5 | 390/493 |
| Mode 8 · AE64/UINT4 · q=0 | 1 | VALID_MEASUREMENT | 500/500 | 210 | 290 | 0 | 173.1 / 430.9 | 469.1 / 511.5 | 30/500 |
| Mode 11 · AE32/UINT4 · q=0 | 1 | VALID_MEASUREMENT | 500/500 | 453 | 47 | 0 | 143.1 / 216.7 | 234.2 / 250.8 | 304/500 |
| Mode 3 · AE128/UINT8 · q=0 | 1 | VALID_MEASUREMENT | 4/500 | 0 | 4 | 496 | 429.3 / 471.6 | 2339.1 / 2379.3 | 0/500 |
| Mode 2 · noAE/UINT4 · q=0 | 1 | VALID_MEASUREMENT | 153/500 | 0 | 153 | 347 | 292.9 / 470.7 | 1437.7 / 1600.2 | 0/500 |
| Mode 1 · noAE/UINT6 · q=0 | 1 | VALID_MEASUREMENT | 1/500 | 0 | 1 | 499 | 482.8 / 482.8 | 2726.0 / 2812.2 | 0/500 |
| Mode 10 · AE32/UINT6 · q=0 | 1 | VALID_MEASUREMENT | 500/500 | 292 | 208 | 0 | 166.6 / 339.6 | 469.5 / 483.2 | 47/500 |
| Mode 7 · AE64/UINT6 · q=0 | 1 | VALID_MEASUREMENT | 413/500 | 2 | 411 | 87 | 218.7 / 451.3 | 959.5 / 975.7 | 0/500 |
| Mode 0 · noAE/UINT8 · q=0 | 1 | VALID_MEASUREMENT · ZERO_COMPLETION | 0/500 | 0 | 0 | 500 | — (no completion) | 3558.9 / 3629.0 | 0/500 |
| Mode 6 · AE64/UINT8 · q=0 | 1 | VALID_MEASUREMENT | 316/500 | 0 | 316 | 184 | 237.8 / 474.2 | 1215.3 / 1233.0 | 0/500 |
| Mode 4 · AE128/UINT6 · q=0 | 1 | VALID_MEASUREMENT | 27/500 | 0 | 27 | 473 | 314.5 / 405.3 | 1865.0 / 1895.4 | 0/500 |
| Mode 9 · AE32/UINT8 · q=0 | 1 | VALID_MEASUREMENT | 476/500 | 123 | 353 | 24 | 179.2 / 397.4 | 600.2 / 609.9 | 8/500 |
| Mode 5 · AE128/UINT4 · q=0 | 1 | VALID_MEASUREMENT | 424/500 | 0 | 424 | 76 | 228.2 / 450.0 | 876.3 / 953.8 | 0/500 |

- **Completion definition:** action-open → successful `edge.compute` with the prediction materialized on CPU. The
  P50/P99 columns cover completed decisions only. The ≤170 / Later / None columns cover every decision.
- **Feedback receipt** at the UE (≤170 ms) is a separate, later boundary, reported only in its own column. It is never
  used as a completion time.

The full quality table, with deadline-aware recall, precision (all completions and timely only) and segmentation IoU,
is `CAMPAIGN_TABLE.md`.

## What the 13 arms show (descriptive; separate sessions)

All 13 arms ran in separate live sessions on the same route and the same FAVORABLE_STABLE trace, in the frozen order.
Session-to-session variation is not controlled, so the differences below are descriptive, not causal.

- **Only four static modes ever complete ≤170 ms in meaningful numbers.** All four use AE32, or AE64 with UINT4:
  - mode 11 (234 kB): 453/500
  - mode 10 (470 kB): 292/500
  - mode 8 (469 kB): 210/500
  - mode 9 (600 kB): 123/500

  Mode 7 (960 kB) reaches 2/500. Mode 5 (876 kB) and every mode with a payload above 1 MB have 0 timely completions.
- **The agent (117 kB P50) has the most timely completions: 479/493.** Mode 11 at q=0 is next with 453/500, at twice
  the payload (234 kB).
- **Feedback ≤170 ms separates the arms more sharply than prediction completion:** agent 390/493, mode 11 304/500,
  mode 10 47/500, mode 8 30/500, mode 9 8/500, every other arm 0.
- **Modes 8 and 10 have equal P50 payload (469 vs 470 kB) but different timeliness (210 vs 292 ≤170).** They ran in
  different sessions, so this difference is not attributed to the codec.
- **Large-payload arms lose most decisions without any compute span.** Their GT producer also refuses many decisions
  for capacity (mode 0 306, mode 4 304, mode 3 319, mode 1 304). Those decisions have no GT, and no GT is never
  scored as zero objects.

## Quality: GT-coverage labels

- **EXACT:** agent, mode 8, mode 11, mode 10. Every decision has measured GT.
- **EXACT_ZERO_NO_TIMELY_PREDICTION:** every arm with 0 timely completions, plus mode 7 person. Deadline-aware recall
  is exactly 0. The denominator is a lower bound (≥), because decisions without GT are excluded, not counted as zero.
- **UPPER_BOUND_UNKNOWN_GT_ON_FAILED_OPPORTUNITIES:**
  - Mode 9 deadline-aware recall: vehicle ≤ 0.182, person ≤ 0.151; segmentation IoU vehicle ≤ 0.281, person ≤ 0.147
    (476/500 evaluable).
  - Mode 7 vehicle: recall ≤ 0.004, IoU ≤ 0.003 (413/500 evaluable).
  - GT is unknown on some failed decisions, so the true value can only be lower.
- **Mode 0:** 0 completed, so all-completion quality and precision are n/a. Deadline-aware recall is 0 against
  ≥ 422 vehicle and ≥ 213 person GT objects. The segmentation union is UNMEASURED, so its IoU is "0 wherever GT
  foreground exists".
- **All-completion quality covers only completed frames.** For low-completion arms (mode 3: 4 frames, mode 1: 1,
  mode 4: 27) it is not a whole-session figure.
- **Precision is shown two ways:** over all completions, and over timely completions only. Where no prediction was
  timely, timely-only precision is n/a.

## Storage, cleanup, readiness

- **Disk:**
  - 41.35 GB free before batch 3.
  - 26.48 GB free now (re-measured after the report: 26,478,804,992 B).
  - That is above the 16 GB gate.
- **Disk used per arm:**

  | Arm | Disk used |
  |---|---:|
  | mode10 | 2.34 GB |
  | mode07 | 2.26 GB |
  | mode00 | 1.85 GB |
  | mode06 | 2.12 GB |
  | mode04 | 1.69 GB |
  | mode09 | 2.34 GB |
  | mode05 | ≈2.27 GB |

  Every arm stayed within the 2.5 GB allowance. Mode 5's figure is taken against the post-run free space, so it also
  includes any unrelated host writes.
- **Host:** clean before and after every arm. No containers, and no CARLA, OAI or GT processes remain.
- **Campaign:** complete, 13/13 arms. Nothing else was launched. M3.3 and M3.4 were not started.

## Files

- `plots/01_campaign_prediction_completion.png`: small multiples; grey = agent reference; counts in panel titles.
- `plots/02_campaign_outcomes.png`
- `plots/03_campaign_payload.png`
- `CAMPAIGN_TABLE.md`
- `CAMPAIGN_REPORT.json`, which includes the no-completion subcategories, GT coverage and non-PASS checks for each arm.

Superseded reports, all kept:
- `campaign_report_04`: the same computation with byte-identical JSON and table. Only its plot layout differs: the
  counts overlapped the curves, the y-tick labels collided between panels, and the outcomes legend collided with the
  x-label. It has no REPORT.md.
- `campaign_report_03`: batches 1–2.
- `campaign_report_02`: labels superseded by `campaign_report_03`.

**Captures:** deployment worktree `rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261009_static_modes_v13/<arm>`.
Each contains `campaign_launch.rc`, `campaign_launch.log` and `ASSESSMENT_V13.json`.

**Stage records:** worktree `experiments/splitfusion_static_modes_v13/`:
- `campaign_stage_1791569686.json` (batch 1)
- `campaign_stage_1791571590.json` (batch 2)
- `campaign_stage_1791574470.json` (batch 3)

**Generator:** `rl_agent/splitfusion_m1_presentation_ext_v1/campaign_report.py`.
