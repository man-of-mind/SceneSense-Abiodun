# M3.5 V15 — operating-range check: one full Route-B capture at constant 12 dB

**Integration: PASS.** The single authorized capture completed the full route. The host exited with status 0, and
the registration-bound assessment returned `PASS_M3_FULL_ROUTE_INTEGRATION`: 24/24 checks, no reasons. Channel
fidelity F1–F5 all matched (`CHANNEL_FIDELITY_MATCH`).

**Screening target (feedback ≤170 ms on ≥80% of policy decisions): NOT MET.** Feedback arrived within 170 ms for
**1,174/1,489 = 0.788** of decisions (Wilson 95% 0.767–0.808). The point estimate is 1.2 pp below 0.80, and the interval
straddles the target.

There was one live run, with no retry and no second condition. V11 GOOD_ALL and V12 BAD_ALL are unchanged; every
comparison below is descriptive and spans different sessions.

## 1. Identity and the one registered change

V15 is V11 with one registered change, the commanded channel schedule. This is the same single-field derivation V12
used.

| | V11 | V12 | **V15** |
|---|---|---|---|
| schedule | GOOD_ALL | BAD_ALL | **OPERATING_12DB** (`CONST_OPERATING_12P0DB_V1`) |
| commanded SNR | 23.5 dB | 8.5 dB | **12.0 dB** |
| RFsim noise_power_dB | −12.0 | −4.0 | **−6.0** |
| Phase-14A anchor | 01 | 09 | **07** (PUSCH 12.0 / P05 11.5 / P95 12.0 dB, MCS 15; hash-bound row) |
| actor feature 22 | 0.947368 | 0.157895 | **0.342105** = (12.0 − 5.5)/19 |

- The actor receives the ACKed commanded target. All 1,489 admitted proxy values were 12.0 dB, and no gNB measurement
  reaches the actor. MCS is not forced.
- The state index is 1 (INTERMEDIATE), the design state whose mean is nearest 12.0 dB. It only labels trace rows and
  the prefix digest; the live loop discards it (`ue_target_snr_cell_runtime_v1.py:240`).
- Everything else is V11: frozen actor, 12 modes, 22 features, Route B, 50/50 traffic and seeds, 100 MHz / 273 PRB /
  4D5U, 170-ms inclusive deadline, GT, accounting V3.v2, P3, tilt-aware AVO, GC policy, retention, map publication,
  drain and cleanup.
- `prepare` compared the derived registration with the actual V11 registration and accepted it. Top-level differences
  are the same allowed identity/binding set as V12. Inside `channel`, only the registered schedule fields and the
  actor feature value differ. All preserved fields are byte-identical, and the normalised sections equal V11's.

Packages are `rl_agent/splitfusion_m3_route_b_v15{,_offline_v1,_analysis}`: mechanical copies of V12 with the anchor
values changed. The shims are `splitfusion_single_gpu_profiler_v1/{child,edge_profiled}_v1_m3_route_b_v15.py`.

Capture root (deployment worktree):
`rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261012_m3_route_b_v15/operating_12db_prospective_01`

- Registration file SHA256: `fc00c54160d1b894bf211aaaa096d2a2aff7987d03b671f9806734993d7fb4d6`.
- Runtime inventory: 604 files.
- Execute token: `M3_5_ROUTE_B_OPERATING_12DB_V15_CANDIDATE_V2_C3900_ONCE_EXECUTE_V1`.

## 2. Preflight (final sources)

| step | result |
|---|---|
| unit tests (`test_m3_v15`) | 59/59 |
| deploy (`deploy_01`) | DEPLOYED_VERIFIED; 27 files; sealed packages identical; no tracked file touched |
| binding regression (worktree `binding_regression_01`) | PASS 22/22, incl. `operating_12db_is_the_registered_anchor_07` |
| real-service proof (`service_proof_02`) | PASS 24/24 |
| offline assessment binding regression (`offline_binding_regression_01`) | PASS 12/12 (bound schedule = 12.0 / −6.0 / v15 controller; V11 and V3 schedules rejected) |
| fresh-process startup verification (worktree `startup_verification_01`) | 6/6 roles; 25/25 negatives rejected; no capture directory; no CARLA/OAI/inference |
| host `verify` with the startup proof | PASS |

The first service-proof attempt failed and is preserved in the worktree's `service_proof_01`. Two tool calls run in
parallel shared one shell, so the proof ran in the worktree, which lacks the rescore helper
(`ModuleNotFoundError`). It was rerun unchanged in the main checkout as `service_proof_02`.

## 3. Storage

- Free at launch: 18,793,779,200 B (16-GiB gate met by 1.61 GB).
- Free after the capture: 11,524,739,072 B. Free at report time: about 11.5 GB.
- The capture tree is 7,026,116,579 B. One more dangling Docker volume appeared (3 → 4).
- **The 16-GiB gate is no longer met for any future launch.** Nothing was deleted or archived.

## 4. Integration result

- **Route:** `ROUTE_COMPLETED`, ordered coverage 301/301, 1,251.9 of 1,268.7 m. Simulation 300.3 s; route wall
  369.3 s; capture span 368.9 s. `PERMITTED_INTERVENTION`: 2 registered roadblock clears, 1 collision incident, no
  watchdog.
- **Traffic:** 50/50 vehicles and pedestrians spawned; live minima 49/49; 2 vehicles and 12 pedestrians replenished.
- **Frames:** 3,003 offers = 3,001 SENT + 1 warm-up + 1 incomplete radar. All 3,001 got edge compute, publication and
  `RESULT_INSTALLED` (0 parse/write/other errors). The drain waited 0.111 s; closed = sent, 0 expired.
- **Decisions:** 1,489 decisions; 1,488 holds; 24 typed fallbacks; policy coverage 1,481/1,503 = 0.985.
  Service outcomes: 1,488 `SCORED` and **1 `NO_GT_FREEZE_QUEUE_FULL`** (see §8).
- **Replay and rescore:** actor replay 1,489 spans, 0 mismatches. Exact rescore 1,488/1,488 scored decisions (the one
  refusal is the queue-full decision). 1,408 previous-outcome pairs checked.
- **Retention:** 99 retained, all `RETAINED_COMPLETE`. Ordinal 1,492 was `NOT_REACHED` because the route ended at
  1,489 decisions. Strict check PASS; 99/99 GT references `RESOLVED`.
- **P3:** 0 rejected; max translation delta 1.479e−5 m at 140.65 m. Calls above the old 1e−5 bound: 66 camera,
  66 radar, 22 per GT worker.
- **Map class:** 56,163 objects joined, 0 mismatched; 113,630/113,630 installed objects typed.
- **Authority:** `registered_assessment_01` returned PASS 24/24. The historical V3-pinned tool
  (`capture_01/VERDICT_registered_p3.json`) STOPs only on `SCHEDULE_rows`, `SCHEDULE_controller_module` and
  `SCHEDULE_actor_observation_constant`, because it reads V3's 21.5 dB schedule. It is not the authority and was not
  modified; this is the same pattern as V11 and V12.

## 5. Channel: commanded vs measured (route window 368.6 s)

- **Commanded (actor-visible proxy, not a measurement):** 3,730 controller rows, all `ACK_ON_TIME`; 12.0 dB →
  RFsim −6.0.
- **Measured per-transmission PUSCH (calibrated T-tracer):** 59,936 transmissions.
  - Distribution: 12.0 dB 47,440 (79.2%), 11.5 dB 12,168, 12.5 dB 193, 11.0 dB 135.
  - P05/P50/P95 = **11.5 / 12.0 / 12.0 dB**. Scheduler avg_snr EMA P50 11.8 dB.
- **Actual MCS:** selected = final MCS **15** on 59,803/59,898 grants (95 grants at 16–28).
- **Grants:** P50 273 PRB. 6,253 empty-buffer grants at 5 PRB. Lead time 3 ms (47,604) / 5 ms (12,294).
- **ULSCH:** rounds 59,249/0/0/0; **0 retransmissions, 0 errors, 0 DTX.**
- **Pre-action RLC backlog:** nonzero in 12/1,489 decisions; max 81,715 B.
- **MAC-log power-control SNR** (avg_snr + tpc, **not channel SNR**): P50 14.1 dB, P95 14.7 dB (199 samples). Kept
  separate from PUSCH.
- **Fidelity:** F1 median 12.0 ∈ [11.75, 12.25]; F2 P05 11.5 ≥ 11.25; F3 P95 12.0 ≤ 12.25; F4 3,685/3,685 rows in
  window; F5 MCS 15 = 15. All PASS.

## 6. Performance (identical definitions for V11, V15 and V12; denominator = all policy decisions)

| metric | V11 GOOD 23.5 dB | **V15 12.0 dB** | V12 BAD 8.5 dB |
|---|---:|---:|---:|
| decisions | 1,771 | **1,489** | 1,550 |
| **feedback ≤170 ms** | 1,546 (0.873) | **1,174 (0.788; Wilson 0.767–0.808)** | 642 (0.414) |
| feedback P50 / P95 / P99 ms | 156.5 / 177.4 / 214.6 | **159.8 / 184.2 / 220.1** | 173.6 / 214.8 / 248.4 |
| prediction completed | 1,771/1,771 | **1,489/1,489** | 1,550/1,550 |
| prediction ≤170 ms | 1,752 (0.989) | **1,470 (0.987)** | 1,286 (0.830) |
| prediction P50 / P95 / P99 ms | 131.0 / 150.5 / 172.1 | **135.2 / 157.3 / 172.4** | 149.8 / 186.1 / 207.5 |
| GT published P50 ms | 120.6 | **118.0** | 114.4 |
| PUSCH P05 / P50 / P95 dB | 23.0 / 23.5 / 24.0 | **11.5 / 12.0 / 12.0** | 7.5 / 8.5 / 8.5 |
| selected MCS P50 | 27 | **15** | 9 |
| L2 payload P50 / P95 / P99 kB | 111.8 / 165.4 / 207.5 | **96.1 / 145.2 / 169.5** | 89.6 / 164.1 / 194.5 |
| mode share | 11: 1,686; 8: 51; 9: 31; 10: 3 | **11: 1,485; 8: 4** | 11: 1,489; 8: 53; 10: 8 |
| mode-11 q×10⁴ P50 (range) | 5,832 | **6,425 (2,649–9,198)** | 6,727 |
| map age at install P50 / P95 ms | 185.1 / 213.8 | **189.2 / 218.8** | 201.9 / 241.8 |
| map AoI P50 / P95 / P99 ms | 248.5 / 316.8 / 356.9 | **252.9 / 323.4 / 360.4** | 265.8 / 337.8 / 371.8 |
| update gaps >500 ms | 0 | **0** (max 413 ms) | 0 |

**Outcome classes:** 1,094 timely-defined, 80 timely-undefined, 315 late. All 1,489 decisions received feedback; no
outcome was missing.

**Where the lateness comes from** (same decomposition as V12). Of the 315 late decisions:
- 291 had **both inputs ready by 170 ms** and were pushed over by the post-ready scoring + feedback tail. The tail is
  unchanged: P50 22.9 / P95 29.5 ms, versus V11 22.7 / 29.7 and V12 23.3 / 30.1.
- 13 had only the prediction late, 5 only GT, and 6 both.

The prediction was the later input in 1,209/1,489 decisions. At 12 dB the prediction is ~4 ms later than at 23.5 dB
(MCS 15 vs 27) rather than V12's ~19 ms. That leaves a ~147-ms effective budget that 79% of decisions met.

**Route profile (30-s bins):** timeliness ranges 0.70–0.92, except two bins:
- **240–270 s: 0.487.** Prediction P50 144 ms and GT 138 ms. This is the densest scene stretch: radar P40 0.57
  (highest), FREEZE CPU P50 93 ms, and the actor chose lower q (q×10⁴ 4,852 → larger payloads).
- **The final partial bin: 0.53.** See §8.

These are descriptive; the scene differs by session.

### Quality (V11 analysis definitions)

Quality is defined for 1,403/1,489 decisions; 86 have GT unavailable and are counted neither as hits nor misses.
Eligible-GT denominators include misses.

| class | eligible GT | within tol. | recall (all predictions) | 3-m precision | pooled seg IoU |
|---|---:|---:|---:|---:|---:|
| vehicle, 1.0 m | 3,391 | 2,518 | **0.743** (V11 0.772 / V12 0.756) | **0.888** (0.943 / 0.911) | **0.724** (0.811 / 0.760) |
| pedestrian, 1.2 m | 1,519 | 910 | **0.599** (0.488 / 0.537) | 0.705 (0.688 / 0.630) | 0.500 (0.489 / 0.445) |

**Deadline-aware recall** (a late prediction counts as a miss; `deadline_quality_02`):

| class | V11 | **V15** | V12 |
|---|---:|---:|---:|
| vehicle | 0.647 | **0.551** (1,869/3,391; Wilson 0.534–0.568) | 0.276 |
| pedestrian | 0.433 | **0.477** (725/1,519; Wilson 0.452–0.502) | 0.213 |

Timely-subset recall for V15 is vehicle 0.736 and pedestrian 0.618.

## 7. GT accounting, memory, GC

- **Accounting (V3.v2, prep only):** 5,952 calls = 4 per scored decision; 0 dropped, errors or foreign callers. Late
  `compute_unbound` P50 4.88 ms (V11 4.99).
- **FREEZE / BIND growth, 0–90 s → 300–390 s:** FREEZE +12.6 ms, BIND +0.7 ms, combined +13.3 ms (V11 +12.4,
  V12 +15.7).
- **Prep RSS:** 705.3 → 761.2 MiB (+55.9 MiB; V11 +41.6, V12 +46.3). GC state restored exactly.
- **GC: no safety stop** (V12 had one at decision 1,386). 0 in-request gen2 collections. Idle full collections:
  1,478; P50 4.7 / P99 20.6 / max 44.4 ms; 3 overlapped a request arrival.

## 8. Runtime anomaly — end-of-route stall (last ~1.5 s, decisions 1,483–1,487)

Both branches slowed in the final second of the route:

| decision | prediction | GT publication | feedback | notes |
|---|---:|---:|---:|---|
| 1,483 | 325.7 ms | 353.5 ms | 373 ms | |
| 1,485 | | 557.9 ms | 581 ms | FREEZE 518 ms wall / 343 ms CPU (route-wide FREEZE P50 77.6, P99 110.4 ms) |
| 1,486 | | 461.8 ms | 479.8 ms | |
| 1,487 | | refused | 169.0 ms | `NO_GT_FREEZE_QUEUE_FULL` → `EVALUATOR_FAULT`, `TIMELY_UNDEFINED` |

- The integration authority accepted this decision; it is the only non-`SCORED` service outcome.
- Decision 1,487's own feedback was within 170 ms and its map frame was installed.
- **Not GC:** there were 0 in-request gen2 collections. **Cause not investigated.**
- Effect on the headline: 3 of the 315 late decisions fall in this window, so feedback ≤170 ms would be about 0.790
  without it. **The screen does not hinge on it.**

**Analysis bookkeeping caused by this decision:** decision 1,487 has no GT service capture timestamp, so the V12
analyzer copy wrote an empty `capture_wall_s` and the route-layer renderer crashed (`analysis_01` chain;
`route_layers_01` etc. are empty/partial and preserved).
- The V15 analyzer copy now falls back to the same frame's map-install capture time. That value is identical on all
  1,488 decisions that have both (difference 0.0 s).
- It records `capture_time_source=map_install_record_same_frame` for that row.
- The `_02` chain is authoritative. No metric definition changed.

## 9. Cleanup

- Target SNR restored (`target_snr_restored: true`); CPU partition restored (`RESTORATION_LATEST.json` passed).
- No CARLA, OAI softmodem, edge, GT or child processes remain, and no capture containers are running.
- No evidence was deleted.

## 10. Deliverables (main checkout `experiments/splitfusion_m3_route_b_v15/`)

- `capture_01/`: launch.sh / log / stderr (empty) / rc 0 / UTC times / free bytes; V3-pinned verdict + P3 record;
  `postflight_registered_p3/` (retention + GT references).
- `registered_assessment_01/`: the authoritative verdict.
- `analysis_02/`: metrics, decision/frame tables and six PNGs. `analysis_01` is preserved (failed chain).
- `route_layers_02/`, `binned_report_02/`, `analysis_summary_02/` (radio, fidelity, accounting),
  `deadline_quality_02/`.
- `v11_v12_comparison_03/01_v11_v15_v12_comparison.png`: three-way comparison. `_02` has identical data and an
  overlapping legend.
- `presentation_02/`: latency/completion and action/channel timelines, figures 01–11.
- `presentation_accounting_growth_02/`.
- Preflight records: `prepare_01.json`, `deploy_01`, `service_proof_02`, `offline_binding_regression_01`,
  `startup_verification_01.stdout`. The worktree holds `binding_regression_01/`, `startup_verification_01/` and the
  failed `service_proof_01/`.
- Command records: `POSTFLIGHT_COMMANDS.sh`, `POSTFLIGHT_COMMANDS_02.sh`,
  `POSTFLIGHT_COMMANDS_03_comparison_layout.sh`, with logs `POSTFLIGHT_RUN*.log`.

Suggested figures:
- Feedback/prediction timeline: `presentation_02/11_latency_map_freshness_time.png`.
- Completion CDF: `presentation_02/01_completion_cdf.png`.
- Action/channel: `presentation_02/09_action_scene_channel_time.png`.
- Route: `presentation_02/06_route_feedback_outcomes.png` and `07_route_snr_mcs.png`.
- Comparison: `v11_v12_comparison_03/01_v11_v15_v12_comparison.png`.

## 11. Answers

1. **Full route and integration?** Yes: 301/301; 3,001/3,001 installed and closed; rc 0; authority 24/24 PASS.
2. **Channel as registered?** Yes: commanded 12.0 / −6.0 on every row; PUSCH 11.5 / 12.0 / 12.0 dB; MCS 15; 0 errors.
3. **≥80% feedback within 170 ms?** **No.** 0.788 (1,174/1,489), Wilson 0.767–0.808: within sampling uncertainty of
   0.80, but the point estimate is below it.
4. **Quality concerns:**
   - Vehicle seg IoU 0.724 and 3-m precision 0.888 are the lowest of the three runs.
   - The deadline costs 852 of 3,391 eligible vehicle GT (deadline-aware 0.551 vs 0.743 all-prediction).
   - Pedestrian recall is higher than both other runs, but that is the scene mix.
   - All of this is descriptive across sessions.
5. **Next (not launched):** per the pre-agreed rule, **14 dB** (anchor 06, RFsim −7.0, MCS 16). It would need
   storage (gate not met) and a separate authorization. If 12 dB is to be reconsidered given the interval straddling
   0.80, that needs a repeat capture or confirmation, not a reinterpretation of this one.
