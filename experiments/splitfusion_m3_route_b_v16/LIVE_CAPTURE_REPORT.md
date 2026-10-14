# M3.6 V16 — operating-range check: one full Route-B capture at constant 14 dB

**Integration: PASS.** The single authorized capture completed the full route; host exit status was 0. The
registration-bound assessment returned `PASS_M3_FULL_ROUTE_INTEGRATION` (24/24 checks, no reasons), and channel
fidelity F1–F5 all matched.

**Screening target (feedback received ≤170 ms on ≥80% of ALL policy decisions): NOT MET.** The result is
**1,112/1,491 = 0.746** (Wilson 95% 0.723–0.767). The upper bound is below 0.80. This is also lower than V15 at 12 dB
(0.788), so the two operating points do not order by SNR. Different sessions and scenes dominate a 2-dB step here.

There was one run, with no retry. V11, V12 and V15 and their reports are unchanged. All comparisons are descriptive.

## 1. Identity and the one registered change

V16 is V15's derivation (V11 + one constant channel schedule) with only the anchor changed:

| | V15 | **V16** |
|---|---|---|
| schedule | OPERATING_12DB | **OPERATING_14DB** (`CONST_OPERATING_14P0DB_V1`) |
| commanded SNR → RFsim noise_power_dB | 12.0 → −6.0 | **14.0 → −7.0** |
| Phase-14A anchor | 07 | **06** (PUSCH 14.0 / P05 13.0 / P95 14.0 dB, MCS 16; hash-bound row) |
| actor feature 22 (commanded target) | 0.342105 | **0.447368** = (14.0 − 5.5)/19 |

- **Actor input:** all 1,491 admitted proxy values were 14.0 dB; no gNB measurement reaches the actor.
- **State index:** stays 1 (INTERMEDIATE, the nearest design mean; it is a trace label only).
- **Everything else is unchanged from V11:** actor, route, 50/50 traffic and seeds, 100 MHz/273 PRB/4D5U, 170-ms
  deadline, GT/accounting, GC policy, retention, map publication and drain.
- **Prepare checks:** `prepare` compared the registration with the actual V11 registration field by field and found
  the same allowed difference set as V12 and V15.

Packages are `rl_agent/splitfusion_m3_route_b_v16{,_offline_v1,_analysis}` (V15 copies with the anchor values
swapped). The shims are `splitfusion_single_gpu_profiler_v1/{child,edge_profiled}_v1_m3_route_b_v16.py`.

Capture root (deployment worktree):
`rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261012_m3_route_b_v16/operating_14db_prospective_01`

- Registration file SHA256: `03e827f82ee9856bf33c4838d91bed256f73da62fccbadea9dba79d0feb97cc8`.
- Runtime inventory: 604 files.
- Execute token: `M3_6_ROUTE_B_OPERATING_14DB_V16_CANDIDATE_V2_C3900_ONCE_EXECUTE_V1`.

The analysis-package copy differs from V15 only in:
- names/labels;
- a four-series comparison script (`compare_v11_v12_v15.py`);
- a docstring correction to the quality-undefined wording (see §6).

## 2. Preflight (existing mandatory checks only)

| step | result |
|---|---|
| unit tests `test_m3_v16` | 59/59 |
| deploy `deploy_01` | DEPLOYED_VERIFIED; sealed v3–v11 packages identical; no tracked file touched |
| binding regression (worktree `binding_regression_01`) | PASS 22/22 |
| real-service proof `service_proof_01` (main checkout) | PASS 24/24 |
| offline assessment binding regression `offline_binding_regression_01` | PASS 12/12 |
| fresh-process startup verification (worktree `startup_verification_01`) | 6/6 roles; 25/25 negatives rejected; no capture dir |
| host `verify` with startup proof (`host_verify_01.stdout`) | PASS |

## 3. Integration result

- **Route:** `ROUTE_COMPLETED`, coverage 301/301, 1,251.9 m driven. Simulation 300.7 s; route wall 376.3 s; capture
  span 375.8 s. 2 registered roadblock clears, 0 collision incidents, no watchdog.
- **Traffic:** 50/50 vehicles and pedestrians spawned; live minima 49/48; 2 vehicles and 14 pedestrians replenished.
- **Frames:** 3,007 offers = 3,004 SENT + 1 warm-up + 1 incomplete radar + 1 end-of-route drain drop. All 3,004 sent
  frames were `RESULT_INSTALLED` (0 parse/write/other errors). Drain waited 0.050 s; closed = sent; 0 expired.
- **Decisions:** 1,491 decisions, 1,490 holds, 23 typed fallbacks. Policy coverage 1,483/1,504 = 0.986. **All 1,491
  service outcomes were `SCORED`.**
- **Replay and rescore:** actor replay 1,491 spans, 0 mismatches. Exact rescore 1,491/1,491, 0 refusals. 1,432
  previous-outcome pairs checked.
- **Retention:** 99 `RETAINED_COMPLETE`. Ordinal 1,492 was `NOT_REACHED` (route ended at 1,491). Strict PASS; GT
  references resolved.
- **P3 and map class:** P3 had 0 rejections (max translation delta 1.479e−5 m). Map-class join: 58,192 objects,
  0 mismatched.
- **Authority:** `registered_assessment_01` returned PASS 24/24. The historical V3-pinned tool STOPs only on the three
  `SCHEDULE_*` checks it reads from V3's 21.5-dB contract; this is unchanged behaviour and the tool is not the
  authority.

## 4. Channel: commanded vs measured (route window 375.5 s)

- **Commanded (actor-visible proxy, not a measurement):** 3,805 controller rows, all `ACK_ON_TIME`; 14.0 dB →
  RFsim −7.0.
- **Measured per-transmission PUSCH (calibrated T-tracer):** 59,656 transmissions.
  - Distribution: 14.0 dB 47,097 (78.9%), 13.0 dB 10,400, 13.5 dB 2,136, 14.5 dB 23.
  - **P05/P50/P95 = 13.0 / 14.0 / 14.0 dB.** Scheduler avg_snr EMA P50 13.8 dB.
- **Actual MCS:** selected = final MCS **16** on 59,455/59,609 grants.
- **Grants:** P50 273 PRB; 6,261 empty-buffer grants at 5 PRB.
- **ULSCH:** rounds 59,227/53/0/0, so **53 first retransmissions, 0 errors, 0 DTX** (V15 had 0 retransmissions).
- **Pre-action RLC backlog:** nonzero in 7 decisions.
- **MAC-log power-control SNR** (avg_snr + tpc; **not channel SNR**): P50 14.2 dB, P95 14.8 dB (202 samples).
- **Fidelity:** F1 median 14.0 ∈ [13.5, 14.5]; F2 P05 13.0 ≥ 12.5; F3 P95 14.0 ≤ 14.5; F4 3,754/3,754 rows;
  F5 MCS 16. All PASS → `CHANNEL_FIDELITY_MATCH`.

## 5. Performance (identical definitions; denominator = ALL policy decisions)

| metric | V11 23.5 dB | **V16 14.0 dB** | V15 12.0 dB | V12 8.5 dB |
|---|---:|---:|---:|---:|
| decisions | 1,771 | **1,491** | 1,489 | 1,550 |
| **feedback received ≤170 ms** | 0.873 | **0.746 (1,112; Wilson 0.723–0.767)** | 0.788 | 0.414 |
| feedback P50 / P95 / P99 ms | 156.5 / 177.4 / 214.6 | **160.5 / 185.8 / 240.0** | 159.8 / 184.2 / 220.1 | 173.6 / 214.8 / 248.4 |
| prediction completed | 1,771/1,771 | **1,491/1,491** | 1,489/1,489 | 1,550/1,550 |
| prediction ≤170 ms | 0.989 | **0.987 (1,471)** | 0.987 | 0.830 |
| prediction P50 / P95 / P99 ms | 131.0 / 150.5 / 172.1 | **135.4 / 159.2 / 175.2** | 135.2 / 157.3 / 172.4 | 149.8 / 186.1 / 207.5 |
| PUSCH P05 / P50 / P95 dB | 23.0 / 23.5 / 24.0 | **13.0 / 14.0 / 14.0** | 11.5 / 12.0 / 12.0 | 7.5 / 8.5 / 8.5 |
| selected MCS P50 | 27 | **16** | 15 | 9 |
| L2 payload P50 / P99 kB | 111.8 / 207.5 | **101.4 / 179.4** | 96.1 / 169.5 | 89.6 / 194.5 |
| modes | 11 ×1,686; 8 ×51; 9 ×31; 10 ×3 | **11 ×1,487; 8 ×4** | 11 ×1,485; 8 ×4 | 11 ×1,489; 8 ×53; 10 ×8 |
| mode-11 q×10⁴ P50 (range) | 5,832 | **6,210 (2,702–9,173)** | 6,425 | 6,727 |
| map age at install P50 / P95 ms | 185.1 / 213.8 | **190.1 / 219.8** | 189.2 / 218.8 | 201.9 / 241.8 |
| map AoI P50 / P95 / P99 ms | 248.5 / 316.8 / 356.9 | **254.4 / 324.9 / 356.0** | 252.9 / 323.4 / 360.4 | 265.8 / 337.8 / 371.8 |
| update gaps >500 ms | 0 | **0** (max 354 ms) | 0 | 0 |

**Outcome classes:** 1,054 timely-defined, 58 timely-undefined, 379 late. Every decision received feedback; no
timeouts without an outcome. All 379 late-feedback frames were still installed on the map.

**Late decomposition** (same rule as V12/V15). Of the 379 late decisions:
- **348 had both inputs ready by 170 ms** and were pushed over by the post-ready scoring + feedback tail. The tail is
  unchanged: P50 23.2 / P95 30.2 ms, versus 22.7–23.3 / 29.5–30.1 in the other runs.
- 13 had only the prediction late, 11 only GT, and 7 both.

The prediction was the later input in 1,193/1,491 decisions. Prediction latency at 14 dB is essentially the 12 dB
value (P50 135.4 vs 135.2 ms).

**Route profile (30-s bins):**
- **0–180 s:** 0.81–0.90 on time.
- **180–375 s:** 0.48–0.77. The 240–270-s bin (0.478) is the same route stretch as V15's 0.487 dip: radar P40 0.71,
  the actor's q×10⁴ 3,826 (larger payloads), prediction P50 146 ms.
- **Late route:** GT publication P50 rises to 131–141 ms and FREEZE CPU P50 to 84–90 ms.

These are observations only; no diagnosis was run.

## 6. Quality (existing analyzer definitions)

Quality was defined for 1,432/1,491 decisions. **The 59 quality-undefined decisions were all scored and had zero
eligible GT objects. This is not missing GT** (V11 92, V12 95 and V15 85 were the same kind; V15's one other
undefined decision was its GT-service refusal). Eligible-GT denominators include misses.

| class | eligible GT | within tol. | recall (all predictions) | 3-m precision | pooled seg IoU |
|---|---:|---:|---:|---:|---:|
| vehicle, 1.0 m | 4,303 | 3,022 | **0.702** (V15 0.743 / V11 0.772 / V12 0.756) | **0.881** (0.888 / 0.943 / 0.911) | **0.857** (0.724 / 0.811 / 0.760) |
| pedestrian, 1.2 m | 1,857 | 955 | **0.514** (0.599 / 0.488 / 0.537) | 0.678 (0.705 / 0.688 / 0.630) | 0.516 (0.500 / 0.489 / 0.445) |

**Deadline-aware recall, boundary = feedback received at the UE ≤170 ms (inclusive).** A decision whose feedback
arrived later keeps its eligible GT in the denominator and counts as a miss.

| class | V11 | **V16** | V15 | V12 |
|---|---:|---:|---:|---:|
| vehicle | 0.647 | **0.498** (2,143/4,303; Wilson 0.483–0.513) | 0.551 | 0.276 |
| pedestrian | 0.433 | **0.392** (728/1,857; Wilson 0.370–0.414) | 0.477 | 0.213 |

Timely-subset recall for V16 is vehicle 0.718 and pedestrian 0.521.

**Quality concerns:**
- Vehicle within-1.0 m recall (0.702) and 3-m precision (0.881) are the lowest of the four runs.
- 502 vehicle and 637 pedestrian false positives.
- This session had 27% more eligible vehicle GT than V15 (4,303 vs 3,391); it was a busier scene.
- Vehicle seg IoU is the highest of the four.
- All of the above is descriptive.

**Label note on the historical V15 report (left unchanged):**
- Its "86 GT unavailable" means quality undefined: 85 scored zero-eligible decisions plus 1 GT-service refusal.
- Its deadline-aware boundary is the same feedback-received ≤170 ms rule.

## 7. Runtime faults, GC, memory

- **GC:** no safety stop. 0 in-request gen2 collections. GC state restored. Idle full collections: 1,483; P50 4.9 /
  P99 19.3 / max 28.8 ms; 5 overlapped a request arrival.
- **Workers:** score and prep workers exited without exception. Launch stderr was empty.
- **No service refusals or faults.**
- **FREEZE wall time:** P50 79.2 / P99 121.6 / max 306.2 ms; 4 operations exceeded 200 ms. Reported, not
  investigated.
- **Prep RSS:** 705.5 → 746.2 MiB (+40.7). Accounting: 5,964 calls, 0 dropped or errors; late `compute_unbound`
  P50 4.93 ms.
- **FREEZE+BIND growth, early → late:** +16.8 ms (V15 +13.3, V11 +12.4).

## 8. Storage and cleanup

- Free at preparation start: 19,976,474,624 B. Free at launch: 19,955,888,128 B (16-GiB gate met by 2.78 GB).
- After the capture: 12,593,332,224 B. At report time: about 12.58 GB.
- The capture tree is 7,105,542,076 B. Dangling Docker volumes went from 4 to 5 (the known one-per-bring-up leak).
- **Below the 16-GiB gate for any further launch.**
- Target SNR was restored (`target_snr_restored: true`) and the partition restored (`RESTORATION_LATEST` passed).
- No CARLA, OAI, edge, GT or child processes remain, and no containers are running. Nothing was deleted, restored or
  archived.

## 9. Deliverables (main checkout `experiments/splitfusion_m3_route_b_v16/`)

- `capture_01/` (launch.sh/log/rc/times/free bytes, V3-pinned verdict + P3, retention postflight)
- `registered_assessment_01/`
- `analysis_01/`, `route_layers_01/`, `binned_report_01/`, `analysis_summary_01/`
- `deadline_quality_01/` (V11/V16/V15/V12)
- `comparison_01/01_v11_v16_v15_v12_comparison.png` (+ CSV/JSON)
- `presentation_01/` (figures 01–11) and `presentation_accounting_growth_01/`
- Preflight records and `POSTFLIGHT_COMMANDS.sh` / `POSTFLIGHT_RUN.log`

Suggested figures:
- Feedback/prediction timeline: `presentation_01/11_latency_map_freshness_time.png`.
- Completion CDF: `presentation_01/01_completion_cdf.png`.
- Action/channel: `presentation_01/09_action_scene_channel_time.png`.
- Route: `presentation_01/06_route_feedback_outcomes.png` and `07_route_snr_mcs.png`.
- Four-run comparison: `comparison_01/01_v11_v16_v15_v12_comparison.png`.

## 10. Screening outcome

Neither 12 dB (0.788) nor 14 dB (0.746) met ≥80% feedback within 170 ms. In both runs:
- the measured channel matched its calibration;
- prediction completion stayed at 0.987 by 170 ms;
- most late decisions had both inputs ready by 170 ms and were pushed over by the unchanged ~23-ms post-ready tail.

The 14 dB run did worse than 12 dB, mostly after 180 s and in a busier scene. On this evidence the screen does not
separate 12 from 14 dB. Stopped here as instructed.
