# M3.3 V13 — one full Town10-map GOOD_TO_BAD capture

**Final status: the sole authorized capture completed the full route and the raw launcher returned 0 / `CAPTURE_VALID`; the registration-bound V13 assessment is `STOP` (28/29 checks) solely on `SCHEDULE_route_progress_fidelity`.** The recorded transition itself is coherent and satisfies the registered timing and channel-fidelity rules. The STOP is preserved: the sealed writer records a complete non-null `crossing` object and a terminal trace with `threshold_crossed:true`, but omits the redundant top-level `summary.threshold_crossed` key that the sealed checker requires. A read-only supplemental trace audit passes, but is not authoritative and does not change the registered verdict.

Exactly one live capture was launched. There was no retry, retraining, tuning, later condition, deletion, commit or push. M3.4 was not started.

## 1. Scope and identity

V13 changes V12's constant BAD_ALL schedule to one route-distance-triggered schedule:

- GOOD on `[0%, 50%)`: commanded target **23.5 dB**, RFsim noise **−12.0**.
- BAD on `[50%, 100%]`: commanded target **8.5 dB**, RFsim noise **−4.0**.
- Trigger: first causal ordered projection on the pinned planned route at or above half its actual 1,268.6773966-m geometry; threshold **634.3386983 m**.
- One-way latch: BAD cannot revert because of jitter, backtracking or loop closure.
- Controller cadence remains 100 ms. Registered bounds are progress age ≤1,000 ms and crossing-sample-to-command issue ≤1,100 ms. Missing/stale progress and command/readback failures abort fail-closed.
- Only the ACKed commanded-SNR feature changes at actor state feature 22. Route progress, phase identity, future settings and measured gNB SNR never enter the actor.

The frozen actor SHA256 is `bb9b0d256de79b4498cd622e1daa59edcf43ff3072dc9f675b265aac579ed3ce`. All 12 modes, 22 state features, q support, reward, corrected V3 storage accounting, tilt-aware AVO, P3 pose rule, GT/GC policy, population/seeds/interventions, CPU partition, 100 MHz/273 PRB/4D5U radio, automatic MCS, 170-ms deadline, independent map publication, retention, capacity and real-ACK drain remain unchanged.

Capture root in the deployment worktree:

`rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261009_m3_route_b_v13/good_to_bad_prospective_02`

- Registration file SHA256 `0533c483b3d7a46b6ea87c94e2f15931cd5f22c381e93201c5619da807a046a9`; canonical registration content SHA256 `723ff3c561fa42f9ef543c0b9a52bd07c83ef5cb8409370ab233cd9be174a737`.
- Prospective SHA256 `dc7789510f317e206bf8601762d9e0cd28722aa8ec5901094314b3189277a2c1`; prerequisite SHA256 `cfe907561741055e2c8fca71816b667eff8a9b3a548784c618d931f532bb1b66`.
- Runtime inventory: 625 files; exact-GT inventory: 600 entries. Source-acceptance SHA256 `76960eaa92a0da8eeeb012af24e081f4edc926d4f6209de6cb84840ee460cbc4`.
- Frozen route JSON SHA256 `fc4518a8746b9417a64616b8e544f59b16b5a31b7585298a316a59662ecfd6e5`; route progress CSV SHA256 `974593859368f24ee2bc4ac31b82118bf2e932d0de1c96858b8771e2dd4d90c0`.
- Execute token: `M3_3_ROUTE_B_GOOD_TO_BAD_V13_CANDIDATE_V2_C3900_ONCE_EXECUTE_V1`.

New production packages are `rl_agent/splitfusion_m3_route_b_v13`, `rl_agent/splitfusion_m3_route_b_v13_offline_v1`, and two V13 runtime shims. The separate `rl_agent/splitfusion_m3_route_b_v13_analysis` package and presentation-renderer extension are postflight only. Older V1–V12 sources, registrations, results and verdicts were not edited. `good_to_bad_prospective_01` is retained but was superseded after a pre-launch source-drift correction and was never launched; `_02` is the sole executed registration.

## 2. Offline qualification and launch gate

| qualification | result |
|---|---|
| focused V13 unit tests | PASS, 11/11: below/at/above half, repeat, backtrack, loop close, stale/missing progress, command failure and actor timing |
| final binding regression `binding_regression_04` | PASS, 15/15 |
| final real-service proof `service_proof_03` | PASS, 24/24 |
| registration-bound authority regression `offline_binding_regression_02` | PASS, 14/14; incorrect levels, fraction, switch count, progress binding, registration, runtime inventory and source all rejected |
| fresh-process production startup `startup_verification_02` | PASS, six roles: host, runner parent, exact runner CLI, UE child, actual slim edge image, GT service/workers |
| startup negatives | all rejected: 6 edge, 1 GT-service, 9 hostile host fixtures plus the legitimate-copy control, and 8 UE-child fixtures |
| service activity in preflight | CARLA false, OAI false, CUDA/live inference false, capture directory false |
| deployment | `deploy_02`: 28 files, four changed in the prospective deployment; all sealed V3–V12 packages verified and prior registrations/results preserved |

Immediately before launch, no experiment CARLA/OAI/edge/GT process or container was active. Free space was **25,341,800,448 B**, above the unchanged 16-GiB gate of 17,179,869,184 B.

Exact sole live command, run from the deployment worktree:

```bash
env -u PYTHONPATH /usr/bin/python3 -B -m rl_agent.splitfusion_m3_route_b_v13.host launch \
  --output rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261009_m3_route_b_v13/good_to_bad_prospective_02 \
  --startup-proof experiments/splitfusion_m3_route_b_v13/startup_verification_02/RESULT.json \
  --execute M3_3_ROUTE_B_GOOD_TO_BAD_V13_CANDIDATE_V2_C3900_ONCE_EXECUTE_V1
```

Launch interval: 2026-10-09 22:48:05Z–22:57:35Z. Numeric host exit status **0**, empty stderr, one terminal `capture_finished`, `passed:true`, raw verdict `CAPTURE_VALID`. No automatic or manual retry occurred.

## 3. Route and transition fidelity

- `ROUTE_COMPLETED`, one loop, ordered coverage **301/301 = 1.000**. Driven distance 1,251.87 m against 1,268.68 m planned; final monotone progress 1,267.925 m. Simulation 309.4 s, route wall 389.55 s, capture span 389.084 s. Return error 0.445 m / 1.144°.
- Outcome `PERMITTED_INTERVENTION`: two registered interventions clearing two roadblocks, two collision incidents, no watchdog abort and no pedestrian braking. The 3,900-frame ceiling was not the completion criterion.
- Requested/spawned population 50 vehicles / 50 pedestrians; live minima 49 / 49; two vehicle and 13 pedestrian replenishments.
- Actual threshold crossing: frame **4329**, cumulative progress **634.599777 m** against 634.338698 m. Progress sample age **43.860 ms**.
- BAD command issued **44.915 ms** after that sample and ACKed at **45.145 ms**; command ACK latency **0.231 ms**. First actor decision observing ACKed 8.5 dB was zero-based decision 699, frame **4330**, elapsed 172.311 s.
- Exactly one GOOD→BAD switch and no reversion. All 3,935 controller rows were `ACK_ON_TIME`. Route-window fidelity used 3,888 rows; first BAD route row was index 1,723.
- Registered F1–F5 all pass: GOOD measured PUSCH median 23.5 dB, BAD median 8.5 dB, correct commanded phases, transition delay in bound, and automatic selected-MCS medians 27 / 9. Result: `CHANNEL_FIDELITY_MATCH`.

### Verdict disagreement

The authoritative registration-bound file `registered_assessment_01/ASSESSMENT.json` is **STOP, 28/29**, reason `SCHEDULE_route_progress_fidelity`. Its route writer/checker mismatch is specific: `ROUTE_PROGRESS_SUMMARY.json` contains a complete crossing object, and the final `ROUTE_PROGRESS_TRACE.jsonl` row records `threshold_crossed:true`, but the summary does not duplicate that boolean at its top level. `phase_analysis_05` records `NON_AUTHORITATIVE_TRACE_AUDIT_PASS` for those raw facts and explicitly sets `changes_registered_verdict:false`.

The older historical V3-pinned tool is also preserved and STOPs 20/23 on its hard-coded constant 21.5-dB assumptions (`SCHEDULE_rows`, `SCHEDULE_controller_module`, `SCHEDULE_actor_observation_constant`). It is not the V13 authority. Neither STOP was rewritten after capture.

## 4. Whole-route and causal phase results

Decision phases below use the actor-observed ACKed command. Offered/sent/map phases use the actual route crossing frame. Radio phases use the first ACKed BAD command. This accounts for the few milliseconds between those three causal boundaries.

| metric | whole route | GOOD | BAD |
|---|---:|---:|---:|
| offered frames | 3,094 | 1,414 | 1,680 |
| sent frames | 3,092 | 1,412 | 1,680 |
| decisions / holds | 1,533 / 1,533 | 699 / 699 | 834 / 834 |
| fallbacks | 26 | 14 | 12 |
| completed predictions | 1,533 | 699 | 834 |
| map installs | 3,092 | 1,412 | 1,680 |
| quality defined / undefined | 1,475 / 58 | 654 / 45 | 821 / 13 |
| outcome: timely defined / timely undefined / late | 793 / 54 / 686 | 577 / 44 / 78 | 216 / 10 / 608 |
| modes | 11×1,448; 8×72; 10×8; 9×5 | 11×692; 8×2; 9×5 | 11×756; 8×70; 10×8 |
| mode-11 q×10⁴ P50 | 6,128 | 5,836 | 6,411.5 |

The two non-sent offers are one warm-up and one incomplete-radar window, both in GOOD. The end drain waited 0.100 s and closed all 3,092 sent obligations: zero open and zero expired on the real clock.

### Latency, payload and map freshness

Prediction completion and feedback receipt are separate boundaries. Fractions use all phase decisions as denominators; no timeout/STALE timestamp is used as inference latency.

| metric | whole route | GOOD | BAD |
|---|---:|---:|---:|
| prediction P50 / P95 / P99 (ms) | 142.30 / 184.23 / 215.47 | 130.73 / 151.19 / 167.20 | 155.96 / 193.05 / 227.50 |
| prediction ≤170 ms | 1,326/1,533 = **0.865** | 692/699 = **0.990** | 634/834 = **0.760** |
| feedback P50 / P95 / P99 (ms) | 167.33 / 209.48 / 243.74 | 154.92 / 176.86 / 205.68 | 180.75 / 219.91 / 248.59 |
| feedback ≤170 ms | 847/1,533 = **0.553** | 621/699 = **0.888** | 226/834 = **0.271** |
| L2 payload P50 / P99 (bytes, sent frames) | 105,209.5 / 243,479.6 | 110,554.5 / 164,597.0 | 100,654.5 / 251,874.7 |
| map insertion age P50 / P99 (ms) | 195.74 / 274.46 | 182.99 / 241.66 | 209.90 / 281.23 |
| map AoI P50 / P99 (ms) | 261.84 / 382.30 | 246.22 / 347.46 | 275.57 / 392.23 |
| map update gap P50 / P99 (ms) | 122.88 / 203.60 | 119.15 / 197.50 | 125.31 / 205.19 |

Payload total was 342,961,075 B. No update gap exceeded 500 ms. All 686 decisions whose feedback was late were nevertheless installed, confirming that map publication remained independent of the 170-ms feedback deadline.

### Quality

All-completion localisation denominators include every eligible GT object, including misses. Undefined GT decisions are neither hits nor misses; no offline prediction substitutes for them. Precision is the separate 3-m-gate measure.

| phase / class | eligible GT | within tolerance | all-completion recall | deadline-aware recall | 3-m precision | pooled segmentation IoU |
|---|---:|---:|---:|---:|---:|---:|
| whole vehicle, 1.0 m | 4,760 | 3,587 | **0.754** | 1,630/4,760 = **0.342** | 0.935 | 0.815 |
| whole pedestrian, 1.2 m | 1,452 | 685 | **0.472** | 440/1,452 = **0.303** | 0.622 | 0.482 |
| GOOD vehicle, 1.0 m | 1,762 | 1,316 | **0.747** | **0.650** | 0.903 | 0.822 |
| GOOD pedestrian, 1.2 m | 900 | 450 | **0.500** | **0.416** | 0.643 | 0.515 |
| BAD vehicle, 1.0 m | 2,998 | 2,271 | **0.758** | **0.162** | 0.954 | 0.811 |
| BAD pedestrian, 1.2 m | 552 | 235 | **0.426** | **0.120** | 0.587 | 0.433 |

Deadline-aware recall keeps GT attached to late predictions in the denominator and treats those objects as deadline misses. The phase quality change is confounded by route position and scene content; the large deadline-aware drop follows the loss of timely feedback, not a collapse in all-completion vehicle recall.

### Channel, backlog, scene and GT service

| metric | whole route | GOOD | BAD |
|---|---:|---:|---:|
| commanded target / RFsim noise (dB) | 23.5→8.5 / −12→−4 | 23.5 / −12 | 8.5 / −4 |
| measured per-transmission PUSCH P05 / P50 / P95 (dB) | mixed phases; P50 8.5 | 23.0 / 23.5 / 24.0 (n=20,834) | 8.0 / 8.5 / 8.5 (n=51,662) |
| selected/final MCS P50 | mixed; 9 | 27 (n=20,838) | 9 (n=51,604) |
| pre-action RLC backlog nonzero decisions | 52/1,533 | 2/699 | 50/834 |
| backlog P50 / P99 (bytes) | 0 / 45,353 | 0 / 0 | 0 / 64,155 |
| phase-bounded retransmissions / errors / DTX | — | 0 / 0 / 0 | 26 / 0 / 0 |
| camera SI P50 | — | 116.891 | 114.895 |
| radar P40 P50 | — | 0.397 | 0.436 |
| GT publication P50 / P95 (ms) | 119.13 / 150.48 | 113.50 / 137.41 | 124.52 / 155.85 |
| four accounting calls per decision, CPU P50 (ms) | 10.21 | 8.99 | 12.38 |

The full decision-wall radio counter window recorded 59 retransmissions, 6 errors and 0 DTX; phase-bounded deltas above exclude counter changes across the split boundaries. Both are retained with their definitions. Automatic MCS was never forced. Commanded target SNR is not a measurement; the calibrated per-transmission PUSCH values are the measured channel telemetry.

The 50/50 population stayed at least 49/49. Eligible object counts in the quality table and SI/P40 values provide the phase scene context. The BAD half has a higher radar P40 median and a different route section, so the lower payload median is not intrinsically “better” and cannot be attributed to channel alone.

GT accounting recorded **6,132/6,132** calls (four per decision), capacity 62,400, with zero dropped, errors or foreign frames. Freeze+bind P50 rose 9.143 ms from the first 0–90 s to the final 300–390 s. The GC safety stop is null: standard collection was not restored during the route, and the original GC state was restored at exit.

### Map and replay integrity

- 3,092/3,092 sent frames computed, published and installed; zero map parse/write/other errors.
- 113,761/113,761 installed objects were typed: 110,167 Vehicle and 3,594 Pedestrian, zero unknown.
- Decision-frame class join checked 56,315 objects over all 1,533 decisions: zero mismatched and zero unjoined.
- Actor replay: 1,533 spans / 1,533 policy decisions, zero mismatches.
- Exact archive rescore: 1,533/1,533; previous-outcome chain 1,478 pairs; zero refusals.
- P3 pose: zero rejections across UE and both GT workers. Retention: 100/100 `RETAINED_COMPLETE`, 100/100 GT references resolved.

## 5. Descriptive V11/V12 comparison

These are corresponding route halves from different sessions and scenes, not paired scene measurements and not a channel-only causal comparison.

| capture / route half | decisions | feedback ≤170 ms | prediction P50 (ms) | feedback P50 (ms) | payload P50 (kB) | vehicle recall | pedestrian recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| V11 GOOD_ALL first | 691 | 0.902 | 131.79 | 154.86 | 110.5 | 0.695 | 0.499 |
| V11 GOOD_ALL second | 1,080 | 0.855 | 130.73 | 157.70 | 112.6 | 0.811 | 0.476 |
| V12 BAD_ALL first | 707 | 0.571 | 143.20 | 166.19 | 73.5 | 0.728 | 0.557 |
| V12 BAD_ALL second | 843 | 0.282 | 156.81 | 181.22 | 106.1 | 0.773 | 0.515 |
| V13 GOOD_TO_BAD first | 691 | 0.889 | 130.81 | 154.97 | 112.6 | 0.747 | 0.498 |
| V13 GOOD_TO_BAD second | 842 | 0.277 | 155.70 | 180.47 | 100.4 | 0.758 | 0.429 |

V13 changes near the actual transition as expected in the radio path: measured PUSCH median 23.5→8.5 dB, automatic MCS 27→9, prediction P50 130.7→156.0 ms, feedback timeliness 0.888→0.271, mode-8 use 2→70 and mode-11 q×10⁴ P50 5,836→6,411.5. Payload P50 falls but P99 rises, while SI/P40 and route scene also change. This is actor response observed in one evolving run, not proof of channel-only causality.

## 6. Figures and reproducibility

All presentation outputs are PNG. Recommended entry points:

- `presentation_03/01_completion_cdf.png`: prediction-completion and feedback CDFs, separately labelled, 170-ms line.
- `presentation_03/02_map_freshness_cdf.png`: insertion age, AoI and update-gap CDFs.
- `presentation_03/03_localisation_cdf.png`: conservative localisation CDFs with all eligible GT in the denominator.
- `presentation_03/06_route_feedback_outcomes.png`: planned/completed full Town10 map with timely/late feedback and the actual transition.
- `route_layers_02/01_route_channel.png`, `02_route_scene_complexity.png`, `03_route_payload_freshness.png`, `04_route_measured_channel.png`: channel, scene, payload/freshness and measured-channel route layers.
- `presentation_03/09_action_scene_channel_time.png` and `10_action_scene_channel_frames.png`: aligned mode/q/payload/commanded SNR/SI/P40 timelines.
- `phase_analysis_05/01_phase_summary.png`, `02_transition_timeline.png`, `03_session_half_comparison.png`: causal phases, actual transition and explicitly descriptive cross-session halves.

From the main checkout, with `CAPTURE=../abiodun_run4_single_gpu_profiler_v1/rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261009_m3_route_b_v13/good_to_bad_prospective_02` and a new create-only output suffix:

```bash
env -u PYTHONPATH /usr/bin/python3 -B -m rl_agent.splitfusion_m3_route_b_v13_analysis.analyze \
  --root "$CAPTURE" --verdict experiments/splitfusion_m3_route_b_v13/registered_assessment_01/ASSESSMENT.json \
  --out experiments/splitfusion_m3_route_b_v13/analysis_REPRO

env -u PYTHONPATH CUDA_VISIBLE_DEVICES= /usr/bin/python3 -B -m rl_agent.splitfusion_m3_route_b_v13_analysis.render_route_layers \
  --capture "$CAPTURE" --analysis experiments/splitfusion_m3_route_b_v13/analysis_01 \
  --out experiments/splitfusion_m3_route_b_v13/route_layers_REPRO

env -u PYTHONPATH /usr/bin/python3 -B -m rl_agent.splitfusion_m3_route_b_v13_analysis.report \
  --capture "$CAPTURE" --analysis experiments/splitfusion_m3_route_b_v13/analysis_01 \
  --route-layers experiments/splitfusion_m3_route_b_v13/route_layers_02 \
  --out experiments/splitfusion_m3_route_b_v13/binned_report_REPRO

env -u PYTHONPATH /usr/bin/python3 -B -m rl_agent.splitfusion_m3_route_b_v13_analysis.summarize_capture \
  --capture "$CAPTURE" --analysis experiments/splitfusion_m3_route_b_v13/analysis_01 \
  --binned experiments/splitfusion_m3_route_b_v13/binned_report_02 \
  --v11-timing ../abiodun_run4_single_gpu_profiler_v1/rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261009_m3_route_b_v11/good_all_prospective_01/gt_gc_trace/storage_accounting_timing.json \
  --out experiments/splitfusion_m3_route_b_v13/analysis_summary_REPRO

env -u PYTHONPATH /usr/bin/python3 -B -m rl_agent.splitfusion_m3_route_b_v13_analysis.deadline_quality \
  --timeline V11=experiments/splitfusion_m3_route_b_v11/analysis_01/DECISION_TIMELINE.csv \
  --timeline V12=experiments/splitfusion_m3_route_b_v12/analysis_01/DECISION_TIMELINE.csv \
  --timeline V13=experiments/splitfusion_m3_route_b_v13/analysis_01/DECISION_TIMELINE.csv \
  --out experiments/splitfusion_m3_route_b_v13/deadline_quality_REPRO

env -u PYTHONPATH /usr/bin/python3 -B -m rl_agent.splitfusion_m3_route_b_v13_analysis.phase_analysis \
  --capture "$CAPTURE" --analysis experiments/splitfusion_m3_route_b_v13/analysis_01 \
  --layers experiments/splitfusion_m3_route_b_v13/route_layers_02 \
  --assessment experiments/splitfusion_m3_route_b_v13/registered_assessment_01/ASSESSMENT.json \
  --v11-analysis experiments/splitfusion_m3_route_b_v11/analysis_01 \
  --v12-analysis experiments/splitfusion_m3_route_b_v12/analysis_01 \
  --out experiments/splitfusion_m3_route_b_v13/phase_analysis_REPRO

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 /usr/bin/python3 -B \
  rl_agent/splitfusion_m3_presentation_v1/render.py \
  --analysis experiments/splitfusion_m3_route_b_v13/analysis_01 \
  --layers experiments/splitfusion_m3_route_b_v13/route_layers_02 \
  --out experiments/splitfusion_m3_route_b_v13/presentation_REPRO \
  --label 'M3.3 · GOOD_TO_BAD · V13 · full Town10 map'
```

The phase analyzer binds its inputs inside `PHASE_METRICS.json`: decision timeline `cdfbbfc4…`, whole metrics `497c9134…`, route join `e8fc148a…`, assessment `9688bb98…`, route summary `55c9588f…`, progress trace `279b1e47…`, transition trace `426bd80c…`, and radio trace `9bdf9d7b…`. `FINAL_OUTPUT_SHA256.json` binds the final analysis, route-layer, phase, deadline and presentation manifests. Important source hashes are:

- route-progress / controller: `6b92da02…` / `83112757…`
- V13 contract / host: `abdb7e00…` / `380c7f82…`
- offline integration: `bfbbe20d…`
- phase analysis: `a3eafbcb…`
- route-layer renderer / presentation renderer: `d8fc18a3…` / `2d0d42ce…`

## 7. Cleanup, storage and limitations

- Target SNR restore was verified (`target_snr_restored`), edge and map stopped, and G9/CPU-partition restoration passed.
- No experiment CARLA, OAI, edge, GT or child process and no container remained after the run. The standing desktop daemon and CUDA MPS service were not experiment processes and were left unchanged.
- Launch-to-immediate-postrun free-space change was 25,341,800,448 → 17,817,063,424 B: **7,524,737,024 B consumed**. About **17.78 GB** remained after analysis (17,776,562,176 B at the recorded cleanup checkpoint; ordinary filesystem activity changes this reading slightly). The sealed prospective/capture tree is 7,276,539,931 B and the main-checkout V13 evidence/report tree is about 38.5 MB. No data was deleted or archived.
- The principal limitation is the preserved registration-bound STOP caused by the postflight writer/checker schema mismatch. The raw full-route evidence can be inspected and compared, but it must not be relabelled as a registered integration PASS.
- GOOD and BAD occupy different route sections with different scenes. V11/V12 are different sessions. These results support a descriptive transition evaluation, not a paired scene comparison or a channel-only causal estimate.
- Commanded SNR is the controller target visible to the actor; measured PUSCH SNR is separate evaluation telemetry. MAC power-control SNR is also separate and is not presented as channel SNR.
- Fifty-eight decisions had unavailable GT and therefore undefined quality. They remain counted in decision/latency outcomes but are excluded from eligible-GT quality denominators; no offline substitute is used.

M3.3 stops here. No M3.4 or further capture is authorized by this task.
