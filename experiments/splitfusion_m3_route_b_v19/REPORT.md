# M3 final spatial channel demonstration

Verdict: **PASS_M3_FULL_ROUTE_INTEGRATION**. Exactly one capture completed the Town10 Route B loop on 12 October 2026. The approved synthetic position-indexed field ran on the qualified V17/R1 containment/accounting stack, with no live retry. This is not measured coverage, building-derived propagation, or an endpoint search.

The low core was actually exercised: 108 policy decisions, 107 predictions completed by 170 ms, and 103 timely scored feedbacks. Its measured PUSCH median was 13.0 dB, versus 23.0 dB outside. Low-core payloads were smaller and feedback was usually timely, but region comparisons involve different scenes and do not establish a causal benefit from worsening the channel.

## Validity and route completion

The [registered spatial assessment](registered_assessment_01/ASSESSMENT.json) passed all 29 revision checks and 27 retained checks. Route coverage was 301/301 ordered checkpoints; driven distance was 1,251.71 m, return error 0.495 m, and collisions zero. The unchanged harness accepted **PERMITTED_INTERVENTION**: two existing roadblock-clear interventions, not an intervention-free run. Density, collision and route checks were not relaxed. The route ended normally at 3,085 transmitted frames, below the 3,900 safety ceiling.

Those frames comprise **1,518 policy decisions, 1,517 holds and 50 fallbacks**. All 3,085 edge computations completed and all 3,085 maps installed. The preparation ledger also records one warm-up and one incomplete-radar-window drop. Actor replay matched all **1,518/1,518** decisions; exact rescoring matched all **1,518/1,518 SCORED** decisions. There were zero GT-unavailable outcomes, zero geometry refusals, and **27 scored zero-eligible frames** (all HIGH), distinct from missing GT. Of these 27, 24 feedbacks were timely and three late. The retained controller calls the 24 timely undefined-quality outcomes `EVALUATOR_FAULT`; this is not a GT-service failure.

## Regional results

Region is the observed pose behind the ACKed command admitted to each policy decision. All 1,518 classifications also agreed with the decision's actual position. Rates use **all policy decisions in that region**. All predictions and feedbacks eventually completed. Pairs below are P50 / P99 unless specified; kB means 1,000 complete SFD4 application bytes, not radio-airtime bytes.

| Metric | HIGH | TRANSITION | LOW_CORE |
|---|---:|---:|---:|
| Policy decisions | 1,385 | 25 | 108 |
| Prediction complete by 170 ms | 1,376/1,385 (99.4%) | 25/25 (100%) | 107/108 (99.1%) |
| Feedback received by 170 ms | 1,233/1,385 (89.0%) | 23/25 (92.0%) | 103/108 (95.4%) |
| SCORED feedback by 170 ms | 1,233/1,385 | 23/25 | 103/108 |
| Defined-quality feedback by 170 ms | 1,209/1,385 | 23/25 | 103/108 |
| GT scored / unavailable / zero-eligible | 1,385 / 0 / 27 | 25 / 0 / 0 | 108 / 0 / 0 |
| Prediction completion ms | 130.7 / 162.6 | 129.3 / 156.7 | 128.2 / 157.8 |
| Feedback receipt ms | 155.8 / 212.0 | 150.2 / 183.5 | 151.6 / 198.2 |
| Payload kB | 114.96 / 220.12 | 96.69 / 154.45 | 71.36 / 135.82 |
| Vehicle localisation within 1.0 m | 3,383/4,539 (74.5%) | 14/20 (70.0%) | 229/296 (77.4%) |
| Pedestrian localisation within 1.2 m | 628/1,436 (43.7%) | 9/11 (81.8%) | 41/60 (68.3%) |
| Pooled segmentation IoU vehicle / pedestrian | 85.92% / 46.23% | 74.44% / 65.05% | 75.36% / 56.02% |
| Policy-frame map installation age ms | 195.3 / 230.8 | 192.3 / 216.1 | 190.4 / 215.8 |
| Time-sampled map AoI ms | 249.8 / 366.3 | 230.7 / 317.1 | 241.3 / 327.4 |
| Measured PUSCH P05 / P50 / P95 dB | 21 / 23 / 24 | 12.5 / 17 / 23 | 11.5 / 13 / 14 |
| Measured PUSCH samples / median MCS | 42,481 / 27 | 777 / 21 | 3,687 / 16 |

Localisation includes every known eligible GT object, including misses. The regional diagnostic pools recorded object/pixel counts from all SCORED records, including zero-eligible frames; it does **not** assign a defined quality scalar to those frames. The unchanged legacy analysis additionally retains its defined-frame-only quality view: whole-run vehicle/person IoU 85.57%/47.13% over 1,491 defined frames, versus 85.48%/47.07% for the all-SCORED pool. Whole-run conservative localisation is 3,626/4,855 vehicles and 678/1,507 pedestrians; the denominator is identical in both views. Small pedestrian and transition samples warrant caution. [Full regional table](regions_01/REGIONS.csv) and [counts, quality and context](regions_01/SUMMARY.json) retain these definitions.

Mode 11 accounted for 1,435/1,518 decisions. Its conditional median drop fraction q was 0.5715 HIGH, 0.6474 TRANSITION and 0.7358 LOW_CORE; higher q means more spatial dropping. HIGH before entry had 517/564 timely feedbacks (91.7%), payload median 123.92 kB and mode-11 median q 0.52695. HIGH after exit had 716/821 (87.2%), 112.13 kB and q 0.58295. These are different route sections, not paired scenes or reset-controlled comparisons. No inference of new latency or quality from actor outputs is made.

## Channel fidelity and causal inputs

All **3,947** controller samples matched the exact approved position formula, calibration interpolation, command chronology and ACK publication; 3,626 were HIGH, 58 TRANSITION and 263 LOW_CORE. The field is time/action/outcome independent: centre (-55,140) m, semi-axes (80,45) m, transition scale 1.25, exactly the approved high/low trigonometric functions and smoothstep blend. No random trace was introduced. The 100-ms controller and 0.25-dB RFsim command granularity are retained.

Across **46,945 measured transmissions**, measured-minus-commanded P05/P50/P95 was -0.589/+0.079/+0.639 dB; absolute residual P50/P99 was 0.324/0.749 dB, maximum 1.437 dB. LOW_CORE residual P05/P50/P95 was -0.976/-0.049/+0.196 dB; measured values ranged 11–14 dB despite targets constrained to 12–14 dB. Thus the measured trace followed the imposed field closely, but was not identical to its targets. No new numeric fidelity acceptance threshold was invented.

Fidelity uses each transmission's latest preceding ACK at telemetry ingestion in the host RAW clock; median ACK age 49.9 ms, maximum 111.1 ms. This is observer-time alignment, not per-payload/HARQ identity or proof of zero propagation lag. Decision plots show the preceding measured transmission. The actor still receives only **acknowledged commanded SNR** through the existing SNR feature, plus its existing prior-MCS feature—not measured PUSCH SNR, position or region. The inherited `METRICS.json.channel.achieved_ul_pusch_snr_db` MAC-log field is not used as measured PUSCH evidence here; [PUSCH_SPATIAL.csv](regions_01/PUSCH_SPATIAL.csv) uses the existing per-transmission T-tracer reader.

Startup waited for an actually observed initial pose. During ACTIVE driving, maximum source age at selection was 248.926 ms and at command 249.108 ms, within the unchanged 1,000/1,100-ms safeguards. After route completion, 46 terminal samples held the last observed position without restamping its source; maximum age 4.546 s, within the retained 20-s terminal bound. These terminal ages are not active-driving freshness exceptions.

## Latency and map boundaries

Whole-run decision prediction completion was **130.5/162.4 ms P50/P99**, with 1,508/1,518 (99.3%) complete by 170 ms. Feedback receipt was **155.4/209.7 ms**, with 1,359/1,518 (89.5%) timely feedbacks and timely SCORED feedbacks; 1,335/1,518 (87.9%) were timely and quality-defined. Prediction latency ends at successful edge.compute completion with CPU outputs materialised; feedback latency ends at actual UE receipt. Both start at action-open. The **original 170-ms feedback deadline** is not a separately qualified prediction SLA. TIMEOUT or STALE timestamps are not latency samples.

All 159 decisions with late feedback still installed on the map. Across all 3,085 updates (including holds/fallbacks), capture-to-install age P50/P99 was **185.3/231.1 ms**, maximum 425.7 ms. Map AoI, sampled every 100 ms (n=3,901), was **249.0/364.8 ms**, maximum 501.0 ms. The 500-ms rule governs installation age, not a guarantee on between-update AoI; maximum inter-install gap was 393.0 ms, with zero gaps above 500 ms. Regional AoI sample counts are HIGH 3,580, TRANSITION 58, LOW_CORE 263.

Map class integrity passed for 117,314 objects: 113,303 Vehicle and 4,011 Pedestrian, zero unknown types. All 57,561 objects on the 1,518 scored decision frames joined exactly to their prediction classes. The final-frame drain completed in 0.168 s: final sent frame 7727 included, 3,085 obligations closed, none open, no deadline extension or timestamp reset. All-transmission payload P50/P99 was 111.26/219.03 kB; total 352,179,209 application bytes.

## Implementation and qualification

New native packages are `rl_agent/splitfusion_m3_route_b_v19` and `rl_agent/splitfusion_m3_route_b_v19_offline_v1`, with V19 child/edge shims. The narrow additions are `spatial_field.py`, `spatial_pose.py`, `spatial_controller.py` and `spatial_checks.py`; registration, native runner/host binding and tests name V19 explicitly. The original V17/R1 GT containment workers, accounting V3.v2, GC, P3, frozen actor, population/seeds, radio, route, retention, direct map path and deadlines remain unchanged. The approved presentation PNG is byte-identical.

Focused regression passed 26/26 checks: exact formula/bounds/determinism/continuity, calibration, actual pose→writer→ACK→actor→checker wiring, stale/missing/future/aborted pose refusal and six deliberate spatial-evidence mutations. Mandatory production verification passed host, runner parent, fresh actual runner CLI preflight, UE child, actual slim edge and real GT-service worker startup. All 28 source/registration negative controls rejected; no CARLA/OAI/live inference started in that preparation proof. The spatial offline binding also rejected a shifted ellipse, the constant V17 channel and a changed actor-input definition. Unchanged qualifications were reused, not benchmarked again.

The registered assessment ran once, as did exact rescoring, retention verification and numerical analysis. Retention is 100/100 complete records, 100/100 GT references resolved, strict verification PASS, zero faults and zero retained host bytes at close. [Preparation notes](PREPARATION_NOTES.md) preserve preparation-only failures and the route-figure clipping correction; none caused a replacement capture or a modified runtime verdict.

## Artifacts and reproduction

The sole registered attempt is in the adjacent deployment worktree at `rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261012_m3_route_b_v19/spatial_prospective_01`. Its actual command and prelaunch disk/idle/binding snapshot are in [PRELAUNCH.json](capture_01/PRELAUNCH.json); numeric exit status is 0 in [LAUNCH_RESULT.json](capture_01/LAUNCH_RESULT.json). This command is historical evidence, **not authorization for another launch**.

Final presentation PNGs, using the existing presentation style:

- [Approved red region and timely or late feedback](presentation_02/01_spatial_region_outcomes.png)
- [Commanded and measured channel, mode, q and payload](presentation_02/02_channel_mode_q_payload.png)
- [Prediction and feedback completion CDFs](presentation_02/03_prediction_feedback_cdf.png)
- [Map freshness CDFs](presentation_02/04_map_freshness_cdf.png)
- [Prediction, feedback and map freshness timeline](presentation_02/05_latency_map_freshness_time.png)

The exact offline commands and stage durations are recorded in [COMMANDS.json](analysis_execution_01/COMMANDS.json) and [RESULT.json](analysis_execution_01/RESULT.json). Numerical sources are [DECISION_TIMELINE.csv](analysis_01/DECISION_TIMELINE.csv), [REGIONS.csv](regions_01/REGIONS.csv), and their hashed manifests. Final rendering is reproducible from this repository with:

```bash
env -u PYTHONPATH CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /usr/bin/python3 -B experiments/splitfusion_m3_route_b_v19/render_spatial_v2.py \
  --analysis experiments/splitfusion_m3_route_b_v19/analysis_01 \
  --layers experiments/splitfusion_m3_route_b_v19/route_layers_01 \
  --regions experiments/splitfusion_m3_route_b_v19/regions_01 \
  --out /tmp/m3_v19_spatial_presentation_reproduction
```

The output path must not exist. No live services are part of this reproduction.

Registration file SHA256: `07eb8c9206bccdf8dbcebbf5653f177a1f4172054645ca63e5350b16025b1e80`.
Full prospective inventory (679 entries): `5493f6eb27c91ce4e2100b3f9b7c044376b91b3512bf24528d5f89efa34bdfb8`.
Runtime inventory (662 entries): `f184f30a603ccf6ce69a11871308f0ce02ebb26babab30a4e5ba40d321549e51`.
Frozen actor weights: `bb9b0d256de79b4498cd622e1daa59edcf43ff3072dc9f675b265aac579ed3ce`.
Approved PNG: `385998cd4cd03cde3e0eb6e37c927c8e85879f0671fc8c4a19560f32c8f4cd75`.

## Timing and cleanup

Timed implementation plus mandatory checks took **12m 12s** from first new source creation to launch; the preceding read-through was not instrumented. The sole host capture lifecycle was **9m 30s** (22:55:53–23:05:23 UTC), including **6m 30s** driving / 308.65 s simulation. Automated assessment, retention verification, analysis and initial figures took **52.1 s** (33.2 s assessment, 7.4 s retention, 5.6 s base analysis, 3.4 s joins, 0.9 s regions, 1.6 s figures); figure review, one axes-limit correction and report preparation are additional offline wall time, recorded through final sealing.

The unchanged 16-GiB gate passed immediately before launch with 29,868,408,832 free bytes. The attempt occupies 7,262,527,488 allocated bytes; filesystem free space decreased by 7,470,997,504 bytes across capture (not an exclusive per-process storage measurement). At the final read-only host audit, **22,370,545,664 bytes remained: 22.37 GB / 20.83 GiB**. Host partition restoration and radio restoration passed; no live processes or containers remained. All evidence and previous verdicts were preserved. No evidence deletion, archival to make space, retraining, commit, push, further capture or spatial-map UI work followed.
