# M3.4 V14 — one full Town10-map BAD_TO_GOOD capture

**Final verdict: `PASS_M3_FULL_ROUTE_INTEGRATION` (29/29 registered checks).** The sole authorized capture completed the full route; the launcher returned 0 and emitted one terminal `CAPTURE_VALID`. There was no retry, retraining, optimization, alternate condition, deletion, commit, push, or M3.3 rerun.

M3.3 remains unchanged and remains `STOP`. Its report SHA256 is `6ef3be42d1473580e44b84e84c3eec2ecdf6c807e7675c1e157d00846c3d7360`; its executed registration file remains `0533c483b3d7a46b6ea87c94e2f15931cd5f22c381e93201c5619da807a046a9`.

## Identity and preflight

V14 reuses the completed V13 pipeline with only the requested reversed channel schedule and the writer/checker repair:

- first 50% of ordered planned-route distance: commanded **8.5 dB**, RFsim noise **-4.0**;
- second 50%: commanded **23.5 dB**, RFsim noise **-12.0**;
- trigger at 634.338698 m of the 1,268.677397-m registered route; one-way GOOD latch;
- `ROUTE_PROGRESS_SUMMARY.json` now writes the top-level `threshold_crossed` boolean required by the registered checker.

The actor, 12 modes, 22 features, GT/GC/accounting, population, route, radio configuration, 170-ms deadline, retention and independent map publication are unchanged.

Registration root in the deployment worktree:

`rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261009_m3_route_b_v14/bad_to_good_prospective_02`

- registration file SHA256 `4ca9a56797734dd9fb443821f9be0c5f7e0343ff4f883b0db3318df7002d999a`;
- canonical registration content SHA256 `7faa74fe2429fd04e4a7041ff423690230fb2676d8ab51b8d4b1f204cf659938`;
- prospective SHA256 `3d100a0786fa07c33e4e1d1e059b117636d909695e0001ac502f3fac64977344`;
- 646-file runtime inventory SHA256 `3d020bb12226f12f96c91ad69452d3614d269f4f2cb6337b1d8a1d0707b12efe`;
- 621-entry exact-source inventory SHA256 `15b8e76fd532c2fc1af349c983fd0239677e49335352070b1d2f39934c6b24a5`;
- execute token `M3_4_ROUTE_B_BAD_TO_GOOD_V14_CANDIDATE_V2_C3900_ONCE_EXECUTE_V1`.

Focused verification passed: 12/12 transition/writer tests, 15/15 runner/binding checks, the unchanged real GT service path, and 14/14 registration-bound assessment checks/negatives. The fresh-process startup proof passed all six roles (host, runner parent, exact runner CLI, UE child, slim edge image, GT service/workers); all host/child/edge/service drift controls were rejected. It records CARLA/OAI/live inference false and no capture directory. The final exact production preflight then passed with 17,746,276,352 bytes free, no foreign I/O process, the expected standing MPS daemons only, GPU 1% utilized, and the capture path absent.

Exact sole live command:

```bash
env -u PYTHONPATH /usr/bin/python3 -B -m rl_agent.splitfusion_m3_route_b_v14.host launch \
  --output rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261009_m3_route_b_v14/bad_to_good_prospective_02 \
  --startup-proof experiments/splitfusion_m3_route_b_v14/startup_verification_01/RESULT.json \
  --execute M3_4_ROUTE_B_BAD_TO_GOOD_V14_CANDIDATE_V2_C3900_ONCE_EXECUTE_V1
```

## Route and transition

- `ROUTE_COMPLETED`, one loop, ordered coverage 301/301 = **1.000**. The run transmitted 3,092 frames; the 3,900-frame ceiling was not the completion criterion.
- Planned/driven/final monotone progress: 1,268.68 / 1,251.85 / 1,267.925 m. Simulation time 309.35 s; route wall time 390.05 s; capture span 389.56 s.
- Accepted outcome `PERMITTED_INTERVENTION`: three registered roadblock interventions and one collision incident; no watchdog abort.
- Population requested/spawned 50/50 vehicles and 50/50 pedestrians; live minima 49 vehicles and 48 pedestrians. Replenishment was three vehicles and 14 pedestrians.
- Halfway crossing: frame 4,169 at 634.481175 m, sample age 10.892 ms. GOOD command issue was 11.121 ms after the crossing sample, ACK was 11.259 ms after it, and command ACK latency was 0.138 ms.
- First actor decision using the ACKed 23.5-dB command: zero-based decision 702, frame 4,172, 174.458 s after first decision capture.
- Exactly one BAD->GOOD switch, no reversion. All 3,940 controller commands were `ACK_ON_TIME`.
- Channel fidelity F1-F5 passed. BAD measured PUSCH median was 8.5 dB and GOOD median was 23.5 dB; automatic selected-MCS medians were 9 and 27. Commanded SNR is the actor-visible target, not measured achieved SNR.

## Whole-run and phase results

Decision phases use the actor-observed ACKed commanded target. Offered/sent/map phases use the actual crossing frame. Radio phases use the first ACKed GOOD command. The small causal boundary offsets are therefore intentional.

| metric | whole run | BAD first half | GOOD second half |
|---|---:|---:|---:|
| offered / sent frames | 3,094 / 3,092 | 1,412 / 1,410 | 1,682 / 1,682 |
| decisions / holds / fallbacks | 1,529 / 1,528 / 35 | 702 / 701 / 7 | 827 / 827 / 28 |
| prediction completed | 1,528/1,529 (99.935%) | 701/702 (99.858%) | 827/827 (100%) |
| prediction completed by 170 ms | 1,380/1,529 (90.255%) | 564/702 (80.342%) | 816/827 (98.670%) |
| feedback received | 1,528/1,529 | 701/702 | 827/827 |
| feedback by 170 ms | 964/1,529 (63.048%) | 296/702 (42.165%) | 668/827 (80.774%) |
| prediction P50 / P95 / P99, completed only (ms) | 137.36 / 179.20 / 201.17 | 150.63 / 187.12 / 206.65 | 131.53 / 152.97 / 173.23 |
| feedback P50 / P95 / P99, received only (ms) | 163.64 / 206.63 / 235.56 | 174.12 / 212.93 / 236.12 | 158.36 / 183.68 / 224.74 |
| payload P50 / P99, sent frames (bytes) | 102,088 / 199,292 | 86,272.5 / 199,046.9 | 109,474.5 / 204,062.0 |
| map installs | 3,090 | 1,408 | 1,682 |
| map age P50 / P99 (ms) | 192.76 / 263.17 | 202.91 / 267.49 | 187.00 / 236.76 |
| map AoI P50 / P99 (ms) | 257.35 / 374.24 | 265.53 / 376.25 | 251.49 / 373.23 |
| measured PUSCH P05 / P50 / P95 (dB) | mixed phases | 7.5 / 8.5 / 8.5 (n=39,758) | 23.0 / 23.5 / 24.0 (n=24,958) |
| mode counts | 11×1,458; 8×37; 9×28; 10×6 | 11×696; 8×2; 10×4 | 11×762; 8×35; 9×28; 10×2 |
| q×10^4 P50 | 6,221.5 (mode 11) | 6,820.5 | 5,991.0 |

The missing decision was the terminal `SUPERSEDED_PENDING` prediction: it has no prediction-completion or feedback latency and is retained in every decision denominator. No STALE/timeout timestamp is used as inference latency. Prediction completion, feedback receipt and 170-ms feedback timeliness are separate boundaries.

The end drain waited 0.173 s, completed its one in-flight send, and closed all 3,092 obligations: 3,090 `RESULT_INSTALLED`, two `SUPERSEDED_PENDING`, zero open and zero expired on the real clock. All 564 late-feedback predictions with results were still installed; feedback lateness did not suppress independent map publication. There were no map write/parse errors and no update gap above 500 ms.

## Quality, replay and map integrity

Localisation denominators include all eligible GT objects, including misses. Undefined quality is reported separately and no offline prediction substitutes for it.

| phase / class | eligible GT | within tolerance | recall | 3-m precision | pooled segmentation IoU |
|---|---:|---:|---:|---:|---:|
| whole vehicle, 1.0 m | 4,321 | 3,440 | 0.796 | 0.941 | 0.867 |
| whole pedestrian, 1.2 m | 2,125 | 871 | 0.410 | 0.680 | 0.498 |
| BAD vehicle, 1.0 m | 1,810 | 1,359 | 0.751 | 0.917 | 0.809 |
| BAD pedestrian, 1.2 m | 951 | 429 | 0.451 | 0.668 | 0.515 |
| GOOD vehicle, 1.0 m | 2,511 | 2,081 | 0.829 | 0.959 | 0.907 |
| GOOD pedestrian, 1.2 m | 1,174 | 442 | 0.376 | 0.692 | 0.486 |

Quality was defined for 1,474 decisions. Fifty-four scored decisions had undefined quality, and the one unscored terminal decision brings the total unavailable-quality count to 55. By phase: BAD 661 defined, 40 undefined-scored plus one unscored; GOOD 813 defined and 14 undefined-scored.

Actor replay was exact for 1,529/1,529 spans/decisions. Exact archive rescoring passed 1,528/1,528 scored predictions, with 1,479 previous-outcome pairs and zero refusals. Map conversion typed 106,607/106,607 objects (102,287 Vehicle, 4,320 Pedestrian), with zero unknown. The decision-frame join checked 52,626 objects with zero class mismatches and zero unjoined. Pose checks had zero rejection in UE, prep and score paths. Storage accounting recorded 6,116/6,116 calls with zero dropped, errors or foreign frames; the GC safety stop is null.

Scene context differs across route halves: camera-SI P50 was 116.526 BAD versus 115.230 GOOD; radar-P40 P50 was 0.4025 versus 0.4098. These phase results describe one evolving route and do not isolate a channel-only causal effect.

## Presentation and reproducibility

All figures are PNG. Principal requested views:

- `presentation_03/06_route_feedback_outcomes.png`: planned/completed route with timely versus late feedback;
- `route_layers_02/01_route_channel.png`: commanded BAD_TO_GOOD target, explicitly labelled as commanded;
- `route_layers_02/02_route_scene_complexity.png`: camera SI and radar P40;
- `route_layers_02/03_route_payload_freshness.png`: payload and map freshness;
- `route_layers_02/04_route_measured_channel.png`: measured per-transmission PUSCH SNR and final MCS;
- `presentation_03/09_action_scene_channel_time.png` and `10_action_scene_channel_frames.png`: mode/q/payload/commanded-SNR timeline;
- `presentation_03/01_completion_cdf.png`, `02_map_freshness_cdf.png`, `03_localisation_cdf.png`, and `05_payload_p50_p99.png`: denominator-labelled completion, freshness, conservative localisation and payload views;
- `phase_analysis_01/02_transition_timeline.png`: the actual BAD->GOOD transition.

The core analysis ran once. The first route-layer render stopped before writing outputs because the initial decision correctly had no prior map-install age. The empty `route_layers_01` directory is preserved; the minimal renderer correction treats that optional value as missing/NaN, and the unchanged plot design completed in `route_layers_02`. Visual review then found a legacy hardcoded `GOOD->BAD` annotation in the shared presentation renderer. `presentation_01` is preserved, and the partial failed `presentation_02` is also preserved. The final sealed `presentation_03` derives the direction and 8.5->23.5-dB commanded values from the recorded transition; all 11 PNGs pass the input-hash and 3840x2160 checks. Its output-manifest SHA256 is `5b0ec730b560a5a4a1a6c6ecc9647790f989505fe3a4d8f0fbaf1fd9485dad51`, and the corrected renderer SHA256 is `90ca1c677d3cc20d2a870945d4fdf22f54dfec69127a8c99cba08ade3c9b72fe`.

Primary machine-readable artifacts are `analysis_01/METRICS.json`, `phase_analysis_01/PHASE_METRICS.json`, `analysis_summary_01/ANALYSIS_SUMMARY.json`, and `registered_assessment_01/ASSESSMENT.json`.

## Durations, cleanup and storage

- recorded preparation: **471 s** (23:38:56Z-23:46:47Z), explicitly a conservative lower bound because initial inspection preceded the timestamp;
- live capture command: **595 s** (23:46:47Z-23:56:42Z);
- post-capture assessment plus analysis/plots through the corrected-presentation cleanup checkpoint: **673 s** (23:56:42Z-00:07:55Z); nested core analysis/plot interval 149 s.

The target-SNR restore check passed, both pre-stop restore commands returned 0, CPU partition restoration passed, and no experiment CARLA, gNB, UE or Docker container remained. Retention finalized cleanly with 100/100 `RETAINED_COMPLETE`; no evidence was deleted.

Free space changed from 17,745,764,352 bytes at launch to 10,263,560,192 bytes immediately after launch: **7,482,204,160 bytes consumed**. The final post-presentation checkpoint had **10,229,698,560 bytes free**. The sealed prospective/capture tree was 7,236,081,700 bytes immediately after capture.

M3.4 stops here. No subsequent capture is authorized or performed.
