# SceneSense Agent Monthly Checklist

Living checklist aligned with `2026_SceneSense-Agent_Research_Proposal_6Month_DRAFT.docx`.

Use this file to keep the work tied to the proposal: every experiment should answer either a baseline, metric, controller, guardrail, spatial-map, or demo question.

## Project North Star

Learn a network-aware split-inference control policy that reduces payload/latency while preserving task utility.

The controller should eventually choose operating points such as:

- AE channels, where supported.
- ROI threshold, where supported.
- Quantization level.
- Frame send/skip.
- Redundancy add/drop.

The policy is only acceptable if task guardrails are respected:

- Object-detection AP / recall should not silently collapse.
- Segmentation mIoU and foreground IoU should remain above configured limits.
- Pedestrian, cyclist, small-object, and safety-critical recall should be protected.

## Baseline Experiment Families

| Family | Sensor/Input | Output Used | Purpose |
| --- | --- | --- | --- |
| Camera-only OD | RGB camera | Boxes / detections | Compare against earlier OD split-inference pipeline. |
| Camera-only SEG | RGB camera | Semantic mask | Compare against earlier segmentation split-inference pipeline. |
| RGB+radar fusion as OD | RGB + radar tensor | Object head: boxes, world position, yaw, size | Evaluate fusion object/localization quality. |
| RGB+radar fusion as SEG | RGB + radar tensor | Segmentation head: mask | Evaluate fusion segmentation quality. |
| Spatial-map fusion | Outputs from one or more clients | Fused object map | Support occlusion-aware physical-AI experiments later. |

Important note: the new RGB+radar fusion model produces both segmentation and object/localization outputs. For evaluation, treat OD and SEG as separate task metrics even when they come from the same model run.

## Month 1: Baselines, Transport, Metrics, and Schema

Proposal exit criterion:

> Repeatable OD/SEG traces with bytes, latency, loss, AP/mIoU, foreground IoU, and class-specific misses.

### 1. Preserve and Reproduce Baselines

- [x] Keep supervisor-provided fusion scripts untouched in `PythonAPI/neu_collab`.
- [x] Run original RGB+radar pole fusion baseline locally.
- [x] Confirm spatial-map server updates from the two pole streams.
- [x] Confirm split-inference back half returns mask + object outputs to the pole client.
- [x] Copy the RGB+radar fusion baseline into `abiodun/` for our editable version.
- [x] Run copied `abiodun/` RGB+radar pole fusion baseline locally.
- [x] Document exact local baseline commands for:
  - [x] Spatial-map server.
  - [x] Pole stream 1.
  - [x] Pole stream 2.
  - [x] Health checks and viewer URLs.

### 2. Establish 5G as Transport Medium

This is a transport baseline only. Do not add low-SNR, bandwidth throttling, packet-loss stress, or resource-limiting experiments yet.

- [x] Camera-only OD split inference over OAI 5G.
- [x] Camera-only SEG split inference over OAI 5G.
- [x] Create RGB+radar fusion OAI transport script/container wiring.
- [x] Document RGB+radar fusion OAI run commands.
- [x] Add two-UE OAI bring-up/check scripts and multi-UE fusion runbook.
- [x] RGB+radar fusion split inference over OAI 5G.
- [x] RGB+radar fusion split inference over two OAI UEs.
- [x] Confirm stream 1 uses UE1 (`10.0.0.2`) and stream 2 uses UE2 (`10.0.0.3`).
- [x] Confirm pole client still receives mask + object results when the split path crosses OAI 5G.
- [x] Confirm spatial-map server still updates while fusion split traffic crosses OAI 5G.
- [x] Record OAI IP/port flow:
  - [x] UE/front host IP.
  - [x] Core/container/back-half IP.
  - [x] Feature UDP ports.
  - [x] Result UDP ports.
  - [x] Spatial-map UDP port.

### 3. Build Repeatable CARLA Scenario Harness

This maps to the proposal phrase: "Instantiate repeatable CARLA scenes with controlled object density, occlusions, ego motion, intersection layouts, OD/SEG split routes, static compression settings, and network stress traces."

Month 1 goal is not to create every final scenario. The goal is to create a small, repeatable scenario battery that we can rerun while changing model/network/compression settings.

- [x] Create first scenario harness workspace under `scenesense_scenarios/`.
- [x] Smoke-run each starter scenario and inspect visually in CARLA.
- [x] Add ego-mounted RGB/radar smoke-test sensor hooks for ego-view inspection.
- [x] Define at least one simple baseline scene:
  - [x] Low object density.
  - [x] Clear line of sight.
  - [x] Known camera/radar placement.
  - [x] Repeatable spawn seed.
- [x] Define at least one crowded scene:
  - [x] More vehicles.
  - [x] More pedestrians.
  - [x] Higher object overlap.
- [x] Define at least one occlusion-focused scene:
  - [x] Pedestrian or vehicle partially hidden.
  - [x] Object appears near intersection or blind spot.
  - [x] Ego-facing occlusion crossing setup with optional scripted ego/target motion.
  - [x] Clean intersection truck/pedestrian occlusion scenario scaffold.
  - [x] Visible crossing failure control validated with target collision/near-miss logs.
  - [x] Curbside parked-vehicle hidden-pedestrian failure validated at spawn 152 (`20260529_201805...`, target collision logged).
  - [x] Lock Month 1 canonical occlusion baseline to hidden-pedestrian dart-out; leave sidewalk prewalk polish for later.
  - [x] Add optional opposite-lane helper vehicle camera path for ego-blind/helper-visible evidence.
  - [x] Add optional non-interfering moving helper vehicle controller and movement summary for ego-blind/helper-visible evidence.
  - [x] Accept curbside hidden-dart-out as the Month 1 baseline; defer visual-realism polish to later demo work.
  - [ ] Right-turn truck/pedestrian hidden-hazard scenario visually validated.
  - [x] Add targeted scout for better right-turn occlusion anchors.
  - [ ] Run right-turn anchor scout and select a more realistic intersection.
  - [x] Occluded crossing failure visually validated from ego camera and observer view.
  - [x] Add evidence-pack support for actor ground truth, event-window CSVs, and ego/helper RGB frames.
  - [x] Fix curbside target motion default to avoid AI sidewalk routing and expose crossing-progress telemetry.
  - [x] Add evidence-pack validator for canonical occlusion run folders.
  - [x] Run canonical evidence-pack validation so ground truth confirms the object exists even if ego view is late/partial (`20260601_183145...`, validator PASS: target progress 0.698, min distance 2.62 m, 80 ego + 80 helper frames).
  - [x] Run collision-tuned evidence validation with forced crossing geometry (`20260602_101912...`, `--require-collision` PASS: 19 target collisions, target progress 0.495, 130 ego + 131 helper frames). Note: visual pedestrian animation still slides in this collision-forcing mode; treat as demo polish, not Month 1 evidence blocker.
  - [x] Add ego-route-location trigger for animated pedestrian collision calibration.
  - [x] Validate animated walker-control collision using ego route-location trigger (`20260602_104540...`, `--require-collision` PASS: 11 target collisions, route lead about 26 m, target progress 0.488).
  - [x] Lock final animated curbside evidence demo (`20260602_125157...`, `walker_control`, `--require-collision` PASS: 9 target collisions, 0.569 target progress, 88 ego + 89 helper RGB frames).
- [x] Define ego-motion settings:
  - [x] Static/parked ego or pole baseline.
  - [x] Slow-moving ego follow-up.
- [x] Define OD/SEG split routes in `SCENESENSE_MONTH1_TRACE_MATRIX.md`:
  - [x] Camera-only OD route.
  - [x] Camera-only SEG route.
  - [x] RGB+radar fusion route evaluated as OD.
  - [x] RGB+radar fusion route evaluated as SEG.
- [x] Define static compression settings for trace collection in `SCENESENSE_MONTH1_TRACE_MATRIX.md`:
  - [x] Quantization options.
  - [x] Entropy coder options.
  - [x] AE channel options, where supported.
  - [x] ROI threshold options, where supported.
- [x] Define first network stress trace placeholders in `SCENESENSE_MONTH1_TRACE_MATRIX.md`:
  - [x] Local no-stress baseline.
  - [x] OAI 5G transport baseline with no intentional impairment.
  - [x] Logged latency/loss traces for later replay.
  - [x] Delay/loss/bandwidth stress settings reserved for Month 2/3.

### 4. Build the Metrics Foundation

Use `payload_fusion_handoff_readme.md` as the reference for payload-characterization output structure and analysis conventions.

- [x] Create SceneSense run-folder structure under `metrics_logs/scenesense_runs/`.
- [x] Add per-stream RGB+radar fusion metrics CSV logging.
- [x] Add per-stream manifest and resolved-config JSON output.
- [x] Add automatic `run_group` labeling so related stream folders are easy to pair during analysis.
- [x] Add lightweight OAI/network snapshot collector script.
- [x] Add first-pass application metrics summary/plot helper.
- [x] Add lightweight UE tunnel network time-series sampler.
- [x] Extend analysis helper to include matching network summaries/plots.
- [x] Document application/network/T-tracer logging plan.
- [x] Add OAI T-tracer smoke-test launch/record/extract helpers.
- [x] Validate T-tracer smoke capture/replay produces populated gNB/UE raw traces and CSVs.
- [x] Enhance T-tracer smoke profile with gNB LCID, PUCCH, RLC, and PDCP events.
- [x] Add gNB MAC stdout parser for BLER, HARQ, SNR, MCS, PRB, MAC bytes, and LCID bytes.
- [x] Add local NR UE decoded-grant trace event for UE-side RL network state.
- [x] Validate enhanced T-tracer/PDCP/gNB-stdout metrics on a live OAI fusion run with matching `run_group`.

Minimum per-run metadata:

- [x] Script name and git/status note.
- [x] CARLA town/map.
- [x] Sensor placement: ego vehicle, pole, or other.
- [x] Model/checkpoint path.
- [x] Front device and back device.
- [x] Resolution and FPS.
- [x] Quantization mode.
- [x] Entropy coder.
- [x] UDP ports and IPs.
- [x] Local run vs OAI 5G run.

Network/split metrics:

- [x] Feature payload bytes.
- [x] Result payload bytes.
- [x] Chunk count.
- [ ] Encode time.
- [ ] Decode time.
- [x] Front-half inference time.
- [x] Back-half inference time.
- [x] Round-trip time.
- [x] Timeout/missed-result count.
- [x] Approximate FPS.
- [x] Packet-loss or missing-frame indicators where available.
- [x] UE tunnel RX/TX bitrate, packet counters, drops/errors, and optional ping RTT/loss.
- [x] UE decoded grant metrics via `NRUE_MAC_DCI_GRANT`: UL/DL MCS, RBs, symbols, TBS, HARQ, NDI/RV.
- [x] Clean UE T-tracer profile that excludes legacy/suspicious UE PHY files by default.
- [x] Windowed UE grant analyzer for scheduled Mbps, grant rate, MCS, RBs, symbols, TBS, and retransmission indicators.
- [x] OAI RAN metrics via logs/T-tracer/stdout: gNB SNR/SINR-like summaries, MCS, PRBs, BLER, HARQ, RLC/PDCP/LCID bytes.
- [x] Validate T-tracer CSV extraction on a live OAI fusion run and align radio metrics with application metrics by `run_group`.
- [ ] Optional later: add a clean NR UE CSI/CQI trace if raw UE-side CQI/SNR becomes necessary beyond decoded-grant features.

Task metrics:

- [x] Camera-only OD: AP or first-pass precision/recall/object recall. Fresh
  loopback/OAI traces collected with `--enable-od-gt` and analyzed with
  `scripts/analyze_camera_od_metrics.py`:
  `month1_camera_od_loopback_20260604_153409.csv` and
  `month1_camera_od_oai_20260604_153845.csv`. Overall: 2380 frames, 9983 GT
  objects, 2294 predicted objects, 1047 matches at IoU 0.5, global recall
  0.105, global precision 0.456, mean matched IoU 0.713. Loopback recall /
  precision: 0.112 / 0.526; OAI recall / precision: 0.092 / 0.358. Vehicle
  recall: loopback 0.226, OAI 0.140. Person recall: loopback 0.047, OAI 0.066.
  Output summary:
  `metrics_logs/month1_camera_od_analysis/month1_camera_od_quality_20260604_154333.md`.
- [x] Camera-only SEG: mIoU, foreground IoU, class IoU. Loopback quality run
  `month1_camera_seg_loopback_20260604_145934.csv` analyzed with
  `scripts/analyze_camera_seg_metrics.py`: 451 frames, 450 GT frames,
  foreground/binary IoU mean 0.195, 3-class macro mIoU mean 0.508, vehicle IoU
  mean 0.172. No visible person GT pixels were present, so person IoU remains
  unmeasured for this trace. OAI SEG-quality repeat can be collected later with
  `--enable-semantic-gt` using `SCENESENSE_MONTH1_COMMANDS.md`.
- [x] RGB+radar fusion object head: first-pass object recall, localization error, yaw/dimension error, and score-threshold sensitivity (`fusion_od_transfer_20260604_01`; deck: `SceneSense_Fusion_Model_Transferability_OD_SEG.pptx`). Note: full confidence calibration/ECE remains a later polish item.
- [x] RGB+radar fusion segmentation head: mIoU, foreground IoU, vehicle/person IoU (`pole_vs_ego_transfer_presentation`; person IoU is zero in the transfer run and should not be over-interpreted without visible person GT).
- [ ] Class-specific misses, especially vulnerable or small objects.

### 5. Define Ground Truth and Evaluation Path

- [x] Identify the CARLA ground-truth source for each task:
  - [x] Semantic segmentation camera for masks.
  - [x] CARLA actors/transforms/bounding boxes for object position and size.
  - [x] Radar detections/raster for fusion input validation.
- [x] Decide where evaluation logs live under `abiodun/`.
- [x] Decide CSV/JSON schema for run metrics.
- [x] Decide whether evaluation is online during the demo or offline from saved traces.
- [x] Create a small repeatable test scene for smoke-test metrics.

### 6. Understand Prior Payload Characterization Work

Month 1 goal: understand and reuse the prior OD-vs-SEG payload-comparison structure before creating new fusion payload experiments.
The six-month proposal does not name a specific `od_seg_fair_latency_*` run
folder; that root is a handoff-specific reference artifact. Treat it as useful
provenance if recovered, not as a proposal-mandated Month 1 blocker.

- [x] Read `payload_fusion_handoff_readme.md`.
- [x] Inspect slide-level OD-vs-SEG traffic-characterization artifact: `AI_traffic_characterization_IDCC_template.pptx`. It summarizes OD/SEG payload sizing, ROI/AE/quantization candidates, and 5QI burst-volume gaps.
- [x] Create current Month 1 camera-only OD-vs-SEG latency/payload comparison
  over loopback and OAI 5G. Artifact:
  `SceneSense_Camera_OD_SEG_Latency_Comparison.pptx`; evidence folder:
  `metrics_logs/scenesense_analysis/camera_od_seg_latency_20260604/`.
  Headline: OD median RTT loopback/OAI `8.2/74.9 ms`; SEG median RTT
  loopback/OAI `13.4/107.9 ms`; SEG median feature payload is about `4.6x`
  OD. OAI config slide records RFsim, band n78, 30 kHz SCS, 106 PRB
  approximately 40 MHz, 5 ms TDD pattern, DNN `oai`, SST 1, 5QI 9.
- [ ] Inspect the completed OD-vs-SEG comparison root:
  - [ ] `metrics_logs/od_seg_latency_comparison/od_seg_fair_latency_recovery_20260520_220356/`. Not present in the current local/remote mirror; keep open unless the raw root is recovered.
- [ ] Understand the key output files:
  - [ ] `per_frame_metrics.csv`.
  - [ ] `run_manifest.json`.
  - [ ] `resolved_config.json`.
  - [ ] `analysis/payload_summary_by_profile.csv`.
  - [ ] `analysis/latency_summary_by_profile.csv`.
  - [ ] `analysis/quality_summary_by_profile.csv`.
- [x] Keep OD and SEG quality metrics separate:
  - [x] OD uses COCO-style AP / mAP or first-pass recall/precision.
  - [x] SEG uses dense mIoU / class IoU / foreground IoU.
- [ ] Treat no-result or saturated runs as saturation evidence, not valid returned-result quality samples.
- [ ] Reuse the same experiment-root style for future RGB+radar fusion payload characterization.

### 7. Parked Ego-Vehicle Starter Track

Scope for Month 1: prove data collection and retraining feasibility, not final model quality.

- [x] Mount RGB + radar on a parked ego vehicle in CARLA for live split-inference and transferability runs.
- [x] Collect synchronized saved training samples, not just live inference logs: RGB, radar tensor/points, semantic mask, object actor labels, sensor pose, and calibration. Remote smoke PASS: `parked_ego_fusion_training_smoke_20260604`, 30 manifest rows, 474 actor-derived object rows, vehicle/person labels, RGB shape `(480, 854, 3)`, mask shape `(480, 854)`, radar tensor shape `(4, 432, 768)`.
- [x] Verify saved samples match the expected fusion training schema. Validator PASS with no errors/warnings; all 30 inspected samples include mask classes `0/1/2`, radar tensors, RGB, and linked object labels. Target dry-run PASS: 30/30 samples build `(7, 432, 768)` fusion inputs, `(432, 768)` segmentation targets, `(1, 432, 768)` heatmaps, `(10, 432, 768)` regression maps, and `(64, 9)` GT object tensors; 369 valid vehicle object targets.
- [x] Confirm whether the original training driver exists. Local scan found fusion model/object-target helpers but no obvious standalone SceneSense fusion training driver; V2Xverse/OpenCOOD trainers are present but belong to a different stack.
- [ ] If training code exists: run a tiny fine-tuning/smoke training job. Leave open unless a remote-only original trainer is found or a recreated trainer is implemented.
- [x] If training code is missing: list the missing pieces needed to recreate it. See `FUSION_TRAINING_DRIVER_GAP_ANALYSIS.md`.

### 8. Freeze the First RL Schema

No full RL training required in Month 1, but the schema should be clear enough that traces can support it later.

- [x] State candidates in `SCENESENSE_RL_SCHEMA.md`:
  - [x] Scene density / object count.
  - [x] Foreground fraction.
  - [x] Vulnerable-object presence.
  - [x] Model confidence / uncertainty.
  - [x] Payload size.
  - [x] Latency / RTT.
  - [x] Timeout or loss indicators.
- [x] Action candidates in `SCENESENSE_RL_SCHEMA.md`:
  - [x] AE channel setting, where supported.
  - [x] ROI threshold, where supported.
  - [x] Quantization setting.
  - [x] Frame send/skip.
  - [x] Redundancy add/drop.
- [x] Reward sketch in `SCENESENSE_RL_SCHEMA.md`:
  - [x] Task utility retained.
  - [x] Minus payload cost.
  - [x] Minus latency cost.
  - [x] Minus loss/timeout cost.
- [x] Guardrail sketch in `SCENESENSE_RL_SCHEMA.md`:
  - [x] Reject/clamp if AP, mIoU, foreground IoU, or class recall drops too far.
  - [x] Use safer fallback settings under low confidence or high loss.

### Month 1 Definition of Done

- [x] Reproducible local commands for camera-only OD, camera-only SEG, and RGB+radar fusion. See `SCENESENSE_MONTH1_COMMANDS.md`.
- [x] Reproducible OAI 5G commands for camera-only OD, camera-only SEG, and RGB+radar fusion transport. See `SCENESENSE_MONTH1_COMMANDS.md`.
- [x] Small repeatable CARLA scenario battery covering simple, crowded, and occlusion-focused cases.
- [x] At least one metrics log format that records network, split-inference, and task data.
- [x] Ground-truth plan confirmed for OD and SEG.
- [x] Parked ego data collection path started: live parked-ego RGB/radar inference, semantic-GT metrics, object-GT logging, pole-vs-ego transfer evidence, and smoke-validated saved training-schema export are in place.
- [x] RL state/action/reward/guardrail schema drafted.

## Month 2: Static Sweeps, 5QI/QoS, and First Controller Harness

Proposal row being covered:

> Implement constrained RL controller over AE/ROI/quantization/scheduling/
> redundancy actions.
>
> Exit criterion: policy can train or evaluate against static policies using
> the same logged metrics.

Working Month 2 interpretation:

> Static payload/task/latency Pareto curves, first 5QI/QoS comparison, and an
> offline controller harness that can score candidate actions against logged
> traces with guardrail/fallback accounting.

Month 2 goal is not to claim a final RL policy. The goal is to produce the
static baselines and QoS evidence that any learned policy must beat.

### 1. Freeze the Month 2 Experiment Matrix

- [ ] Select the Month 2 scenario subset:
  - [ ] Low-density clear scene.
  - [ ] Crowded scene.
  - [ ] Curbside hidden-pedestrian scenario.
  - [ ] Optional parked-ego fusion transfer scene if supervisor wants parked ego prioritized.
- [ ] Select the Month 2 route subset:
  - [ ] Camera-only OD.
  - [ ] Camera-only SEG.
  - [ ] RGB+radar fusion as SEG.
  - [ ] RGB+radar fusion as OD.
- [ ] Define canonical run durations:
  - [ ] Short smoke: 30-60 s.
  - [ ] Measurement run: 180 s.
  - [ ] Long stability run: 300-600 s, only after smoke passes.
- [ ] Define canonical run-group naming:
  - [ ] Static sweeps: `month2_static_<route>_<profile>_<transport>`.
  - [ ] 5QI sweeps: `month2_5qi_<value>_<route>_<transport>`.
  - [ ] Controller replay: `month2_controller_replay_<date>`.
- [ ] Update or create the Month 2 command sheet/runbook section once the first
  smoke commands are validated.

Completion criteria:

- [ ] A table exists listing scenario x route x transport x profile combinations.
- [ ] Every planned run has a reproducible command, expected output folder, and
  analyzer command.
- [ ] Remote sync instructions are recorded for any edited scripts/config files.

### 2. Static UE-Side Payload/Task Sweeps

These sweeps establish the fixed operating points that the future controller
must beat.

- [ ] Run camera-only OD static sweeps where supported:
  - [ ] Resolution/profile baseline.
  - [ ] Quantization/compression profile, if exposed by the active OD route.
  - [ ] Confidence/score threshold sweep for returned detections.
  - [ ] Optional frame-rate or send-rate sweep if supported.
- [ ] Run camera-only SEG static sweeps where supported:
  - [ ] Resolution/profile baseline.
  - [ ] Segmentation input size sweep.
  - [ ] Saliency/drop sweep if using the route with `--saliency-drop-q`.
  - [ ] Mask output size: model vs camera, where relevant.
- [ ] Run RGB+radar fusion static sweeps:
  - [ ] Quantization: `per_tensor_uint8`, `per_channel_uint8`, `per_channel_uint4`
    where supported.
  - [ ] Entropy coder: `zlib`, `zstd` if installed, and `none` for diagnosis.
  - [ ] Object score threshold / top-k sweep for fusion_as_od.
  - [ ] Semantic-GT enabled sweep for fusion_as_seg.
- [ ] For each accepted static profile, collect:
  - [ ] Application metrics CSV.
  - [ ] Run manifest/resolved config.
  - [ ] Task-quality summary: OD recall/precision or fusion OD recall/localization.
  - [ ] SEG summary: mIoU, foreground IoU, vehicle/person IoU where visible.
  - [ ] Payload summary: bytes/frame, chunks/frame, compression ratio.
  - [ ] Latency summary: front, back, RTT, timeout/no-result rate.

Completion criteria:

- [ ] At least one static Pareto plot exists per priority route:
  payload vs latency vs task utility.
- [ ] A "best fixed static policy" is identified for each priority route.
- [ ] A "lowest-byte unsafe policy" is identified to motivate guardrails.
- [ ] Runs that saturate or time out are labeled as saturation evidence, not
  valid task-quality samples.

### 3. OAI 5QI/QoS Experiments

Hypothesis: the current default OAI QoS path is too close to best-effort/eMBB
behavior for safety-critical cooperative perception. Month 2 should test
whether changing the 5QI/QoS profile improves application latency, result
receive rate, and task utility under fixed payloads.

- [ ] Record current OAI QoS baseline:
  - [ ] DNN name.
  - [ ] S-NSSAI / SST / SD.
  - [ ] Active 5QI.
  - [ ] AMBR UL/DL.
  - [ ] UE tunnel IPs.
  - [ ] Any visible QFI/DRB mapping in OAI logs/traces.
- [ ] Create reversible OAI config variants or scripts for candidate 5QIs:
  - [ ] `5QI 9`: current/default baseline.
  - [ ] `5QI 7`: live video / interactive non-GBR candidate.
  - [ ] `5QI 2`: conversational video GBR candidate.
  - [ ] `5QI 79`: V2X message candidate.
  - [ ] `5QI 85` or `86`: low-delay V2X / remote-driving style candidate.
  - [ ] `5QI 88`, `89`, or `90`: split-AI / visual-content candidates if OAI accepts them.
- [ ] For each 5QI candidate, verify whether the configured value is actually
  active:
  - [ ] Core config snapshot saved.
  - [ ] CN/gNB/UE restart sequence documented.
  - [ ] UE registration/session success confirmed.
  - [ ] QFI/5QI/DRB evidence captured if available.
  - [ ] If OAI does not expose/accept the candidate cleanly, record the failure
    and keep the baseline unchanged.
- [ ] Run fixed-payload OAI traces for each accepted 5QI:
  - [ ] Same scene.
  - [ ] Same route/model/checkpoint.
  - [ ] Same payload profile.
  - [ ] Same duration.
  - [ ] Same UE count.
  - [ ] Same chunk size.
- [ ] Run at least one 5QI sweep under background load:
  - [ ] No background load.
  - [ ] Fixed iperf uplink/downlink load.
  - [ ] Optional two-UE competing perception load.
- [ ] Analyze 5QI effects:
  - [ ] RTT median/p95/p99.
  - [ ] Timeout/no-result rate.
  - [ ] Receive rate.
  - [ ] UE tunnel drops/errors.
  - [ ] gNB/UE grants: MCS, RBs, TBS, HARQ/retransmission proxy.
  - [ ] Task utility: OD recall/localization or SEG IoU.

Completion criteria:

- [ ] A 5QI comparison table exists with config evidence, app metrics, network
  metrics, and task metrics.
- [ ] A plot shows whether lower-delay/V2X/split-AI 5QIs improve application RTT
  and receive rate under fixed payload.
- [ ] The conclusion explicitly says whether static 5QI alone is enough, or
  whether payload control is still required.

### 4. First Offline Controller Harness

This is the first bridge from measurement to control. It should replay logged
traces and score decisions offline before any online RL touches CARLA/OAI.

- [ ] Implement trace loader that joins:
  - [ ] Application metrics by run group / stream / frame or timestamp window.
  - [ ] Network sampler metrics.
  - [ ] T-tracer / gNB metrics where available.
  - [ ] Scenario metadata.
  - [ ] Task-quality summaries.
- [ ] Implement action-profile catalog:
  - [ ] Safe/high-quality profile.
  - [ ] Balanced profile.
  - [ ] Low-byte profile.
  - [ ] Hazard/guarded profile.
  - [ ] Route-specific unsupported actions are masked.
- [ ] Implement reward scorer:
  - [ ] Task utility retained.
  - [ ] Minus payload cost.
  - [ ] Minus latency cost.
  - [ ] Minus timeout/loss cost.
  - [ ] Minus stale-map or vulnerable-object penalty where available.
- [ ] Implement first non-RL baselines:
  - [ ] Always-safe/send-everything.
  - [ ] Always-low-byte.
  - [ ] Best fixed profile.
  - [ ] Network-only rule.
  - [ ] Task-only rule.
  - [ ] Simple heuristic rule using scene + network state.
- [ ] Optional learning baseline:
  - [ ] Contextual bandit or DQN over discrete action profiles.
  - [ ] Do not use SAC unless continuous knobs are introduced and simple
    baselines are already beaten.

Completion criteria:

- [ ] Controller replay produces a CSV/JSON summary for every baseline policy.
- [ ] Baseline policy comparison plot exists:
  task utility vs bytes vs latency/timeout.
- [ ] The best simple heuristic is identified as the first policy the learned
  controller must beat.
- [ ] No online action execution is enabled until offline replay passes sanity
  checks.

### 5. Guardrail Threshold Draft

Month 2 guardrails can remain offline, but thresholds must become concrete
enough for controller replay.

- [ ] Draft route-specific task floors:
  - [ ] Camera OD recall/precision or AP proxy floor.
  - [ ] Camera SEG foreground IoU / mIoU floor.
  - [ ] Fusion_as_od recall/localization floor.
  - [ ] Fusion_as_seg foreground/vehicle/person IoU floor.
- [ ] Draft vulnerable-object rules:
  - [ ] No frame skip when pedestrian/cyclist/hidden-hazard flag is active.
  - [ ] No aggressive saliency/ROI drop when vulnerable-object confidence is low
    or uncertainty is high.
  - [ ] Safer fallback when map freshness is stale.
- [ ] Draft network fallback rules:
  - [ ] If timeout/no-result rate rises, prefer smaller payload before dropping
    safety-critical frames.
  - [ ] If UE tunnel drops/errors rise, reduce payload detail or send compact
    hazard messages.

Completion criteria:

- [ ] Guardrail thresholds are written in config or a replay script, not only in prose.
- [ ] Replay reports accepted, clamped, and rejected actions separately.
- [ ] Fallback cost is measurable in bytes/latency/task utility.

### 6. Spatial-Map Sharing Groundwork

The full map-sharing RL agent is Month 5, but Month 2 should make the closed-loop
case study measurable.

- [ ] Define spatial-map utility fields needed by the future map-sharing agent:
  - [ ] Object class.
  - [ ] Pose / velocity.
  - [ ] Confidence / uncertainty.
  - [ ] Provenance stream id.
  - [ ] Freshness / age.
  - [ ] Occlusion or hazard flag.
  - [ ] Intended recipient or affected ego vehicle.
- [ ] Define curbside hidden-hazard utility metrics:
  - [ ] Warning lead time before collision / near-miss.
  - [ ] Vulnerable-object recall before collision.
  - [ ] Stale-object rate.
  - [ ] False hazard rate.
  - [ ] Bytes per useful warning.
- [ ] Create a local-only vs cooperative comparison plan:
  - [ ] Ego-only evidence.
  - [ ] Helper/observer evidence.
  - [ ] Spatial-map shared warning.
  - [ ] Send-everything map update baseline.
  - [ ] Compact hazard-only update baseline.

Completion criteria:

- [ ] A replayable curbside run folder can produce warning-lead-time and
  freshness metrics.
- [ ] Spatial-map entries can be joined back to CARLA ground truth and evidence
  traces.
- [ ] There is a clear Month 3/4 path from spatial-map measurement to online map
  sharing.

### Month 2 Definition of Done

- [ ] Static sweeps produce at least one payload/latency/task Pareto curve for
  camera-only SEG or fusion SEG.
- [ ] Static sweeps produce at least one OD/localization trade-off curve for
  camera-only OD or fusion_as_od.
- [ ] 5QI/QoS sweep has at least baseline 5QI 9 plus two accepted alternative
  5QI candidates, or documented OAI blockers if alternatives fail.
- [ ] OAI runs include matching application, UE tunnel, and radio/grant metrics
  under a shared `run_group`.
- [ ] Offline controller replay can compare at least five policies: safe,
  low-byte, best fixed, network-only, and heuristic scene+network.
- [ ] Guardrail replay reports accept/clamp/reject counts and fallback costs.
- [ ] A Month 2 slide/report summarizes:
  - [ ] Static Pareto curves.
  - [ ] 5QI/QoS findings.
  - [ ] First controller baseline results.
  - [ ] Remaining gap before online RL.

## Month 3: Guardrail Stress Tests

- [ ] Add controlled stress profiles: jitter, delay, queueing, packet loss, or bandwidth limits.
- [ ] Test whether byte-minimizing choices damage AP/mIoU/class recall.
- [ ] Add deterministic guardrail layer.
- [ ] Compare learned/proposed actions with and without guardrails.
- [ ] Produce plots showing guardrail rejection rate, fallback cost, and protected task metrics.

## Month 4: Physical-AI Spatial Map Ingestion

- [ ] Convert accepted split-model outputs into spatial-map entries.
- [ ] Store class, pose, velocity, confidence, provenance, freshness, and occlusion state.
- [ ] Validate map entries against CARLA ground truth.
- [ ] Measure map freshness, stale-object rate, false hazard rate, and localization error.

## Month 5: Learned Map Sharing

- [ ] Define map-sharing actions: what to send, when to send, and at what payload cost.
- [ ] Train or evaluate map-sharing policies under bandwidth/freshness constraints.
- [ ] Prioritize occluded or safety-critical objects.
- [ ] Compare learned sharing against simple periodic or send-everything baselines.

## Month 6: Navigation Override Demo and Paper Package

- [ ] Build intersection scenario with occluded hazard.
- [ ] Use shared spatial map to warn or override an autonomous vehicle.
- [ ] Measure time-to-warning, braking/replanning latency, avoided collisions, and unnecessary overrides.
- [ ] Prepare paper/demo package:
  - [ ] Method figure.
  - [ ] Evaluation tables.
  - [ ] Ablation plots.
  - [ ] Scenario screenshots.
  - [ ] Demo narrative.
  - [ ] Invention-disclosure notes.

## Open Decisions

- [ ] Final location for metrics logs and schemas.
- [ ] Whether RGB+radar fusion over 5G uses one combined client process first or separate front/back roles immediately.
- [ ] Whether the spatial-map server runs on the UE/front host, the OAI/core host, or a third machine.
- [ ] Whether parked ego retraining uses the existing fusion model unchanged or a smaller first-pass model.
- [ ] Which task metric thresholds become hard guardrails.
