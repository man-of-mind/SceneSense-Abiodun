# SceneSense Agent Monthly Checklist

Living checklist aligned with `2026_SceneSense-Agent_Research_Proposal_6Month_DRAFT.docx`.

Last reconciled with repository evidence: **2026-08-14**.

Use this file to keep the work tied to the proposal: every experiment should answer either a baseline, metric, controller, guardrail, spatial-map, or demo question.

## Current Status Snapshot

- **Month 1:** complete (baselines, transport, scenarios, logging, metrics, schema).
- **Month 2:** complete. The accepted 36-profile zstd table, executable seven-
  profile catalog, reward-v5 scorer, grouped replay loader, and non-RL ladder
  (fixed/rule/greedy/LinUCB/MPC) all exist. The richer-corpus run
  `rl_agent/policy/experiments/controller_ladder/20260813_063514` is retained as
  a **noncausal matched-support study**, not a deployable controller evaluation.
- **Month 3:** the native-10-Hz advisor-rich corpus is accepted (23 clean runs
  after one impact exclusion) for perception/workload studies and freshness is
  re-scored. The **static measured-profile selection** NO-GO remains credible;
  the full dynamic-controller NO-GO is reopened by the causal audit. Task B's
  observed-pedestrian/cyclist hard-guardrail logic is implemented; its replay
  cost/lift numbers are noncausal. Queue/jitter/loss packaging and Sionna realism remain open.
- **Month 4 groundwork started early:** moving-ego/two-source map display,
  record/replay, synthetic FoV occlusion reasoning, and the recipient-specific
  Phase-2 local contract. The latter passes synthetic plumbing acceptance;
  real CARLA map-GT, freshness, false-hazard, and warning-gain evaluation remain
  open.
- **Policy/reward:** reward v5 and the table-driven surrogate are implemented.
  Task A found no practical seg-inclusive class/range rank reversal; Task C
  found that lambda-RDO is not exactly equivalent to full enumeration on the
  full 36-profile scalar design space. Its retained-catalog runtime equivalence
  is noncausal replay evidence. No RL training is authorized before a causal,
  pre-registered residual gap exists.
- **Binding path now:** freeze the causal Phase-2 control/schema contract and the
  paired helper-recipient corpus specification; review a two-trajectory pilot;
  only then collect the designed and naturalistic suites and evaluate C2 locally
  and over two-UE OAI RFsim. OTA/venue is a parallel risk, not an RFsim blocker.

## 2026-08-14 reconciliation note (causal audit, paper reframe, and Phase-2 gate)

- Accepted corpus: `policy_corpus_advisor_rich_v5/20260813_045142_full`, 23/24
  structurally valid runs, on-contract radar density, both classes populated,
  clean traffic; `pcarv5_mixed_va01` excluded for the ego-walker impact.
- The single-UE ladder and expanded-action gate use same-frame post-tail object
  observations plus GT-assisted matching/track identities. They are retained
  only as noncausal matched-support studies; they do not close the deployable
  dynamic-controller question. The static measured-table result remains valid.
- Measured N=2/modeled-large-N contention found no application-layer admission
  gap under its registered contract; that real-network result is unaffected by
  the replay audit.
- **Task A complete:** 36 profiles x exact 1,683 common frames, with per-frame
  segmentation. Verdict `NO_PRACTICAL_REVERSAL_ON_AVAILABLE_CONTEXTS`; the
  strongest class/range cells miss the registered +0.010 utility gate.
- **Task B complete:** observed vulnerable objects remove SKIP; low-confidence
  vulnerable objects clamp ROI to zero. It improves matched-safe rate by 1.63 pp
  but costs 0.0477 finite matched reward and +1.10 Mbps. Detector misses and
  absent cyclist examples remain outside empirical protection. The rule is
  implementation-valid; the quoted empirical deltas are noncausal replay results.
- **Task C complete:** the 36-profile scalar problem has four lambda-supported
  profiles and only 80.56% budget-breakpoint agreement with exact enumeration;
  the retained seven-profile runtime ladder has 100% lambda-RDO agreement only
  within the noncausal replay. The freshness baseline is AoI-index-inspired, not Whittle.
- The paper/system north star is the end-to-end multi-modal cooperative-
  perception system over OAI RFsim, with transport-conditioned cooperation gain,
  a qualified safety contract, and deployable design rules. Measurement findings
  support those contributions; they are not themselves the paper spine.
- The accepted v5 corpus remains useful but lacks a paired recipient, synchronized
  hazard truth, raw aligned sensing, unfiltered detections, and causal pre-action
  signals. It cannot estimate C2 warning lead and will not be relabelled as the
  Phase-2 dataset.
- Next data unit: `phase2_paired_causal_v1`, with separate pre-registered designed
  opportunity and naturalistic suites. A two-trajectory pilot (one positive
  occlusion/hazard and one matched benign negative) must prove causality and C2
  computability before any full collection. No CARLA/OAI run is yet authorized.

## 2026-08-11 reconciliation note (historical; superseded where the causal audit differs)
- Built the table-driven **policy surrogate environment** (`rl_agent/policy/`) from the channel sweep + knob
  matrix + staleness model; dual oracles (clairvoyant + shielded) + rule/greedy/LinUCB/MPC ladder.
- **Reward** hardened v2->v5 (two-layer: hard C1 mask + live tail-risk shield; per-object AoI localization;
  post-action map utility; normalized costs). **v5 (advisor 2026-08-11):** `U_task = 0.35 seg / 0.40 ped /
  0.25 vehicle`; explicit `C_ROI` removed (learned implicitly); localization remains safety-side; `j_G` is the
  freshness-driving object and `G` is its budget-binding error.
- **Shield safety calibration:** ucb_k / channel-pessimism / estimator-lag all found inert in the deterministic
  surrogate (calibrated uncertainty is a live-validation item); shield sound at <=25 m, unsound at 40 m;
  achievability frontier ~54% feasible at eps=2 m even with perfect info.
- **Corpus saga (honest):** a metric artifact (send-needed measured along a competent controller) and a
  collection-config regression (radar 5k vs validated 200k pps) were caught and fixed; MODEL confirmed intact
  (checkpoint SHA matches, validates 0.855 ped / 0.893 veh); pedestrian detection genuinely ~17%
  (perception-limited, range-dependent). Advisor's Town10HD_Opt CARLA scenario scripts received to rebuild the
  corpus properly (close crossing pedestrians + routed egos) — `rl_agent/advisor_helper_scripts/codes/`.
- **Advisor plan for the rest of this week:** (1) nail the cost<->action relationship + a solid reward
  formulation (the block diagram), (2) evaluate the baselines; **next week: begin RL agent training.**
- **Current execution gate (2026-08-11):** reward v5 is formalized. Richer-corpus orchestration is held before
  smoke because the received route-authoring UI bundle is missing `traffic_light_pole_camera_ui_client_v1.py`,
  `ego_route_config.py`, and `physical_ai_scenario_config_v2.yaml`. Standalone traffic/blocker scripts pass
  static entry-point checks; no substitute route or premature RL run is authorized.

## Project North Star

Build and evaluate an instrumented, network- and safety-aware multi-modal
cooperative-perception system whose helper observations reach a recipient's
spatial map over the OAI 5G stack (RFsim), with quantified cooperation gain,
freshness/safety limits, and deployable control rules. Learned control is used
only if a pre-registered gap remains after exact and simple baselines.

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
| RGB+radar fusion SEG/localization | RGB + radar tensor | Segmentation mask plus localization-style output | Evaluate fusion segmentation and spatial localization quality. Do not treat this as true OD boxes/classes/AP. |
| Single-ego OD/SEG controller harness | RGB camera | Either OD boxes or SEG mask, selected by controller | Exercise task scheduling with two separate split-inference task pipes on the same ego/camera stream. |
| Spatial-map fusion | Outputs from one or more clients | Fused object map | Support occlusion-aware physical-AI experiments later. |

Important note: the RGB+radar fusion model should be described as a fusion
segmentation/localization model, not a true OD model. True OD currently means
the camera-only Faster R-CNN split route. Keep the old fusion localization
analysis as localization/transferability evidence, not OD AP evidence.

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
- [ ] Explicit downlink/result-return latency split:
  - [ ] Edge tail-result ready timestamp.
  - [ ] Result serialization/send timestamp.
  - [ ] Ego/recipient result-receive timestamp.
  - [ ] Display-ready timestamp for segmentation/localization overlays.
  - [ ] Downlink result payload bytes by result type.
- [x] Timeout/missed-result count.
- [x] Approximate FPS.
- [x] Packet-loss or missing-frame indicators where available.
- [ ] Fresh-delivery indicators: generated frames, queued frames, sent frames,
  edge-received frames, tail-completed frames, downlink-received frames,
  displayed frames, stale displayed frames, queue wait, drop reason, and
  fresh-delivered FPS.
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

## Month 2: Parked-Ego RGB+Radar Models, Static Sweeps, and Controller Groundwork

Proposal row being covered:

> Implement constrained RL controller over AE/ROI/quantization/scheduling/
> redundancy actions.
>
> Exit criterion: policy can train or evaluate against static policies using
> the same logged metrics.

Working Month 2 interpretation, updated after the 2026-06-11 supervisor
discussion:

> Get the perception models right first on local loopback. Select a strong
> parked-ego intersection viewpoint, collect CARLA RGB+radar training data,
> train parked-ego RGB+radar SEG/localization and true OD models where possible,
> then use the trained models to produce static payload/task/latency curves and
> offline controller traces. OAI/5QI remains important but is deferred until the
> model behavior is strong and interpretable.

Month 2 goal is not to claim a final RL policy. The goal is to produce the
parked-ego perception base models, data/metrics pipeline, and static baselines
that any learned policy must beat.

Status snapshot for the last week of Month 2, updated 2026-07-09:

- **Overall Month 2 progress:** about **90-95%** complete. The perception,
  radar diagnostic, static knob characterization, and RL-agent measurement
  infrastructure are in place. The remaining Month 2 closure item is the
  offline controller replay/harness over the completed action-cost matrix.
- **RL/controller groundwork:** about **95%** complete for the Month 2 offline
  objective. `rl_agent/` now contains the sweep runner, static
  quantization/entropy/ROI/AE action profiles, deterministic payload and
  latency aggregation, offline accuracy-vs-compression evaluation, M-prime
  drop-aware training pipeline, Gate-A acceptance check, loopback transport
  sweep, and `COMPLETE_KNOB_MATRIX.md` with 19 candidate actions.
- **Priority for the remainder of Month 2:** run the offline controller
  harness over the completed matrix: compare always-safe, always-low-byte,
  best-fixed, simple heuristic, and LinUCB/contextual-bandit policies; report
  guardrail accept/clamp/reject decisions; then freeze the Month 2 summary
  before moving deeper into OAI/Sionna/QoS or online learned-policy training.

Month 2 preflight / terminology correction completed:

- [x] Corrected model taxonomy: current RGB+radar fusion checkpoint is
  segmentation/localization, not true OD. True OD currently means the RGB-only
  Faster R-CNN split route.
- [x] Built and smoke-tested a single-ego OD/SEG controller harness with two
  separate UDP task pipes, timer-based OD/SEG gating, and original demo traffic
  density defaults.
- [x] Built a clean RGB-only transferability harness for moving/autopilot and
  parked ego viewpoints without modifying the controller harness.
- [x] Ran clean RGB-only transferability set locally: moving SEG moving/parked,
  pole-trained SEG TL14/parked-near-TL14, and moving OD moving/parked.
- [x] Generated presentation-ready transferability plots:
  `metrics_logs/rgb_ego_transfer/analysis_clean_20260610/seg_vehicle_iou_clean.png`
  and
  `metrics_logs/rgb_ego_transfer/analysis_clean_20260610/od_recall_precision_clean.png`.
  Updated fusion-geometry-matched SEG plot:
  `metrics_logs/rgb_ego_transfer/analysis_fusionmatched_20260611/seg_vehicle_iou_fusionmatched.png`.
  Headline: moving SEG is weak in both settings because it is mostly pretrained
  and not CARLA-trained (`0.20 -> 0.19` vehicle IoU); pole-trained SEG is much
  stronger because it is CARLA-trained and drops from TL14 pole to parked ego
  (`0.89 -> 0.69` vehicle IoU);
  moving OD recall drops on parked ego (`0.35 -> 0.18`) while precision rises
  (`0.67 -> 0.93`), indicating a conservative parked-view detector.
- [x] Supervisor agreed that the next primary action is to train parked-ego
  RGB+radar models for SEG/localization and true OD, using CARLA-collected
  data from a dense intersection viewpoint.

### 1. Parked-Ego RGB+Radar Training Track

This is now the primary Month 2 track. Network/QoS analysis resumes after the
parked-ego models are good enough to make task-quality comparisons meaningful.

- [x] Inventory available local training scripts:
  - [x] RGB-only CARLA LR-ASPP training workflow:
    `pole_lraspp_training/pole_lraspp_training/{collect_dataset.py,train_lraspp.py,evaluate_lraspp.py,run_pipeline.py}`.
  - [x] RGB+radar fusion SEG/localization workflow:
    `pole_lraspp_multimodal_fusion/pole_lraspp_multimodal_fusion/{collect_dataset.py,train_fusion.py,evaluate_fusion.py,run_pipeline.py}`.
  - [x] Launcher/status helpers exist:
    `pole_lraspp_multimodal_fusion/launch_unattended_fusion_training.sh`,
    `status_unattended_fusion_training.sh`, and
    `stop_unattended_fusion_training.sh`.
  - [ ] Normalize launcher paths before running from the `abiodun/` copy; the
    copied shell helpers still assume the workflow lives at the `neu_collab/`
    root, while the complete local copy is under `abiodun/`.
  - [ ] Confirm with supervisor whether there is a separate true RGB+radar OD
    training workflow. Local scan found fusion SEG/localization training, but
    no separate Faster-R-CNN-style RGB+radar OD trainer yet.
- [x] Select the parked-ego training intersection:
  - [x] Add a parked-ego training-view scout that ranks real CARLA spawn
    points by intersection/crosswalk coverage and emits repeatable collection
    commands.
  - [x] Scout Town10HD intersections with high vehicle and pedestrian traffic.
    Current top candidate from `20260611_172311_parked_ego_view_scout`: TL16,
    spawn 80, right offset `-3 m`, yaw offset `-3.02 deg`, quality `235.55`,
    road/crosswalk proxy coverage `40/20`, left/center/right road coverage
    `17/11/12`.
  - [x] Pick a parked ego view family with broad view of moving vehicles,
    pedestrians, crosswalks, and occlusions: TL16 with spawn80/right `-3`,
    spawn80/right `0`, and spawn85/right `0`.
  - [x] Record spawn index or anchor coordinates, camera mount, FoV, traffic
    density, pedestrian density, weather, and rationale. Pilot recipe uses
    `1280x720`, FoV `120`, model input `768x432`, radar HFoV `120`, 35 NPC
    vehicles, 45 pedestrians, and `sample-stride 5`.
  - [x] Visually confirm the parked camera sees enough vehicle/person foreground
    coverage before collecting a large dataset. Contact-sheet review: spawn85
    is cleaner/broader; spawn80 variants are stronger occlusion/near-vehicle
    views.
  - [x] Rerun/inspect the parked-ego scout with supervisor-preferred
    right-side-road placement so the ego is out of the active travel lane and
    the camera sees side-passing, oncoming, crossing, and pedestrian profiles
    before full collection.
    Selected candidate after visual inspection: TL16 spawn `80`, forward
    offset `4.0 m`, right offset `7.0 m`, yaw offset `-28.414 deg`.
    A second plausible near-intersection parked view was selected at TL16
    spawn `80`, forward offset `16.0 m`, right offset `8.0 m`, yaw offset
    `-28.414 deg`, enabling View A / View B / combined A+B training.
- [ ] Collect parked-ego RGB+radar training data:
  - [x] Smoke set: 30-100 samples, validator PASS. TL16/spawn80 smoke dataset
    `parked_ego_training_tl16_spawn80_60samp` has 60 samples, 3,493 actor rows,
    1,761 vehicle labels, 1,732 person labels, all mask classes present, and no
    validator errors/warnings.
  - [x] Pilot set: enough samples to estimate vehicle/person pixel coverage and
    actor-label density. Three 300-sample pilots PASS validation and target
    dry-run: total 900 samples, 12,294 vehicle actor rows, and 15,367 person
    actor rows.
  - [x] Full overnight set: about 12k-18k saved samples for the first serious
    parked-ego SEG/localization training run, with train/val/test coverage and
    traffic-density variation where practical. Preferred first plan: 3 density
    profiles x 4,000 saved samples with `--sample-stride 2`. Completed for
    View A and View B; combined A+B dataset has 24,000 saved samples.
  - [x] Merge low/medium/crowded folders into one training dataset using
    `scripts/merge_fusion_training_datasets.py`.
  - [x] Save RGB, radar tensor/points, semantic masks, actor boxes/classes,
    calibration, ego/camera/radar poses, split labels, and scenario metadata.
- [ ] Validate dataset and target construction:
  - [x] `scripts/validate_fusion_training_dataset.py` PASS for
    `parked_ego_training_tl16_spawn80_60samp`.
  - [x] `scripts/dry_run_fusion_training_targets.py` PASS for
    `parked_ego_training_tl16_spawn80_60samp`: 60/60 samples produce positive
    vehicle targets, `(7,432,768)` features, `(432,768)` segmentation targets,
    and `(64,9)` GT tensors. Minor dense-scene center-overlap warnings are
    recorded but not blocking.
  - [x] Extend localization targets from vehicle-only to class-aware
    `vehicle/person` heatmaps. New object head is 12 channels: 2 center
    heatmap channels plus 10 shared regression channels.
  - [x] Confirm person samples are now localization positives. Class-aware
    pilot dry-runs PASS on all three TL16 pilots with valid person target
    counts: 4,562; 4,216; and 4,164 respectively.
  - [x] Confirm actor-derived OD labels have enough visible vehicles and
    pedestrians for true OD training. Pilot data has enough actor-derived boxes
    to justify implementing the RGB+radar OD training path.
- [ ] Train parked-ego RGB+radar SEG/localization:
  - [x] Start with the existing `pole_lraspp_multimodal_fusion` trainer or adapt
    it to consume the parked-ego dataset directory. Trainer now accepts parked
    `.npy` radar tensors and class-aware `vehicle/person` object targets.
  - [x] Run a tiny smoke training job before any overnight run.
    `experiments/parked_ego_classaware_train_smoke_20260611` completed one
    CPU epoch on the 60-sample TL16 parked-ego dataset and wrote `best.pt` plus
    `last.pt`. Checkpoint metadata confirms `object_channels=12` and
    `object_class_names=['vehicle', 'person']`.
  - [x] Launch full training in `screen`/`nohup` when leaving the office.
    Completed View A, View B, and combined View A+B runs.
  - [x] Evaluate held-out mIoU, vehicle IoU, person IoU, localization recall,
    xy error, yaw error, payload, and latency. Held-out viewpoint evaluation
    completed for A-on-A, A-on-B, B-on-A, B-on-B, A+B-on-A, A+B-on-B, and
    A+B-on-combined. Presentation plots are in
    `analysis_outputs/parked_ego_fusion_viewpoint_eval/`.
- [ ] Train or obtain true parked-ego RGB+radar OD:
  - [ ] Confirm target model family: Faster-R-CNN-style boxes/classes, CenterNet
    style, or extension of the current object-head targets.
  - [ ] Confirm class labels: vehicle/person at minimum.
  - [ ] Train/evaluate recall, precision, AP proxy, class recall, and payload.
- [ ] Transfer tests after parked models are strong:
  - [x] Parked-trained SEG/localization on parked ego test set.
  - [x] Add dedicated moving-ego RGB+radar fusion collector:
    `carla_collect_moving_ego_fusion_training_data.py`. The parked collector
    remains parked-only; moving collection writes `route_progress.csv` and
    `route_summary.json` for distance/loop diagnostics.
  - [x] Run and validate moving-ego collector smoke/probes. Final stable
    collection mode uses CARLA autopilot with a fixed spawn-index route,
    traffic-light waits excluded from stuck detection, lane changes disabled,
    and loop-count stopping instead of mid-route sample-count stopping.
  - [x] Parked-trained SEG/localization on moving ego. This is now treated as a
    negative-control/domain-gap result, not the primary target: parked A+B on
    moving test reached only about `mIoU=0.262`, `vehicle_iou=0.054`.
  - [x] Train/evaluate moving-ego RGB+radar SEG/localization candidate models.
    The 8-loop moving model is currently stronger on moving test
    (`mIoU=0.825`, `vehicle_iou=0.874`, `person_iou=0.630`) than the 12-loop
    repeated-route run (`mIoU=0.813`, `vehicle_iou=0.846`,
    `person_iou=0.624`). Extra repeated loops alone did not improve model
    quality; next improvement should target route/view diversity,
    sensor-processing changes, or training/threshold tuning.
  - [ ] Parked-trained OD on parked ego test set.
  - [ ] Parked-trained OD on moving ego.
  - [x] Decide whether a separate moving-ego RGB+radar model is needed. Yes:
    moving-domain model training is the main path; parked A+B does not transfer
    to moving data.
- [ ] Study LiDAR/person-localization diagnostics and translate useful ideas to
  radar:
  - [x] Copy the supervisor LiDAR diagnostic into `abiodun/` before modifying;
    do not edit the shared `neu_collab/` original.
  - [x] Document what the semantic-LiDAR script is doing beyond raw LiDAR:
    semantic tags/object ids, high point density, actor-box association,
    voxel accumulation, RGB/semantic colorization, and optional radar dynamic
    filtering.
  - [x] Separate deployable ideas from CARLA-only oracle ideas. Semantic tags
    and object ids are upper-bound/debug signals; point-density, temporal
    accumulation, clustering, rasterization, and radar parameter sweeps can
    inform the real radar pipeline.
  - [x] Run a controlled pedestrian-heavy diagnostic comparing radar support,
    LiDAR support, and semantic-LiDAR upper-bound support for vehicle/person
    localization. The useful result became radar-focused: static/moving
    pedestrian PPS-vs-distance sweeps, model-utility-by-radar-points analysis,
    and controlled CEP diagnostics.
  - [x] Test radar-processing upgrades inspired by the LiDAR path: higher
    radar points-per-second, wider FoV where justified, multi-frame
    accumulation, class-aware/actor-aware supervision analysis, and adjusted
    raster splat radius/channels for sparse pedestrian returns. Completed
    diagnostics include PPS sweeps up to 300k, raster-radius tradeoff,
    temporal-accumulation tradeoff, and CEP50/CEP90 control-knob plots for
    moving pedestrians.
- [x] Record controller-relevant static model/action profiles for the current
  fusion route:
  - [x] Baseline/high-quality model. Current static analysis uses the M-prime
    ROI-robust RGB+radar fusion model as the controller-ready baseline.
    Gate-A comparison against the 200k reference preserves segmentation and
    localization (`mIoU=0.841`, `vehicle_iou=0.933`, localization about
    `1.21 m`), with a small object-recall cost handled by guardrails.
  - [x] Lower-payload compression profile(s). `COMPLETE_KNOB_MATRIX.md`
    records 19 profiles over quantization, entropy coding, ROI drop, and AE
    bottlenecks. Per-channel 4-bit + zstd is the best reliable loopback
    all-rounder at about `365 KB/frame`, near-baseline task quality, and 100%
    result delivery in the loopback sweep.
  - [x] Robust/action-cost evidence. `LOOPBACK_LATENCY.md` captures the
    payload-to-delivery cliff: roughly <=400 KB/frame delivers reliably, while
    ~700 KB and larger profiles return results only about 10-30% of the time
    on the CARLA loopback transport.
  - [ ] Frame-rate/send-rate profile(s). Defer unless needed for the first
    controller replay; payload/action selection is now the stronger Month 2
    static-control evidence.
  - [ ] Scene complexity metadata: visible vehicle/person pixels, actor counts,
    occlusion/density proxies, ego/traffic speed.

Completion criteria:

- [x] One canonical parked-ego intersection/viewpoint is selected and documented.
- [x] Parked-ego RGB+radar dataset passes schema and target dry-run validation.
- [x] At least one parked-ego RGB+radar SEG/localization training run completes
  and produces held-out metrics.
- [ ] True RGB+radar OD training path is either located from the supervisor or
  specified clearly enough to implement.
- [ ] A short report/slide explains model performance, visible-object coverage,
  and whether parked-trained models transfer to moving ego. Parked-viewpoint
  report plots are complete; moving-ego transfer remains pending.

### 2. Freeze the Month 2 Experiment Matrix

- [ ] Select the Month 2 scenario subset:
  - [ ] Low-density clear scene.
  - [ ] Crowded scene.
  - [ ] Curbside hidden-pedestrian scenario.
  - [ ] Optional parked-ego fusion transfer scene if supervisor wants parked ego prioritized.
- [ ] Select the Month 2 route subset:
  - [x] Camera-only OD.
  - [x] Camera-only SEG.
  - [x] RGB+radar fusion SEG/localization.
  - [x] Single-ego OD/SEG controller harness.
  - [ ] Future RGB+radar OD route, only if/when a true fusion OD model exists.
- [ ] Define canonical run durations:
  - [ ] Short smoke: 30-60 s.
  - [x] Measurement run: 180 s.
  - [ ] Long stability run: 300-600 s, only after smoke passes.
- [ ] Define canonical run-group naming:
  - [x] RGB-only transferability: `month2_clean_<model>_<task>_<view>_<resolution>`.
  - [ ] Static sweeps: `month2_static_<route>_<profile>_<transport>`.
  - [ ] 5QI sweeps: `month2_5qi_<value>_<route>_<transport>`.
  - [ ] Controller replay: `month2_controller_replay_<date>`.
- [x] Update or create the Month 2 command sheet/runbook section once the first
  smoke commands are validated. See `SCENESENSE_MONTH2_COMMANDS.md`.

Completion criteria:

- [ ] A table exists listing scenario x route x transport x profile combinations.
- [ ] Every planned run has a reproducible command, expected output folder, and
  analyzer command.
- [ ] Remote sync instructions are recorded for any edited scripts/config files.

### 3. Static UE-Side Payload/Task Sweeps

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
- [x] Run RGB+radar fusion static sweeps:
  - [x] Quantization: `per_tensor_uint8`, `per_channel_uint8`,
    `per_channel_uint6`, and `per_channel_uint4` where supported.
  - [x] Entropy coder: `zlib`, `zstd`, and `none` for diagnosis.
  - [ ] Object score threshold / top-k sweep for fusion_as_od.
  - [x] Offline task-quality sweep for fusion SEG/localization under codec
    round-trip. `rl_agent/analysis/accuracy_vs_compression.md` shows
    per-channel 8/6/4-bit profiles preserve the 200k model almost exactly:
    baseline `mIoU=0.837`, `vehicle_iou=0.934`, `person_iou=0.579`, object
    recall `0.775`; per-channel 4-bit remains close at `mIoU=0.836`,
    object recall `0.770`.
  - [x] M-prime action matrix: `rl_agent/COMPLETE_KNOB_MATRIX.md` combines
    offline task utility, payload bytes, loopback front latency, RTT, and
    delivery for 19 actions. AE profiles are included but flagged because they
    preserve segmentation while collapsing object detection/localization.
- [x] For each accepted static profile, collect:
  - [x] Application metrics CSV.
  - [x] Run manifest/resolved config.
  - [x] Task-quality summary: fusion object recall/localization and
    segmentation utility for accepted fusion profiles.
  - [x] SEG/localization summary: mIoU, vehicle/person IoU, object recall, and
    vehicle/person localization MAE for the evaluated fusion profiles.
  - [x] Payload summary: bytes/frame, chunks/frame, compression ratio.
  - [x] Front-latency and result-return summary for loopback static sweep.
    Loopback RTT and result-delivery are now recorded; OAI/Sionna channel loss
    remains Month 3 follow-up.

Completion criteria:

- [x] At least one static Pareto plot exists per priority route:
  payload vs latency vs task utility.
- [x] A "best fixed static policy" is identified for the current priority
  route. The current per-model Pareto pick is integrated AE-128 + per-channel
  u4 + ROI off: about `127 KB` in the offline matrix, `mIoU=0.819`, pedestrian
  recall `0.887`, and localization about `0.88 m`. Its OAI deployment is the
  strongest cross-layer point measured so far: about `142 KB`, `77 ms` mean
  RTT, and `99%` result delivery.
- [x] A "lowest-byte unsafe policy" is identified to motivate guardrails.
  Integrated AE models are no longer categorically unsafe; the earlier
  standalone-AE object-head collapse was resolved by joint/integrated
  training. Aggressive ROI+AE combinations remain unsafe for segmentation,
  while large no-AE payloads are unsafe for OAI availability/latency.
- [x] Runs that saturate or time out are labeled as saturation evidence, not
  valid task-quality samples.

### 4. OAI 5QI/QoS Experiments

Status update 2026-07-16: a limited single-UE, no-impairment RFsim study is
complete. TDD `7:2` vs `4:5` and 5QI `9` vs `1` changed RTT/delivery only
slightly; payload reduction was the effective lever. The full QoS matrix,
background load, multi-UE contention, channel impairment, and clean QFI/DRB
evidence remain deferred. See `oai_config_sweep/OAI_CONFIG_FINDINGS.md`.

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
- [ ] Run reliability/delivery stress before learned policy training:
  - [ ] FPS sweep: 5, 10, 15, 20, 30 FPS.
  - [ ] Buffer policy/size sweep: latest-only/1, 2, 4, 8, 16.
  - [ ] Payload profiles: no-AE u8 ROI0 and AE-128 u4 ROI0.
  - [ ] Network modes: loopback, clean OAI, later Sionna-varying channel.
  - [ ] Report delivered FPS and fresh-delivered FPS separately.
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

### 5. First Offline Controller Harness

This is the first bridge from measurement to control. It should replay logged
traces and score decisions offline before any online RL touches CARLA/OAI.

- [x] Create dedicated RL-agent workspace under `rl_agent/` with clear stage
  documentation and scripts for static sweeps, accuracy aggregation,
  M-prime training, and action-cost matrix construction.
- [x] Build generic static sweep runner:
  `rl_agent/sweep_runner.py` with config fan-out under
  `rl_agent/configs/static_sweep_quant_entropy.json`.
- [x] Build deterministic sweep analyzer:
  `rl_agent/sweep_analyze.py`, `rl_agent/overnight_analyze.sh`, and
  `rl_agent/analysis/static_sweep_summary.md`.
- [x] Build deterministic offline accuracy aggregator:
  `rl_agent/accuracy_aggregate.py` and
  `rl_agent/analysis/accuracy_vs_compression.md`.
- [x] Build M-prime drop-aware robustness pipeline:
  `rl_agent/run_pipeline_m_prime.sh`, `rl_agent/m_prime/stage1_seg_drop.json`,
  `rl_agent/m_prime/stage2_obj_drop.json`, and `rl_agent/gate_a_check.py`.
  Status as of 2026-07-09: M-prime is accepted as the robust controller
  baseline. It preserves clean segmentation/localization against the 200k
  reference (`mIoU=0.841`, `vehicle_iou=0.933`, localization about `1.21 m`);
  object recall has a small residual cost that should be protected by the
  controller guardrail rather than blocking the matrix.
- [x] Build action-cost matrix scaffold:
  `rl_agent/build_knob_matrix.py`. The current authoritative action table is
  `rl_agent/PERMODEL_KNOB_MATRIX.md`: 42 profiles across integrated AE-128/64/32,
  no-AE, quantization 8/6/4, and ROI 0/0.3/0.5. The older
  `COMPLETE_KNOB_MATRIX.md` is a pre-integrated-AE snapshot.
- [x] Implement trace loader that joins:
  - [x] Application metrics by run group / stream / frame or timestamp window.
    `rl_agent/policy/replay.py` discovers paired accepted-corpus GT/prediction
    traces, applies the frozen class thresholds, and resamples them to the
    controller clock with episode-grouped splits.
  - [x] Loopback/OAI latency and reliability metrics for action profiles.
    `latency.py` joins the measured 90-KiB channel anchors to the profile table
    and flags projected payload/rate cells; it does not pretend all cells were
    directly measured.
  - [ ] Network sampler metrics for OAI/Sionna phase.
  - [ ] T-tracer / gNB metrics where available for OAI/Sionna phase.
  - [x] Scenario metadata via run group, family, variant, episode, and split.
  - [x] Task-quality summaries via the action catalog plus per-frame observed/
    truth replay state.
- [x] Implement action-profile catalog:
  - [x] Segmentation-safe/core profiles (90 and 129.2 KiB).
  - [x] Balanced and low-byte ROI-escalation profiles.
  - [x] SKIP plus five target-FPS choices per retained profile.
  - [x] Unsupported/out-of-support and C1-inadmissible actions are masked by the
    shared shield; vulnerable-object clamps are separately logged.
- [x] Implement reward scorer:
  - [x] Reward-v5 task utility retained after delivery/map update.
  - [x] Minus PRB/load cost and mode-switch cost.
  - [x] Latency/delivery cost enters through map AoI and localization error;
    drops retain prior map quality rather than receiving selected-profile credit.
  - [x] Localization remains in the safety contract plus a small normalized
    margin; explicit ROI cost is intentionally absent in v5.
- [x] Implement first non-RL baselines:
  - [x] Fixed action with shield fallback.
  - [x] Explicit network/AoI/speed rule.
  - [x] Exact one-step measured-table enumerator.
  - [x] Short-horizon MPC.
  - [x] Lambda-RDO supported-hull lookup and AoI-index-inspired heuristic.
- [x] Optional learning baseline:
  - [x] Disjoint LinUCB over the discrete action catalog; it does not beat the
    simpler greedy/MPC references.
  - [ ] Do not use SAC unless continuous knobs are introduced and simple
    baselines are already beaten.

Completion criteria:

- [x] Controller replay produces a CSV/JSON summary for every baseline policy.
  Done 2026-08-11: `rl_agent/policy/experiments/controller_ladder/...` (summary.csv +
  per_frame_metrics.csv + figures) for fixed/rule/greedy/LinUCB/MPC.
- [x] Baseline policy comparison plot exists:
  task utility vs bytes vs latency/timeout. (controller_ladder figures.)
- [x] The best simple reference is identified. On the accepted richer corpus,
  greedy finite matched reward is 0.19655 and MPC is 0.19834 (+0.91%) at the
  same 91.13% matched-safe rate. **Causal-audit correction:** this identifies the
  best reference only inside the noncausal matched-support replay; it is not a
  deployable dynamic RL NO-GO.
- [x] No online action execution is enabled until offline replay passes sanity
  checks. (Surrogate-only; no CARLA/OAI execution.)

### 6. Guardrail Threshold Draft

Month 2 guardrails can remain offline, but thresholds must become concrete
enough for controller replay.

- [ ] Draft route-specific task floors:
  - [ ] Camera OD recall/precision or AP proxy floor.
  - [ ] Camera SEG foreground IoU / mIoU floor.
  - [x] Fusion localization recall/error floor drafted through Gate-A and
    static compression metrics: current M-prime gate compares against the
    det_pps200000_v2 reference on mIoU, vehicle/person IoU, object/person
    recall, global/person XY MAE, and dimension MAE.
  - [x] Fusion SEG foreground/vehicle/person IoU floor drafted through the
    200k baseline and Gate-A acceptance checks.
- [x] Draft vulnerable-object rules:
  - [x] No SKIP when an **observed** pedestrian/cyclist is active. No hidden-
    hazard signal exists yet, and detector misses remain outside this rule.
  - [x] No aggressive saliency/ROI drop when observed vulnerable-object confidence is low.
    A separately calibrated uncertainty signal is not yet available.
  - [x] Stale/unmapped objects are handled by the localization shield; if the
    vulnerable rule conflicts with every C1-admitted action, the conflict is
    surfaced rather than hidden.
- [ ] Draft network fallback rules:
  - [x] If timeout/no-result rate rises, prefer smaller payload before dropping
    safety-critical frames.
  - [ ] If UE tunnel drops/errors rise, reduce payload detail or send compact
    hazard messages.

Completion criteria:

- [x] Guardrail thresholds are executable in
  `rl_agent/policy/configs/track_a_pilot.yaml` and enforced by `shield.py`.
- [x] Replay reports guardrail application, removed actions, opportunity counts,
  and unachievable conflicts separately.
- [x] Fallback cost is measured in reward, offered load, selected payload, and
  safety in `experiments/vulnerable_guardrail/20260814_215337`.

### 7. Spatial-Map Sharing Groundwork

The full map-sharing RL agent is Month 5, but Month 2 should make the closed-loop
case study measurable.

- [ ] Define spatial-map utility fields needed by the future map-sharing agent:
  - [x] Object class.
  - [x] Pose / velocity.
  - [x] Confidence / uncertainty.
  - [x] Provenance stream id.
  - [x] Freshness / age.
  - [x] Occlusion or hazard flag.
  - [x] Intended recipient or affected ego vehicle (`recipient_ue_id` is a
    required runtime field and maps are recipient-isolated).
- [ ] Define curbside hidden-hazard utility metrics:
  - [x] Warning lead time before collision / near-miss defined as paired first-
    warning lead on the same evaluation truth trajectory; real estimate pending.
  - [ ] Vulnerable-object recall before collision.
  - [x] Stale-object rejection/rate field defined; real estimate pending.
  - [x] False hazard rate defined via evaluation-only truth association; real estimate pending.
  - [x] Exact application/on-wire bytes per advanced warning defined; real estimate pending.
- [x] Create a local-only vs cooperative comparison plan:
  - [x] Ego-only evidence.
  - [x] Helper/observer evidence.
  - [x] Spatial-map shared warning.
  - [x] Send-everything map update baseline.
  - [x] Compact recipient-hazard-only update baseline.

Completion criteria:

- [ ] A replayable curbside run folder can produce warning-lead-time and
  freshness metrics.
- [ ] Spatial-map entries can be joined back to CARLA ground truth and evidence
  traces.
- [ ] There is a clear Month 3/4 path from spatial-map measurement to online map
  sharing.

### Month 2 Definition of Done

- [x] Parked-ego RGB+radar training viewpoint is selected and documented with
  visual evidence.
- [x] Parked-ego RGB+radar dataset collection and validation are repeatable.
- [x] Parked-ego RGB+radar SEG/localization model is trained and evaluated on a
  held-out test split.
- [x] True RGB+radar OD training path is located or its missing architecture/
  label requirements are documented.
- [x] At least one static payload/latency/task profile is collected using the
  new parked-ego model or the best available substitute.
- [x] Offline controller replay can score simple static policies once at least
  two valid action/model profiles exist. **DONE 2026-08-11:** the controller
  ladder (rule/greedy/LinUCB/MPC vs static baselines) ran on the surrogate
  environment; this closes the Month-2 implementation/plumbing exit item only.
  **2026-08-14 caveat:** its same-frame/GT-assisted observation is noncausal, so
  the numbers are not Phase-2 controller evidence.
- [x] A Month 2 slide/report summarizes:
  - [x] Chosen parked-ego scene and dataset coverage.
  - [x] SEG/localization and OD-model-status performance.
  - [x] Payload/latency/task tradeoffs for available profiles.
  - [x] Remaining gap before OAI/QoS and online RL. Current next-step summary:
    finish offline controller replay, then add controlled impairment and
    multi-UE contention. The single-UE OAI compression/config baseline is now
    measured; Sionna/channel-stress integration remains open.

## Month 3: Guardrail Stress Tests

Requirements groundwork completed before the policy stress campaign:

- [x] Validate live model accuracy and fix the actor-origin vs bounding-box-
  center ground-truth mismatch.
- [x] Measure localization error vs object speed and analytical latency.
- [x] Measure held-map staleness vs FPS and combined `Y_up + 1/FPS` age.
- [x] Split the latency result by straight/curve/intersection road state.
- [x] Complete natural-scene post-hoc radar/camera FoV-position split for
  range-aware edge risk. Controlled lateral diagnostic remains optional and
  must pass centered-baseline parity before any sweep.
- [ ] Log downlink/result-return latency and payload using existing loopback/OAI
  split-inference path.
  - [x] Ideal-loopback no-AE 200k FPS sweep complete:
    `downlink_latency_fps/IDEAL_LOOPBACK_RESULTS.md`.
  - [x] Bounded/default-buffer loopback calibration complete and classified as
    a reliability/buffer-failure condition:
    `downlink_latency_fps/BOUNDED_LOOPBACK_CALIBRATION.md`.
  - [ ] Default OAI sweep pending; first health check found OAI core healthy but
    `oaitun_ue1` absent, so UE/RAN/back-half bring-up is needed.
- [ ] Extend freshness constraint from `Y_up + 1/FPS` to
  `Y_up + 1/FPS + Y_down + Y_map_share` after downlink logging.
- [ ] Measure FPS × buffer size × payload reliability, including delivered FPS,
  fresh-delivered FPS, queue wait, stale-result age, drops, and timeout rate.
- [ ] Integrate Sionna/ray-traced channel traces after the logging schema is
  stable.
- [ ] Add object speed, road state, and map age to the executable controller state.
  Speed and map AoI are implemented; road state remains missing from the
  observation, so this compound item is intentionally still open.

- [ ] Add controlled stress profiles: jitter, delay, queueing, packet loss, or bandwidth limits.
- [x] Test whether byte-minimizing choices damage mIoU/class recall. The
  36-profile Task A analysis includes per-frame segmentation and both class
  recalls; true OD AP remains outside the fusion-head claim.
- [x] Add deterministic guardrail layer for C1, localization, OOD, graceful
  degradation, and observed vulnerable objects.
- [x] Compare the exact proposed controller with and without observed-
  vulnerable-object guardrails on paired held-out replay.
- [x] Produce guardrail application/fallback cost plots and structured metrics:
  `rl_agent/policy/experiments/vulnerable_guardrail/20260814_215337`.

## Month 4: Physical-AI Spatial Map Ingestion

Groundwork completed early (does not satisfy the formal Month-4 exit criterion):

- [x] Live moving-ego follow-map and two-source color-by-source view.
- [x] Offline record/replay and synthetic two-view scenes.
- [x] Synthetic FoV-membership occlusion prototype with known toy ground truth.
- [x] Recipient-specific contribution schema, source-snapshot adapter, causal
  hazard-only selector, map engine, warning baseline, separate truth join, and
  production-header-compatible chunk/reassembly contract. Offline acceptance:
  `phase2_map_sharing/experiments/20260814_222111` (synthetic only). Existing-
  recording adapter smoke also passes at
  `phase2_map_sharing/experiments/snapshot_adapter/20260814_222354` (37 paired-
  active snapshots; 26 fresh accepted, 11 stale rejected; no hazard truth).
- [x] Implement the separate Phase-2 v2 offline contract foundation: strict
  source-local schema, object+recipient covariance propagation, aligned-clock
  rejection, per-stage causal allowlist/audit logging, truth/shadow rejection,
  isolated counterfactual arm state, and pre-write raw-retention quotas.
- [x] Wire the v2 safeguards into a derived paired helper-recipient collector,
  exact-frame external-ticker orchestration, isolated three-arm replay, and the
  nine-gate verifier. Phase-2 tests pass 36/36 and collector regressions pass
  61/61; this is implementation evidence, not C2 evidence.
- [x] Review and freeze the road-legal helper/recipient UI geometry: helper
  lane `+1`, recipient lane `-2`, accepted positive and matched-benign arms.
- [x] Confirm host GPU headroom for the correctness-only shared-`cuda:0`
  assignment and cut separate pilot-only contract/integration configs. CARLA
  alone used 6,362 MiB and 16,748 MiB remained free. Only the two-trajectory
  CARLA pilot is authorized; shared-GPU inference timing is non-citable.
- [x] Launch the detached two-trajectory pilot, stop at its completion sentinel,
  then separately run replay and the nine hard gates. Accepted capture:
  `data_collection/experiments/phase2_paired_causal_v1/20260817_181354_pilot`.
  The create-only `evaluation_v4` / `verification_v4` repair proves a
  registered-target, helper+recipient capture-to-warning chain; all nine gates
  PASS. Warning parameters and shared-GPU timing remain non-citable.
- [x] Freeze the post-pilot warning evaluation design in
  `phase2_map_sharing/WARNING_EVALUATION_DESIGN_FREEZE.md`: correct
  target/non-target/unmatched/false-warning labels, trajectory-clustered units,
  bounded calibration grid, 5 pp miss and 2 pp nuisance C2 non-inferiority
  margins, and a 0.5 s minimum meaningful lead. Phase-2 tests: 48/48 PASS;
  data-collection regressions: 65/65 PASS.
- [x] Implement and test the evaluation-only future-trajectory hazard
  adjudicator. Authoritative output `hazard_adjudication_v2` preserves all
  runtime hashes, performs class-constrained one-to-one matching, and uses the
  matched benign recipient trajectory as the positive no-yield
  counterfactual. The intervention-contaminated v1 output is superseded.
  Stopping/clearance is computable but non-attributable until warnings actuate
  a fixed downstream controller. Phase-2 tests pass 57/57 and data-collection
  regressions pass 65/65.
- [x] Generate the deterministic powered Suite A/B design candidate and hashed
  grouped split manifest. Suite A is designed opportunities (120 independent
  groups); Suite B is naturalistic operation (90 groups). Exact 20/20/60
  assignments produce 330 world trajectories, ~16 h staged capture, and a
  54.61 GB tiered-retention estimate under the 80 GB cap. The 0.5 s lead
  sensitivity is 0.883 approximate power at SD 1.25 s and 10% censoring; this
  is not a pilot variance estimate.
- [x] Author, visually accept, and hash-freeze the signalized-corner and
  parked-van-midblock pedestrian geometries. The latter now gravity-settles
  its curbside occluder before freezing physics; its two 33-point routes are
  immutable.
- [x] Resolve renderer quality as an explicit corpus contract. Primary Phase-2
  rows lock CARLA `Epic` (`-quality-level=Epic`); existing Low captures are a
  labelled stress diagnostic only. The dense <=12 m weighted comparison was
  inconclusive due zero near-pedestrian support, so this is an operational
  freeze rather than an all-class dominance or training-lineage claim.
- [ ] Author and visually accept three pending vehicle geometries plus two
  paired naturalistic routes, then run only the 15-trajectory calibration
  audit. The registered estimator must demonstrate >=0.80 simulated power and
  adequate non-inferiority precision before validation. Full collection is
  still unauthorized.
- [ ] Real-data ray/visibility-grid occlusion disambiguation and warning path.

- [x] Add an offline adapter from existing raw split-model spatial-map outputs
  to recipient contributions; live CARLA validation remains pending.
- [x] Store class, pose, velocity, confidence, provenance, freshness, recipient,
  occlusion state, and causal hazard score in the Phase-2 contract.
- [ ] Validate map entries against CARLA ground truth.
- [ ] Measure map freshness, stale-object rate, false hazard rate, and localization error.

**2026-08-14 critical-path contract:** scope the first formal Phase-2 path to
one helper/source and one recipient ego. Define target association, recipient
selection, contribution publication, warning timing, and GT joins locally
before inserting the existing two-UE OAI RFsim transport. This is the binding
C1/C2 path; do not start map-sharing RL before periodic/send-everything and
hazard-only baselines establish a residual gap.

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
- [x] Keep the frozen M-prime model; the on-contract diagnostic cleared the
  pedestrian head and no retraining is warranted for the accepted corpus.
- [ ] Which task metric thresholds become hard guardrails. C1, localization,
  observed-vulnerable no-skip, and low-confidence ROI0 are executable; route-
  specific OD/SEG floors and hidden-hazard semantics remain open.
- [ ] Target venue and whether an OTA leg is required. This is a parallel paper-
  acceptance risk, not a blocker for stabilizing Phase 2 over RFsim.
