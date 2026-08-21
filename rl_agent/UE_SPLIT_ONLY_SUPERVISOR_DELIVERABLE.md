# Simplified UE split-inference data-collection plan

> **Historical pre-meeting handoff.** The supervisor discussion on 2026-08-20
> replaced the reduced 3+1/static-regime proposal with the full 72-action,
> time-varying-network design. The current authority is
> [`UE_SPLIT_ONLY_EXPERIMENT_PLAN_V2.md`](UE_SPLIT_ONLY_EXPERIMENT_PLAN_V2.md).
> This file remains as a record and should not be used to launch the experiment.

**Status:** Draft for supervisor discussion — no experiment run is authorized.

**Date:** 2026-08-20

**Prepared by:** Abiodun

**Editable presentation:**
[SceneSense_UE_Split_Baseline_Supervisor_2026-08-20.pptx](presentation/ue_split_supervisor_20260820/SceneSense_UE_Split_Baseline_Supervisor_2026-08-20.pptx)

## 1. Objective

The immediate question is:

> How should a single UE choose a split-inference transmission profile under
> varying network conditions so that useful detections reach the edge spatial
> map with acceptable quality, delay, and freshness?

This is the measurement foundation for the UE agent. It is not yet an RL
training experiment. We will first determine whether the measurements support
a simple profile-selection rule and whether network conditions actually create
meaningful action choices. A learned policy will be considered only if a
useful sequential decision problem remains after that baseline.

## 2. Scope

### Included in this baseline

- One vehicle UE and one edge spatial-map endpoint.
- Split inference only.
- A small proposed catalog of registered model/compression bundles.
- The same aligned RGB-plus-radar source samples for every comparison.
- Fixed UE and edge compute allocation.
- Four existing OAI network regimes.
- Perception, processing, network delivery, accepted map-update latency, and
  map freshness measurements.

### Deferred until this baseline is understood

- `SKIP` and full `LOCAL` inference actions.
- Radar-conditioned action selection and urgency logic.
- Occlusion reasoning, helper/recipient cooperation, and map-sharing policy.
- DQN, discrete SAC, MPC, reward tuning, or controller training.
- Complex CARLA/NPC choreography and a new CARLA run for every network cell.
- Sionna or a larger SNR sweep.

The fixed experimental flow is:

```text
retained UE sensor sample
    -> one registered split bundle
    -> OAI uplink
    -> the matching edge decoder and tail
    -> accepted edge spatial-map update
```

CARLA ground truth is used only after a result is produced, for evaluation. It
is never an input to profile selection.

## 3. Proposed split action space

We already have offline payload and perception evidence for 72 configurations:
four trained model families, three quantization settings, and six ROI-drop
settings. The table below is a small experimental shortlist drawn from that
pool. It is proposed for discussion; it is not yet a deployed action catalog.

The remaining configurations are not discarded. They stay in the evidence
pool and can be reconsidered if the network measurements expose a useful
payload boundary that this shortlist does not cover. We avoid sending all 72
through OAI because many are near-equivalent or inferior trade-offs and would
add runs without adding a distinct decision. During this baseline, one bundle
is fixed per replicate; per-frame model switching is a later implementation
step after the useful catalog is known.

Here, `q` is the fraction of the lowest-ranked feature cells dropped before
lossless `zstd` level-3 compression. It is not an object-confidence threshold.

| Role | Registered bundle | P95 payload | Estimated UDP/IP load at 10 Hz | Vehicle recall | Pedestrian recall | Secondary mIoU | Purpose |
|---|---|---:|---:|---:|---:|---:|---|
| Degraded rescue | AE32 / uint4 / q0.9 | 19.8 KiB | 1.63 Mbps | 0.921 | 0.841 | 0.424 | Last resort only if no normal action is feasible |
| Compact normal | AE32 / uint4 / q0.5 | 51.9 KiB | 4.25 Mbps | 0.920 | 0.852 | 0.656 | Lowest-payload proposed normal action |
| Balanced normal | AE64 / uint4 / q0.5 | 66.5 KiB | 5.45 Mbps | 0.918 | 0.871 | 0.702 | Moderate payload with stronger pedestrian recall |
| Quality normal | AE128 / uint4 / q0.3 | 101.3 KiB | 8.30 Mbps | 0.924 | 0.886 | 0.684 | Highest recall and best world-location accuracy of this shortlist |

The current map service, `OBJECT_MAP_V1`, prioritizes:

- vehicle or pedestrian class and confidence;
- predicted actor-reference location in world XY;
- capture/sample identity; and
- an explicit distinction between a valid empty result and a missing update.

Dense segmentation is retained and its class IoU/mIoU is reported, but it is a
secondary diagnostic because the current spatial map is object-centric. The
rescue bundle falls below the normal pedestrian-recall floor and therefore
cannot count as normal-quality success. Using it would create explicit service
debt rather than silently weakening the service requirement.

The reported quality values are from the same 2,162-frame offline evidence
set. They are catalog-screening evidence, not a guarantee of identical live
performance. The live measurements must preserve the same sensor,
preprocessing, checkpoint, and decoder contracts and later include a bounded
shadow validation.

## 4. Network conditions

The first baseline reuses four calibrated, static OAI/AWGN regimes. Within a
regime, latency and delivery vary from sample to sample, but these are not yet
time-varying mobility traces.

| Report name | Historical configuration name | Achieved SNR reference | MCS reference | Historical served-capacity reference |
|---|---|---:|---:|---:|
| Clear | `clear` | 50.3 dB | 28 | 36.68 Mbps |
| Mild | `mild` | 19.5 dB | 24 | 27.78 Mbps |
| Mid | `mid15` | 15.6 dB | 19 | 19.71 Mbps |
| Poor | `strong` | 8.2 dB | 9 | 10.39 Mbps |

The historical name `strong` meant strong impairment, not a strong channel;
this report therefore calls it `poor`.

These capacity values are useful screening references, not exact 10-Hz
results for the proposed bundles. The historical producer achieved roughly
5.8–8.0 sends/s, used proxy payloads, and stopped before an authoritative
map-update record. It therefore does not provide exact 10-Hz latency, drop, or
map-AoI values for any proposed action.

All four proposed actions are below the central historical capacity estimate
even in the poor regime. Only the quality action is inside the uncertainty
band around that poor-link estimate. Therefore, we must not assume in advance
that these network conditions require profile switching. Finding that one
normal action works across all four regimes would be a useful result, not a
failed experiment.

## 5. Environment and fixed controls

- Select and hash one representative retained Town10HD single-UE sequence.
- Replay the identical samples in the identical order for every measured cell.
- Give each replayed sample a new monotonic release timestamp at 10.00 Hz; the
  historical CARLA timestamp remains provenance only.
- Hold the sensor contract, hardware, packetization, `zstd` level, OAI radio
  resources, edge path, decoder, and map semantics fixed.
- Run each registered bundle with its matching checkpoint and decoder/tail.
- Use one isolated map namespace or a complete empty-state reset per replicate.
- Give every replicate a unique stream ID and every sample a monotonic sequence
  ID so late or duplicate results cannot replace a newer map contribution.
- Count an update only when the map server records it as accepted and complete.
  A UE send or edge enqueue is not a map update.

The current decoder remains fixed throughout this experiment. A separate
offline audit found that duplicate predictions explain most raw vehicle false
positives and that predicted-only suppression can improve precision. However,
the most aggressive tested setting reduced recall and worsened localization,
so it is not promoted here. That audit used q0 outputs and therefore cannot
rerank the q0.3/q0.5/q0.9 shortlist. Decoder refinement remains a separate
validation task, not an agent action and not a reason to retrain the model now.

## 6. Logical combination sheet and bounded measurement sequence

The full planning sheet contains the four proposed profiles crossed with the
four network regimes: 16 logical rows. Every row is currently a composed
projection, not a directly measured fixed-10-Hz result.

| Action | Clear | Mild | Mid | Poor |
|---|---|---|---|---|
| Degraded rescue | Projected | Projected | Projected | Conditional last resort |
| Compact normal | Projected | Projected | Projected | Conditional after balanced |
| Balanced normal | Projected | Projected | Projected | Conditional after quality |
| Quality normal | Initial direct control | Projected | Conditional diagnostic | Initial direct boundary |

The proposed execution sequence is deliberately adaptive:

1. Measure **Quality × Clear** as the best-channel exact-path latency control.
2. Measure **Quality × Poor** as the only current capacity-boundary candidate.
3. If Quality × Poor is stable, stop the first round. Infer only transport
   feasibility for the lower-load actions; leave their latency, drop, map
   update, and AoI fields explicitly unmeasured.
4. If Quality × Poor fails or is borderline, measure **Balanced × Poor**, then
   **Compact × Poor** only if needed.
5. Measure **Rescue × Poor** only if no normal action is feasible.
6. Use **Quality × Mid** only to bracket a failed or borderline poor result.
7. If Quality × Poor passes comfortably and a network-conditioned decision is
   still scientifically required, propose one adjacent worse or loaded
   condition whose service capacity lies between approximately 5.45 and 8.30
   Mbps. That extension requires a separate decision; it is not pre-authorized.

For each directly measured cell, use three independent replicates with at
least 200 post-warm-up releases per replicate at achieved 10 Hz. Sequential
samples will be analyzed as ordered run data rather than treated as independent
draws.

The machine-readable sheet is
[UE_SPLIT_ONLY_SUPERVISOR_COMBINATIONS_V1.csv](UE_SPLIT_ONLY_SUPERVISOR_COMBINATIONS_V1.csv).
Its `measurement_authorized` field is `false` for every row.

## 7. Measurements and data-sheet fields

### Profile cost and perception

- Exact feature payload and estimated on-wire bytes.
- UE front/encoding and edge decoding/tail processing time.
- Vehicle and pedestrian precision, recall, and F1.
- False positives per frame.
- World-XY location error, reported together with recall.
- Segmentation vehicle/person IoU and mIoU as secondary diagnostics.

### Network delivery

- Achieved SNR and MCS, rather than only the commanded regime name.
- Release, delivery, and accepted-map-update rates.
- Retransmissions, queue/backlog evolution, drops, and timeouts.
- Offered and served throughput.

### Latency

- UE processing, network transport, edge processing, and map-update stages.
- End-to-end release-to-accepted-map-update latency.
- p50, p90, p95, maximum, and deadline-miss rate.
- Slow observations remain in the primary analysis unless a predeclared
  invalid-run event is proven from logs. A p95 value alone does not determine
  pass or fail.

### Map freshness

At a query time `t`, map Age of Information is:

```text
map_AoI(t) = t - release_time(newest accepted map update)
```

This uses the source sample's replay release time, not the send time or ACK
arrival time. We will report the AoI distribution, time above candidate
thresholds, and time-aligned world-location error. A single universal
`AoI_max` will not be invented beforehand: the supervisor can select the
acceptable service tolerance after reviewing the measured age/error trade-off
and the existing staleness sensitivity.

## 8. What these measurements teach us about the agent

The completed data sheet will supply:

1. a feasibility mask: which profiles can be served without unstable queues
   under each measured condition;
2. a cost/outcome model for payload, latency, drops, map freshness, and object
   quality;
3. a deterministic rule or greedy baseline for selecting a feasible profile;
   and
4. evidence about whether a temporal learning policy is justified.

If the same normal profile is preferred and stable in every regime, we should
not manufacture an RL problem. We would either use the simple rule or run one
bounded adjacent condition to locate the true boundary. If action preferences
change with lagged channel and queue state, the measured table becomes the
foundation for a causal sequential environment. `SKIP`, `LOCAL`, dynamic
channel traces, MPC, and RL would then be added in later stages, one justified
extension at a time.

## 9. Deliverables and acceptance

This planning stage produces:

- this supervisor-facing experiment plan;
- the 16-row logical combination/status sheet;
- links to the existing offline profile, network, and staleness evidence; and
- a bounded, conditional measurement sequence.

Before any execution, the owners will separately approve the exact replay
sequence, fixed runtime versions, logging joins, run duration, and output
directory. No CARLA, OAI, model training, or policy training is authorized by
this document.

## 10. Decisions requested in the discussion

1. Confirm the one-UE, split-only baseline scope.
2. Comment on the proposed three normal profiles plus one degraded rescue.
3. Confirm use of the four calibrated OAI regimes, with `strong` renamed
   `poor` in reporting.
4. Confirm that object class and world-location quality are primary and
   segmentation is secondary for `OBJECT_MAP_V1`.
5. Confirm the two-cell initial measurement and conditional step-down sequence.
6. Confirm that `AoI_max` should be selected from the measured freshness/error
   trade-off rather than chosen arbitrarily.

## 11. Supporting evidence

- Full technical and reproducibility contract:
  [UE_SPLIT_ONLY_EXPERIMENT_PLAN.md](UE_SPLIT_ONLY_EXPERIMENT_PLAN.md)
- Current UE execution checklist:
  [UE_AGENT_EXECUTION_CHECKLIST.md](UE_AGENT_EXECUTION_CHECKLIST.md)
- Reuse-only Stage-A evidence:
  [Stage-A report](experiments/ue_split_stage_a_v1/20260820_024055_review/REPORT.md)
- Offline candidate proposal:
  [Candidate report](experiments/ue_split_catalog_proposal_v1/20260820_042414_candidate/REPORT.md)
- Decoder investigation, retained as a separate non-promoted result:
  [Decoder report](experiments/model_precision_decoder_audit_v1/20260819_210004/REPORT.md)
