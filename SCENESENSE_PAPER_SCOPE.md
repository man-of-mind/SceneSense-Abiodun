# SceneSense Agent Paper Scope

Living outline for turning the project into a focused MobiSys-style systems paper.

This document should evolve as experiments produce evidence. The goal is to keep the research story sharp while the engineering work grows.

Last updated: 2026-06-05.

## Working Title

SceneSense Agent: Task-aware Split Inference and Spatial-Map Sharing for Network-aware Cooperative Perception

## One-Sentence Claim

SceneSense Agent learns when and how to transmit split-inference perception evidence under network constraints while preserving object-detection and segmentation utility for cooperative autonomous systems.

## Current Paper Thesis

Future autonomous machines should not stream every pixel or every feature all the time. They should transmit task-relevant perception evidence based on scene difficulty, model confidence, network state, and downstream safety value.

The first paper-quality contribution should focus on a measurable closed loop:

```text
scene/model/network state
  -> split-inference control action
  -> payload/latency/task-utility outcome
  -> guardrail acceptance or fallback
  -> spatial-map update for cooperative perception
```

## Refined Scope After Month 1

The strongest first-paper shape is no longer "build a cooperative perception
model." It is:

> **SceneSense is a network-aware evidence-sharing layer for cooperative
> perception.** It measures task utility and network cost for OD, SEG, and
> RGB+radar fusion over loopback and OAI 5G, then controls what perception
> evidence is transmitted so safety-relevant map updates arrive within useful
> latency/byte budgets.

Month 1 supports the premise:

- Camera-only OD and SEG split-inference routes run locally and over OAI.
- RGB+radar fusion runs locally and over OAI, including multi-UE transport.
- OAI adds a large application-level RTT/tail penalty compared with loopback.
- Fusion_as_seg and fusion_as_od can now be evaluated separately.
- Pole-to-parked-ego transferability is measured: segmentation partially
  transfers, while object detection/localization drops enough to motivate
  guardrails or fine-tuning.
- The curbside hidden-pedestrian scenario provides a concrete safety case for
  spatial-map sharing.

The paper-critical gap is still the controller: the current evidence proves the
measurement problem is real; the final paper needs a system intervention that
improves outcomes over static baselines.

## Primary Scope

- Split-inference payload control for object detection and segmentation.
- Camera-only OD and SEG baselines.
- RGB+radar fusion model evaluated with separate OD and SEG metrics.
- OAI 5G as the real transport baseline.
- CARLA scenarios with controlled object density, occlusion, and ego/pole placement.
- Task guardrails for AP, mIoU, foreground IoU, and vulnerable-object recall.
- Spatial-map ingestion and selective sharing as the physical-AI extension.

## Explicit Non-Goals for the First Paper

- Full 3GPP QoS enforcement through PCF/NEF/NWDAF.
- Full PC5 sidelink implementation.
- Large-scale multi-city cooperative perception benchmark.
- End-to-end autonomous driving planner replacement.
- Generative reconstruction or neural codec comparison unless the core split-control result is already strong.

## Paper Structure Inspired by SCAN-AI

### 1. Introduction

- Problem: cooperative perception cannot send everything all the time.
- Safety risk: static compression can erase critical objects.
- Opportunity: scene-aware, model-aware, and network-aware split-inference control.
- Evidence hook: OAI transport makes a fixed fusion payload much slower and
  less reliable than loopback, while viewpoint transfer can break object-level
  utility.
- Contributions summarized clearly.

### 2. Motivation and Challenges

- Static compression knobs are scene-blind.
- Network-only adaptation is task-blind.
- Task metrics are not directly known online.
- Cooperative maps need freshness and relevance, not just more data.

### 3. Preliminary Measurements

Evidence we need before claiming an agent is useful:

- Payload changes under quantization/compression settings.
- Latency and timeout behavior over local and OAI 5G transport.
- AP/mIoU/foreground-IoU drop under aggressive settings.
- Scene density or occlusion causing harder perception outcomes.
- Spatial-map freshness and stale-object behavior.

Month 1 measurement figures already available or near-ready:

- Camera-only OD loopback vs OAI latency and OD quality.
- Camera-only SEG loopback vs OAI latency and SEG quality.
- RGB+radar fusion OAI vs loopback latency/receive-rate comparison.
- Fusion pole-vs-parked-ego transferability for SEG and OD.
- Curbside evidence pack demonstrating hidden-pedestrian failure geometry.

### 4. SceneSense Agent Design

- State:
  - Scene state: density, foreground fraction, occlusion indicators.
  - Model state: confidence, uncertainty, class risk.
  - Network state: RTT, loss, payload, queue delay, bandwidth estimate.
  - Map state: freshness, provenance, downstream vehicle risk.
- Actions:
  - Quantization.
  - AE channels, where supported.
  - ROI threshold, where supported.
  - Frame send/skip.
  - Redundancy.
  - Map update recipient/detail level.
- Reward:
  - Task utility retained minus byte/latency/loss/staleness cost.
- Guardrails:
  - Reject or clamp unsafe actions using validated thresholds and online proxies.

First implementation target:

- Start with a rule-based or contextual-bandit controller before full RL.
- Actions should be simple and measurable: send/skip, detail level, saliency
  drop/compression profile, stream priority, and spatial-map update detail.
- The controller should have an explicit fallback: if task risk is high or
  confidence proxy is low, use higher-detail evidence even when the network is
  pressured.

### 5. Implementation

- CARLA 0.10 testbed.
- Camera-only OD and SEG split-inference routes.
- RGB+radar fusion route with object and segmentation heads.
- OAI 5G transport path.
- Metrics repository and experiment manifests.
- Spatial-map server and map-update schema.

### 6. Evaluation

Core comparisons:

- Send-everything or highest-quality baseline.
- Static compression configurations.
- Lowest-byte policy.
- Rule-based adaptive policy.
- Learned policy.
- Learned policy with and without guardrails.

Required baselines:

- Local-only ego perception.
- Send-everything / highest-quality split features.
- Best fixed static policy selected offline.
- Network-only adaptive policy that ignores task utility.
- Task-only adaptive policy that ignores network state.
- V2Xverse/CoDriving-style collaborative driving or trace-level comparison.
- Where2comm-style selective sharing as the closest "what to communicate"
  baseline.

Core metrics:

- Payload bytes and chunks.
- RTT and total application latency.
- Timeout/no-result rate.
- OD AP, recall, localization error.
- SEG mIoU, foreground IoU, class IoU.
- Vulnerable-object recall.
- Map freshness, false hazard rate, stale-object rate.
- End-to-end warning or override latency.

### 7. Spatial-Map Case Study

- Occluded pedestrian or vehicle near intersection.
- Multiple sensing nodes observe different views.
- Server-side map-sharing policy decides what/when/who.
- Target outcome: the affected vehicle gets useful warning; unaffected vehicles are not spammed.

Concrete Month 1 case study candidate:

- Curbside hidden pedestrian behind parked vehicle(s).
- Ego camera misses or sees late; helper/observer view has earlier evidence.
- Spatial map should deliver a pedestrian/hazard update before the collision
  window, with lower byte cost than sending everything continuously.
- Metrics: warning lead time, vulnerable-object recall before collision,
  stale-object rate, false hazard rate, application RTT, and bytes per useful
  warning.

### 8. Discussion and Limitations

- Runtime guardrails rely on proxies, while AP/mIoU ground truth is offline.
- Pole-trained fusion model may not transfer directly to ego-mounted sensors.
- OAI transport baseline is real, but early experiments may not enforce full 3GPP QoS policies.
- RL must beat simpler heuristics to justify its complexity.

## Candidate Contributions

These should be tightened as results arrive:

1. A CARLA + OAI 5G cooperative-perception measurement stack that exposes
   payload, latency, receive-rate, task utility, and spatial-map freshness for
   OD, SEG, and RGB+radar fusion.
2. A task-aware/network-aware evidence controller that adapts perception
   evidence detail and stream priority under OAI transport pressure.
3. A guardrail layer that preserves safety-critical OD/SEG/fusion utility under
   compression and network stress.
4. A spatial-map sharing case study that routes cooperative perception updates
   based on risk, freshness, and bandwidth in a hidden-hazard scenario.

## MobiSys Acceptance Bar

Likely acceptable shape:

- Full working implementation with reproducible commands and artifacts.
- Strong measurement showing why fixed sharing fails over OAI/network stress.
- Controller that beats best fixed/static policies on both network cost and
  task utility.
- Closed-loop hidden-hazard case showing useful warning/map improvement.
- Honest comparison to V2Xverse/CoDriving and Where2comm-style assumptions.

Likely weak shape:

- OAI-vs-loopback measurement only.
- CARLA perception accuracy only.
- Fusion retraining only.
- RL claim without beating simple heuristics.
- Spatial-map visualization without downstream utility metrics.

## Month-by-Month Evidence Needed

Month 1:

- Reproducible baselines.
- Scenario harness.
- Metrics schema.
- OAI transport path.
- First payload/task traces.

Month 2:

- Static sweeps and payload characterization.
- Initial controller harness.
- Baseline comparisons.

Month 3:

- Network stress and guardrail ablations.
- Evidence that learned or adaptive policies avoid unsafe task collapse.

Month 4:

- Spatial-map ingestion validated against CARLA ground truth.

Month 5:

- Map-sharing policy evaluation.

Month 6:

- Occlusion-warning or navigation-override demo.
- Paper figures, tables, and narrative.

## Reviewer Questions To Keep Answering

- Why is an adaptive policy needed instead of static knobs?
- Why is RL needed instead of a simpler heuristic or bandit?
- What information is available online versus only for offline evaluation?
- Does the policy preserve task utility, or only reduce bytes?
- Does it generalize to new scenes or network conditions?
- What is the cost of the controller itself?
- Does the spatial-map sharing actually help a downstream vehicle?
- How does this differ from V2Xverse/CoDriving beyond using a different
  simulator?
- What is the strongest fixed baseline and why is it insufficient?
- Is OAI RF-sim/no-impairment representative enough, and what claims are not
  being made?
