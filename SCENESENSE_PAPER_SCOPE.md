# SceneSense Agent Paper Scope

Living outline for turning the project into a focused MobiSys-style systems paper.

This document should evolve as experiments produce evidence. The goal is to keep the research story sharp while the engineering work grows.

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

### 8. Discussion and Limitations

- Runtime guardrails rely on proxies, while AP/mIoU ground truth is offline.
- Pole-trained fusion model may not transfer directly to ego-mounted sensors.
- OAI transport baseline is real, but early experiments may not enforce full 3GPP QoS policies.
- RL must beat simpler heuristics to justify its complexity.

## Candidate Contributions

These should be tightened as results arrive:

1. A task-aware split-inference control framework that adapts payload knobs using scene, model, and network state.
2. A CARLA + OAI 5G evaluation harness for OD/SEG/fusion payload characterization.
3. A guardrail layer that preserves safety-critical task utility under compression and network stress.
4. A spatial-map sharing extension that routes cooperative perception updates based on risk, freshness, and bandwidth.

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
