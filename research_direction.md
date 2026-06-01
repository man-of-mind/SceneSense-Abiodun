# Research Direction — Network-Aware Selective Feature Transmission for Multi-UE Cooperative Perception

> **Status:** Living document. Last updated 2026-05-20.
> **Target venue:** ACM MobiSys.
> **Owner:** Abiodun (under Subhramoy's track in the IDCC × NEU collaboration).
>
> This is a working draft. Sections marked **[OPEN]** are unresolved questions
> we'll address through ongoing brainstorming sessions. Sections marked
> **[FIRM]** reflect our current shared understanding.

---

## 1. One-paragraph problem statement [FIRM]

A vehicle's perception is fundamentally limited by occlusion and viewpoint. Multiple connected vehicles can collectively perceive a scene better than any one of them alone — but **sharing perception requires transmitting intermediate model features over a cellular link**, and the bandwidth, latency, and reliability of that link are not free. We propose a *network-aware, scene-aware, model-aware* feature transmission policy for split-inference systems that **adaptively decides which feature tensors (and which parts of them) to transmit**, conditioned on current channel state, scene complexity, and the downstream task. The system extends SCAN-AI's single-UE cross-layer adaptation framework to the multi-UE cooperative-perception setting, where feature-level decisions must be made under real-time safety-critical constraints.

---

## 2. Motivation [FIRM]

### Why split inferencing?

The natural counter-argument — *"why not just put more compute on the edge?"* — has a real answer: **multi-UE cooperative perception fundamentally requires information sharing across vehicles, not just local compute scaling.** A car occluded by a truck cannot solve its problem by adding GPUs; it must receive information from a better-positioned vehicle. Split inferencing is the mechanism by which that information moves: instead of sharing raw video (too large) or final detections (too thin to fuse confidently), vehicles share **intermediate feature tensors** that are rich enough to fuse and small enough to ship.

### Why network-aware?

Feature tensors are still large — uncompressed Faster R-CNN FPN features at 1280×720 produce ~MBs of data per frame across 5 scales. The cellular uplink is shared, congested, and lossy. A perception system that ships a fixed payload regardless of channel conditions will either (a) over-spend bandwidth when conditions are good, starving other users, or (b) flood the link when conditions are bad, missing the safety-critical deadline. **The link state must inform the perception system's transmission decisions.**

### Why scene-aware?

Not all frames carry equal task value. A frame with a busy intersection containing pedestrians and crossing vehicles needs more spatial detail than a frame of empty highway. **The compression policy should reflect what the scene contains**, not apply a uniform rate.

### Why model-aware?

Different downstream models have different feature shapes and tolerances. Detection (sparse outputs) and segmentation (dense outputs) have *different payload sizes for the same input image*, and different sensitivity to compression error. **The network needs visibility into what model is running upstream of it** so resources can be allocated correctly.

### The gap

SCAN-AI (Mohanti et al., MobiCom 2026 submission) solved the single-UE cross-layer adaptation problem for video uplink, but only one UE and only one task (teleoperation). CoDriving (Liu et al., V2Xverse) demonstrates multi-agent cooperative driving with shared features but treats network conditions as a black box — robustness to bandwidth limits is evaluated but not actively optimized. **The gap: a learned, real-time, network-aware policy for selective feature transmission, across multiple cooperating UEs.**

---

## 2.5. Research strategy: gap-finding, not solution-first [FIRM — supervisor steer 2026-05-21]

Per supervisor guidance, the project methodology is **identify and address gaps in SOTA**, not "have an idea then find a fit." Concretely:

1. **Read and reproduce SOTA** in cooperative perception (CoDriving, Coopernaut, Where2comm, and adjacent).
2. **Run their solutions** (starting with V2Xverse / CoDriving since that appears to be the strongest recent baseline) and characterize their behavior.
3. **Document the limitations / drawbacks systematically** — what do they treat as black-boxed, abstract, or out of scope?
4. **Identify which limitations are addressable** given our testbed strengths (real OAI 5G stack, ray-traced channels, NWDAF/RIC integration paths).
5. **Pick a limitation we can attack** that matches the team's strategic positioning (AI-RAN / O-RAN, mobile systems).
6. **Develop the targeted improvement** and compare against the SOTA baseline using their metrics + ours.

The four gap categories supervisor named:

| Gap category | Examples of sub-questions |
|---|---|
| **What to share** | Which features, channels, scales, spatial regions get transmitted |
| **When to share** | Per-frame, on-demand, event-triggered, or scheduled |
| **Whom to share to** | All peers, selected neighbours, fusion server, hierarchical |
| **Orchestration / architecture** | Who coordinates the decisions; where does the policy live; how do decisions reconcile across UEs |

## Scope: what we are NOT doing [FIRM]

Per supervisor: **the contribution is NOT new perception algorithms.** We are consumers of perception models (Faster R-CNN, LR-ASPP), not designers of them. We don't beat anyone on COCO. We don't propose new fusion architectures. We don't propose new attention mechanisms for cooperative perception.

What we *do* contribute is the **systems layer that the perception community treats as a black box**: when/what/to-whom information moves, conditioned on the real network and the real task, on a real radio testbed.

## 3. Primary use case [FIRM]

**Cooperative perception for safety-critical autonomous driving over 5G.** Multiple vehicles in shared airspace transmit selected feature tensors to a fusion server (or to each other directly), enabling the perception system to see around occlusions. Safety-critical decisions (braking, lane change, hazard alert) depend on receiving relevant features within tight latency budgets.

**Secondary use cases worth considering as comparison points:**
- Roadside-infrastructure-assisted perception (RSU cameras share features with ground vehicles)
- Drone-assisted perception (overhead view shared with ground)
- Adaptive task offloading (per-frame decision: local / split / fully remote)

(See [[user_profile]] and [[research_direction]] memory entries for the full brainstorm history.)

---

## 4. Three dimensions of awareness — the framing [FIRM]

We organize the problem along three axes:

```
                 +--------- Scene awareness ---------+
                /  (scene complexity, object density, /
               /   motion, scale distribution,        /
              /    where the salient things are)     /
             +-----------------------------------+
            /                                       /
           +     Network                Model      +
           +     awareness              awareness  +
            \   (channel quality,       (which model is running,
             \   bandwidth, latency,    task type, feature shape,
              \  packet loss,           compression sensitivity,
               \ resource availability) model output rate)
                +---------------------+
```

The agent's input is some subset of these. The agent's output is a transmission policy.

---

## 5. Research questions [FIRM, ordered roughly by depth]

### RQ1 — Measurement
**What is the payload-vs-accuracy curve of selective feature transmission in our testbed, under static drop policies?**
This is the *baseline characterization*. Required before any adaptive policy can claim to "beat" something.

### RQ2 — Importance metric
**Is L2 norm of channel activations (the current saliency metric in the codebase) the right signal for selecting which feature cells to ship? What alternatives exist (gradient-based, learned, task-conditioned), and how do they compare on payload-vs-accuracy?**

### RQ3 — Adaptive single-knob policy
**Can a learned policy that adapts the saliency-drop fraction `q` based on current scene complexity outperform the best fixed `q`?**

### RQ4 — Multi-knob joint policy
**When multiple knobs are jointly optimized (drop fraction, bit depth, per-FPN-scale priority), does a learned multi-knob policy dominate any single-knob policy at the same average bandwidth?**

### RQ5 — Network-conditioned policy
**When channel-quality signals (e.g. CQI, throughput estimates from OAI) are added as input, does the policy gracefully degrade under bad channel and exploit good channel?**

### RQ6 — Multi-UE scaling
**As more vehicles share features in the same airspace, where do bottlenecks appear (radio resource, fusion server compute, latency)? At what scale does a coordination layer become necessary?**

### RQ7 — Coordination architecture **[OPEN]**
**Should the policy be centralized (one network agent decides for all UEs), distributed (each UE decides independently), or hierarchical (UEs make fast local decisions; network refines)? How does the answer depend on latency budget and number of UEs?**

### RQ8 — Latency-aware decisions **[OPEN]**
**How does the agent reason about its own decision latency? Is the decision policy itself fast enough to act within the safety-critical decision window?**

---

## 6. Proposed phased approach [FIRM at Phase 0-2, soft thereafter]

### Phase 0 — Foundation (DONE / DOING)
- Understand the codebase (split inference detection + segmentation, multi-sensor streaming) ✓
- Understand the slides (testbed implementation, split inferencing walkthrough) ✓
- Read SCAN-AI ✓; in-progress: V2X / CoDriving paper
- Have working development environment ✓

### Phase 1 — Baseline characterization (next 2-4 weeks)
- Run the existing `carla_split_inference_udp_segmentation_demo.py` with the **existing saliency gate** at varying `--saliency-drop-q` values (e.g. 0.0, 0.2, 0.4, 0.6, 0.8).
- Produce a **payload-vs-mIoU** (or mAP for detection) curve as the static baseline.
- Capture per-FPN-level payload contribution (the script already logs this).
- Run on supervisor's machine if local lag continues to interfere.
- Deliverable: a single figure showing the static-saliency Pareto frontier.

### Phase 2 — Importance-metric study (4-6 weeks)
- Compare alternative importance metrics on the same testbed:
  - L2 norm of activations (current)
  - Per-channel L2 / sparsity-weighted norm
  - Gradient-based importance (Grad-CAM style)
  - Learned auxiliary network (tiny saliency predictor)
- For each, the same payload-vs-accuracy curve.
- Deliverable: a comparative figure + recommendation of which metric to build on.

### Phase 3 — Adaptive single-knob policy (4-6 weeks)
- Train a policy (start with a simple bandit, scale to RL if needed) that selects `q` per-frame based on scene-complexity input only (SI/TI or a tiny scene classifier).
- Compare to the best fixed `q` from Phase 1.
- Deliverable: evidence that adaptive beats fixed, by how much.

### Phase 4 — Multi-knob policy (6-8 weeks)
- Expand action space to include: per-FPN-level `q`, bit depth, EMA range tracker `alpha`.
- Compare to single-knob Phase 3 policy.
- Deliverable: joint Pareto frontier; new vs old policy comparison.

### Phase 5 — Network-conditioning (6-8 weeks)
- Bring in channel-quality signals from OAI (throughput estimates, congestion indicators, packet loss).
- Train a network-conditioned policy.
- Compare across simulated channel conditions (good / moderate / bad).
- Deliverable: degradation graph showing the policy gracefully handles bandwidth pressure.

### Phase 6 — Multi-UE scaling **[OPEN]**
- Scale from 1 UE to N (2, 4, 8) sharing features for cooperative perception.
- Identify where the system breaks first (which resource saturates).
- Address coordination architecture (see RQ7).

### Phase 7 — Closed-loop QoS integration **[OPEN]**
- Wire policy decisions to OAI's QoS framework (PCF, NEF, NWDAF).
- Demonstrate end-to-end network-aware safety-critical pipeline.

---

## 7. Control knobs we can vary [FIRM]

### UE / camera-side knobs
- **Input resolution** (`--camera-resolution`, `model.transform.min_size/max_size`) — controls all downstream feature map sizes
- **Frame rate** (`--fps`) — affects bandwidth and freshness
- **Camera FOV** — affects scene composition but rarely varied

### Model-internal knobs
- **Which FPN scales to compute** (could skip P3 if no small objects expected) — current code does all 5
- **Bit depth** for quantization (currently fixed at 8) — could be 6, 4, mixed
- **EMA range tracker `alpha`** — controls quantization-range adaptation speed
- **Choice of detection vs segmentation** model — affects payload size and task

### Feature-prioritization knobs (the ones most relevant to *our* contribution)
- **Saliency drop fraction `q`** (`--saliency-drop-q`) — currently static
- **Per-FPN-level drop fraction** (could be (q3, q4, q5, qpool) instead of single q)
- **Saliency metric** (L2 norm is current; alternatives in RQ2)
- **Spatial priority bias** (give priority to certain regions of frame — e.g. road area, intersections)
- **Temporal priority** (frames at critical moments — sudden motion — get more bytes)

### Compression-pipeline knobs
- **Tile layout** (currently sqrt-factorization; could be packing-aware)
- **Symmetric flip on/off** (currently always on)
- **Codec choice** (zlib only currently; could swap to autoencoder)
- **Codec quality** (zlib levels; H.265 QP)

### Network-side knobs
- **Application bitrate target** (mappable to OAI QoS class)
- **Packet priority / DSCP marking** — needs CAP_NET_ADMIN
- **Retransmission policy** (FEC, ARQ) — out of scope initially

### Coordination knobs (multi-UE only) **[OPEN]**
- **Who sends what** (sender selection)
- **Aggregation strategy** at fusion server (average, attention, learned)
- **Trust / weighting** between UE contributions

---

## 8. Open design questions [OPEN — these are the brainstorming targets]

### Q1 — Where does the agent live? [updated 2026-05-21 with supervisor input]
| Option | What it means | Research community fit |
|---|---|---|
| **UE side** (application) | Car itself decides what to share | App-layer ML; lowest latency; only local info |
| **Fusion server side** (application) | Decisions at the aggregation point | App-layer ML; sees all UEs but adds RTT |
| **5G core network** (e.g. NWDAF) | Agent inside operator's analytics function | 3GPP standardization track |
| **RIC in O-RAN** (xApp/rApp) | ML application inside the RAN intelligent controller | **O-RAN / AI-RAN Alliance** — designed exactly for cross-layer ML, sees real-time channel state |
| **Hierarchical** | Fast local at UE; slow system-wide at RIC or core | Best of both worlds, but two policies to train and coordinate |

**Supervisor's emphasis:** the RIC option is taken seriously. Placing the agent as an xApp (Near-RT RIC, 10ms-1s loop) or rApp (Non-RT RIC, >1s loop) positions the work for the AI-RAN community, not just MobiSys. The two communities have overlapping but distinct review criteria — needs to be a deliberate decision.

### Q2 — How many agents?
| Option | Trade-off |
|---|---|
| **One per UE** | Simple; ignores cross-UE coordination |
| **One central** | Sees everything; latency cost; single point of failure |
| **One per service area** | Compromise; needs handoff logic |
| **Multi-agent RL (MARL)** | Principled treatment of coordination; harder to train |

### Q3 — What does the agent see?
At minimum: scene complexity, channel state, current payload size, recent accuracy proxy.
Open: should it see other UEs' policies? Other UEs' raw signals?

### Q4 — What does the agent output?
Direct knob values? A policy distribution? A latent representation that maps to knobs?

### Q5 — How is the agent trained?
Offline RL from CARLA traces? Online RL in CARLA? Simulator-to-real (NEU 5G testbed)?
Reward signal: combination of detection accuracy + bandwidth + latency? How are they weighted?

### Q6 — Latency budget of the policy itself
The agent's decision must come *before* the frame is encoded. How fast can it be? Microseconds? Single-millisecond? Does that constrain agent architecture (no large neural nets)?

---

## 9. Baselines we will compare against [FIRM at low end, growing]

1. **No compression / raw transmission** — theoretical upper bound on accuracy, infeasible bandwidth. Sanity check.
2. **Pixel-codec baseline (H.264/H.265 of raw video)** — what you'd do without split inferencing at all. Frames standard codec as the *machine-blind alternative*.
3. **Static saliency gating** — the existing code with `q = 0.0, 0.2, 0.4, 0.6, 0.8` fixed. Our system's *no-policy* baseline.
4. **Uniform compression** — same compression rate applied without saliency selection.
5. **Best fixed `q` for the dataset** — strongest static baseline.
6. **(Later)** Existing learned approaches from VCM literature.

A good MobiSys paper typically shows: clear win over the strongest static baseline + clear win in a regime where the static baseline breaks (high bandwidth pressure, congested channel).

---

## 10. Experimental plan [SOFT — to refine]

### Metrics we will report
- **Task accuracy:** mAP (detection), mIoU (segmentation)
- **Payload:** mean and tail (95th, 99th percentile) bytes/frame
- **Latency:** end-to-end perception latency (frame capture → fused detection); decision latency (agent step)
- **Reliability:** fraction of frames where safety-critical detection arrives within budget
- **System cost:** CPU/GPU utilization on UE and server

### Scenarios to test
- **Channel conditions:** good (>50 Mbps available), moderate (10-50), bad (<5), bursty
- **Scene complexity:** sparse (highway), moderate (suburb), dense (urban intersection)
- **Number of cooperating UEs:** 1, 2, 4, 8 (in later phases)

### Test environments
- **CARLA** (primary, controllable, ground-truthed)
- **NEU 5G testbed** (if we get access — gives real channel conditions)

---

## 11. Related reading / literature map [FIRM, growing]

### Direct foundations
- **SCAN-AI** (Mohanti et al., MobiCom 2026 submission) — `abiodun/SCAN_AI_03_13_26_2.pdf`. The single-UE cross-layer adaptation we generalize from.
- **CoDriving / V2Xverse** (Liu et al., arXiv 2404.09496) — `abiodun/V2X_for_AD.pdf`. Multi-agent cooperative driving; treats network as black box.

### Video coding for machines (the field we sit in)
- **MPEG VCM (Video Coding for Machines)** standardization activity — search for recent VCM CTC documents.
- **Choi et al.** — Information-bottleneck approaches for feature compression.
- **CompressAI / CompressAI-Vision** (InterDigital) — the comparison codebase the testbed walkthrough explicitly references.

### Feature importance / pruning (RQ2 background)
- **Liu et al., "Learning Efficient Convolutional Networks through Network Slimming"** — L1/L2-based channel importance.
- **Grad-CAM** (Selvaraju et al.) — gradient-based saliency.
- **Knowledge distillation feature matching** — FitNets, attention transfer.

### Cooperative perception (multi-agent context)
- **CoDriving / V2Xverse** (Liu et al., arXiv 2404.09496, 2024-25) — the strongest recent baseline per supervisor. *Read with the differentiation-table lens — see [brainstorm_log.md](brainstorm_log.md) for the table template.*
- **Coopernaut** (Cui et al., CVPR 2022) — *"End-to-End Driving with Cooperative Perception for Networked Vehicles"*. Predecessor of CoDriving; another comparator.
- **Where2comm** (Hu et al., NeurIPS 2022) — selective communication for cooperative perception, closest existing work to our "what to share" question.
- **DiscoNet** (Li et al.) — earlier cooperative perception.
- **F-Cooper** (Chen et al., 2019) — feature-level cooperative perception.
- **V2X-ViT** (Xu et al., ECCV 2022) — transformer-based V2X perception.
- **CoBEVT / CoBEVFlow** (Xu et al.) — cooperative BEV (bird's-eye view).
- **AttFuse, FPV-RCNN, V2VNet** — common cooperative-perception comparators.
- **DAIR-V2X, V2V4Real, OPV2V, V2XSIM2.0, TUMTraf-V2X** — V2X benchmark datasets.

### RL / cross-layer optimization (Phase 3+)
- **SAC** (Haarnoja et al.) — the RL algorithm SCAN-AI uses. Likely our starting point.
- **FiLM** (Perez et al.) — feature-wise linear modulation; SCAN-AI uses this for cross-layer conditioning.

### Network / radio side
- **3GPP NWDAF (TS 23.288, TS 29.520)** — for network state exposure (core-network path).
- **O-RAN architecture documents** — for the RIC option. Key concepts: Near-RT RIC, Non-RT RIC, xApps, rApps, E2 interface, A1 interface.
- **OAI documentation** — for the actual platform.

> **Add to this list as we read.** Each entry should eventually have a one-line note on *what it gives us / what gap it leaves*.

---

## 12. Differentiation from existing work [SOFT — to sharpen]

> *This is the section that needs to be airtight before MobiSys submission. Sharpen with supervisor.*

| Prior work | What it solves | What it leaves open (our contribution) |
|---|---|---|
| SCAN-AI | Single-UE cross-layer video adaptation | Multi-UE, feature-level (not video-level), cooperative perception |
| CoDriving / V2Xverse | Multi-agent perception fusion | Network is black box; no learned transmission policy |
| Where2comm | Selective communication via attention | No network-state conditioning; offline-trained; no real radio testbed |
| MPEG VCM / CompressAI-Vision | Feature compression codecs | Codec is fixed; no policy layer; no network awareness |

---

## 13. Notes from supervisor (2026-05-21 sync — *latest*) [FIRM]

- **Strategy: gap-finding, not solution-first.** Read SOTA, find drawbacks, address them. See §2.5.
- **CoDriving (2024-25) is likely the strongest baseline / SOTA reference point.** Install V2Xverse, run their reference solution, document limitations.
- **Coopernaut (CVPR 2022) is the other comparator** to cover. CoDriving's "first of its kind" claim is about their specific combination; Coopernaut existed first in a different form.
- **Identify a broader set of related solutions** through the cooperative-perception literature.
- **Scope: not perception algorithms.** Contribution is in *how information is shared* — what/when/whom/orchestration. See "Scope: what we are NOT doing".
- **Architectural placement is a real question.** Options include UE, fusion server, core network (NWDAF), or RIC in O-RAN. See §8 Q1 (updated).
- **Keep an open mind** — problem statement not finalized; project is still in formulation phase.

## 13b. Notes from supervisor (2026-05-19 sync) [FIRM]

- **System lag** on the dev machine is an environment issue, not a code issue. Don't optimize further here; can use his machine if needed.
- **Visualization isn't required** for research output. CSV logs and offline analysis are what matter. CARLA is *just visualization*.
- **Direction:** focus on understanding frameworks and formulating the problem statement; then discuss specifically:
  1. How will the model be trained to select which tensors are important to send?
  2. How does the network influence that selection?
  3. Latency-aware decisions.
- After this document is in good shape, schedule a sync to align on phase priorities and the supervisor's specific preferences on the open questions in §8.

---

## 14. Next actions [FIRM — updated 2026-05-21]

1. **Install V2Xverse** (after checking with supervisor where to install — possibly on his machine; check CARLA version compatibility). Run CoDriving reference solution to verify.
2. **Read CoDriving with the differentiation-table lens.** Fill in §12 row by row as you go.
3. **Read Coopernaut** (CVPR 2022) and add to the differentiation table.
4. **Skim adjacent works** (Where2comm, V2X-ViT, F-Cooper) and note which they treat as black-boxed.
5. **Read the saliency-gating implementation** in [`carla_split_inference_udp_segmentation_demo.py`](../carla_split_inference_udp_segmentation_demo.py) (around lines 1003 and 1629) — this is the *primitive* form of what your research extends.
6. **Read up on O-RAN / RIC / xApp basics** if the architectural question survives further conversations. Start with O-RAN Alliance whitepapers.
7. **Bring filled-in differentiation table + V2Xverse run results to next supervisor sync.** Use this to discuss which gap to attack first.

(Phase 1 experiments — payload-vs-accuracy curve — are still planned but now happen after the SOTA characterization, not before.)

---

## Change log

- **2026-05-21** — Supervisor sync. Added §2.5 (research strategy = gap-finding). Added "Scope: what we are NOT doing". Updated §8 Q1 with RIC / O-RAN option and community-fit framing. Added Coopernaut and other comparator papers to §11. Restructured next actions (§14) — SOTA characterization comes before Phase 1.
- **2026-05-20** — Initial consolidation. Captures phased plan, control knobs, open questions, related reading, and supervisor's recent steer.
