# Brainstorm Log

> Raw idea capture, conversation notes, half-formed thoughts.
> Newest entries at the top.
> When an idea matures, promote it to [research_direction.md](research_direction.md) and reference it back here.

---

## Running follow-ups (open items to chase)

- [ ] Read [`carla_split_inference_udp_segmentation_demo.py`](../carla_split_inference_udp_segmentation_demo.py) saliency gating code in depth (~line 1003, ~line 1629).
- [ ] Read **Where2comm** (Hu et al., NeurIPS 2022) — closest prior work on selective cooperative communication.
- [ ] **Finish CoDriving paper with the differentiation-table lens (see 2026-05-21 entry).**
- [ ] Run Phase 1 baseline (payload-vs-accuracy at different `--saliency-drop-q`).
- [ ] Survey what feature-importance metrics exist in the pruning / knowledge-distillation literature.
- [ ] Check whether the testbed scene-complexity signal (SI/TI from SCAN-AI) is already wired into the split-inference scripts, or only into `scan_sender_v2*`.
- [ ] **Investigate V2Xverse as a complementary scenario source on top of OAI (rather than as a replacement for CARLA).**
- [ ] Promote RQ9 (graceful degradation under low peer density) to [research_direction.md](research_direction.md) §5 if it survives supervisor review.
- [ ] Ask supervisor: which of the §8 open design questions in [research_direction.md](research_direction.md) does he have strong opinions on?
- [ ] Ask supervisor: target paper length for the Phase 1 deliverable — workshop submission, or build toward full MobiSys?

---

## 2026-05-21 (later) — Supervisor sync

### Key outcomes
- **Strategy = gap-finding.** Don't solution-first. Read SOTA, find what they treat as black-boxed or abstract, then formulate the problem statement against that gap.
- **CoDriving is the relevant recent baseline.** Coopernaut (CVPR 2022) is another. Both live in the cooperative-perception community; both likely treat the network as a hyperparameter, not a signal.
- **Scope: NOT perception algorithms.** Our contribution is systems-layer (information sharing, orchestration, architecture). We are consumers of perception models, not designers.
- **Four gap categories** to look for in any SOTA paper:
  1. What to share
  2. When to share
  3. Whom to share to
  4. Orchestration / architecture
- **Architectural placement options for the agent** (added to §8 Q1 of research_direction.md):
  - UE (application layer on car)
  - Fusion server (application layer on cloud/edge)
  - 5G core network (e.g. NWDAF — 3GPP analytics)
  - **RIC in O-RAN** (xApp on Near-RT RIC, or rApp on Non-RT RIC — designed for cross-layer ML, sees real-time channel state)
  - Hierarchical (fast UE + slow RIC/core)
- **Keep an open mind** — problem statement not finalized; this is the formulation phase, not the execution phase.

### Things I (Abiodun) was unsure about → notes for self
- **RIC / O-RAN / xApp**: O-RAN is the open-RAN industry initiative; RIC is the AI/ML control component (Near-RT for <1s loops, Non-RT for >1s); xApps and rApps are the ML applications that run inside the RICs. Placing the feature-transmission agent at the RIC = positioning the work for the **AI-RAN Alliance** community, which IDCC is already engaging with.
- **"Between network and application layer"** = somewhere in the protocol stack where you can see both app-level intent (which features matter) and network-level state (which channel is good). The RIC is one natural spot; NWDAF is another.

### Immediate next step
**Install V2Xverse and run their CoDriving reference solution.** Note limitations as I go.

Caveats:
- Check **CARLA version compatibility** before installing V2Xverse alongside CARLA 0.10. They may want an older CARLA. Use a separate environment if so.
- Check with supervisor whether to install on this dev box or his machine.
- Don't break the existing CARLA 0.10 setup that the team uses.

### Coopernaut quick reference
- Cui et al., *"Coopernaut: End-to-End Driving with Cooperative Perception for Networked Vehicles"*, CVPR 2022.
- Earlier cooperative driving in CARLA via V2V messages.
- Likely cited in CoDriving's related work — that's the easy place to see how CoDriving positions itself against it.

### Other works to find (literature map)
Where2comm, DiscoNet, F-Cooper, V2X-ViT, CoBEVT, CoBEVFlow, AttFuse, FPV-RCNN, V2VNet. All in the perception/V2X community. Look for what they treat as black-boxed on the network side — that's our opening.

### Differentiation table — promoted to "core methodology"
The differentiation table (research_direction.md §12) is now central, not optional. Every paper read should fill in one row. After reading 4-6 papers it will be clear which gap is largest and most attackable.

---

## 2026-05-21 — Session: CoDriving-induced existential moment + differentiation analysis

### What triggered it
Started reading CoDriving / V2Xverse paper. First few pages felt like the project had already been done. Classic "have I been scooped?" PhD moment.

### Why it's actually a *good* sign
- This is the universal experience of reading the most-relevant prior work. It means you're doing your due diligence.
- The right reframe: don't ask "did they do my project?", ask "**what did they treat as a black box that I'll open up?**"

### Concrete differentiation from CoDriving (to verify by reading the rest)
CoDriving's likely assumptions (which become *our* contribution surface):
1. **Network = abstract bandwidth limit.** No real channel state input. Robustness *to* constraints, not adaptation *to* signals.
2. **No real radio testbed.** Pure V2Xverse simulation. No OAI, no real RAN/PHY.
3. **Policy is heuristic, not learned and real-time network-conditioned.** "Regions near planned waypoint" sounds like a rule.
4. **No QoS-class / 3GPP integration.**
5. **Single time scale.** Per-frame selection only; no fast/slow loop architecture.
6. **No sender selection / multi-access problem.** "Who sends" at the radio layer is out of scope.

If even half of these hold, the contribution surface is large and well-defined.

### Venue logic
CoDriving targets perception venues (CVPR/ICCV/RSS/IROS). MobiSys is a different community with different review criteria:
- Real systems (real testbed, real radio)
- Cross-layer wins (network state ↔ application decisions)
- Quantified system costs (latency, throughput, scaling)
- 3GPP-relevant insights

A paper that satisfies CoDriving's reviewers does not satisfy MobiSys's, even with overlapping vocabulary.

### V2Xverse question
**Conclusion: complement, don't replace.** Use V2Xverse for scenario generation (multi-vehicle benchmarks, reviewer comparability) but keep the OAI 5G stack for the actual network testbed. The *combination* (V2Xverse scenarios + real 5G PHY/MAC + ray-traced channels + QoS framework) is a stronger setup than either alone, and it's the differentiator that pure-V2Xverse papers cannot offer.

### New research question raised: RQ9 — graceful degradation
**"What if there are no peer vehicles to share with?"** — Abiodun spotted this edge case unprompted.

Reframe as a research question:
> RQ9: How does the system's safety-critical performance scale with peer density (cooperator count per scene)? Below what density does cooperation provide measurable improvement, and below that does the system safely fall back to local-only perception without increasing risk?

This is MobiSys-friendly: ties algorithm behavior to deployment conditions and makes the work robust to "what if no peers?" reviewer pushback. Also opens the door to RSU (roadside infrastructure / streetlamp cameras) as permanent cooperators that solve the "no peer" problem at intersections — already hinted at in the team's slide 9.

### Differentiation-table exercise
When reading CoDriving (and later Where2comm), build the table as you go:

| Sub-problem | CoDriving | My answer (TBD) |
|---|---|---|
| What gets selected? | driving-critical regions near planned waypoint | content × channel × model task |
| When does selection adapt? | ??? (fill in) | per-frame, conditioned on real channel state |
| Network model | abstract bandwidth limit | real OAI 5G + SionnaRT channel |
| Policy learning | (likely offline / heuristic) | online RL with cross-layer state |
| Sender selection (multi-UE) | (out of scope?) | explicit research question |
| ... | ... | ... |

This table fills out §12 of [research_direction.md](research_direction.md).

---

## 2026-05-20 — Session: deck-2 walkthrough + research framing

### What came up
- The split-inferencing walkthrough deck uses YOLOX `[128, 40, 40]` as a single-tensor running example; the testbed uses Faster R-CNN + FPN (5 tensors). Same compression pipeline (quantize → tile → flip) applies to both. The deck simplification is pedagogical.
- The "code for machines, not for humans" framing (slide 7) is the *premise slide* of the field. Standard codecs (JPEG/H.264) preserve perceptual quality (PSNR, SSIM); we preserve *task quality* (mAP, mIoU). When the decoder is a neural network, we can be far more aggressive about what we throw away.
- 192 Mbps/camera at raw float32 feature transmission — slide 8 quantifies why this is unaffordable on a real cellular uplink. Multi-UE multiplies this. Strong motivation paragraph material.

### Key discovery (codebase)
**`saliency_drop_masks()` in the segmentation demo already implements feature-importance-based dropping**, using L2 norm of channel activations per cell as the importance metric. The drop fraction `q` is controlled by `--saliency-drop-q` (currently a static CLI arg). The code comment explicitly says it's the segmentation analogue of the OD RPN-objectness gate.

**Implication:** the research isn't to *build* a saliency gate — it's to *upgrade* the existing one from static to dynamic, network-aware, scene-aware, model-aware.

### My (Abiodun's) intuition that landed
- FPN matters for self-driving because objects span huge size ranges (far pedestrian to nearby truck) → multi-scale features needed → all 5 always produced, but *activation strength per scale* varies with scene.
- Different models have different payload sizes (segmentation > detection because dense outputs need more spatial detail to be preserved). The network must know which model the UE is running.
- "Lot of awareness need to be considered" — captured as the three-axis framing (network / scene / model) in §4 of research direction.

### Control knobs I thought of
- Input resolution (`rcnn-min/max-size`) → cascades to all FPN sizes
- Pre-compute RPN on UE so we can zero out low-importance regions (note: this actually moves the split point; it's a different architecture, not just a knob)
- ROI-objectness threshold for zeroing out clear-sky-like regions
- Per-FPN-level payload probing (already exists in metrics CSV) → use to identify which scale is "expensive" per byte
- (Added by Claude) Bit depth (currently 8, could be 6/4 mixed)
- (Added by Claude) EMA range tracker `alpha` (controls quantization-range adaptation speed)

### Open questions that surfaced (now in research_direction.md §8)
- Where does the agent live? (UE / network / distributed / hierarchical)
- How many agents?
- What does it see / what does it output / how is it trained?
- What's the agent's own latency budget?

### Next actions
- Created [research_direction.md](research_direction.md) (structured, MobiSys-target paper-shaped).
- Created [brainstorm_log.md](brainstorm_log.md) (this file, raw capture).
- Plan: read saliency-gating code; run Phase 1 baseline; sync with supervisor.

---

## How to use this log

- **Capture early, refine later.** Half-formed ideas go here first; mature ones graduate to `research_direction.md`.
- **Date every entry.** Even quick captures.
- **Mark follow-ups with `- [ ]` checkboxes** at the top so the running task list stays visible.
- **Cross-link**: when an idea ends up in `research_direction.md`, link back to the section.
- **Don't delete obsolete entries** — strike-through (`~~text~~`) or annotate why an idea was dropped. Reviewer questions can resurface them.
