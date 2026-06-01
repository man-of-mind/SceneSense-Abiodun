# Abiodun's working notes — IDCC × NEU `neu_collab` project

> Living document. Updated each working session. Started **2026-05-18**.

---

## 1. My research direction (open)

**The pipeline is fixed: split inference + feature compression + network-aware transmission, extending SCAN-AI.** The *use case* is still open — supervisor explicitly said keep an open mind. Multi-UE cooperative perception is the lead candidate; see Section 4 for the brainstorm of alternatives to discuss with him.

**Lead candidate: Multi-UE cooperative perception over 5G** — many vehicles sharing perception features so occluded cars benefit from better-positioned cars' views.

### The intuition (supervisor's framing)

A car on a busy road may not see everything — a pedestrian crossing might be hidden by a parked truck. Another nearby car has a clearer view. *What if cars could share what they see, so each can make more informed decisions?* That's cooperative perception. The research question: **how do we make this work over a real cellular network, for many cars, in real time?**

### Three sub-problems my supervisor named

1. **Latency** — a "pedestrian ahead!" feature that arrives 300 ms late is worse than useless.
2. **Network-aware prioritization** — when the radio is congested, the network must reserve resources for safety-critical feature flows. The system has to know *what* is being shipped and *why it matters* (channel quality + resource awareness as inputs).
3. **Coordination** — with N cars, who shares what to whom? Avoid broadcast storms. Favor unique-information senders.

### Where my work starts (vs SCAN-AI)

I'm extending **SCAN-AI** (Mohanti et al., MobiCom 2026 submission — PDF in this folder). SCAN-AI is the **single-UE** version: one car uploading H.265 video for teleoperation, with proactive cross-layer adaptation (scene complexity + vehicle telemetry + channel state). My job is to scale this up to **multi-UE cooperative perception**, where the new problems are latency budgets, multi-flow prioritization, and sender/receiver coordination.

---

## 2. Why split inference is the enabling mechanism

This was the part that didn't click for me at first. Here's the way it clicked:

**Three options for what cars share with each other:**

| Option | What's shipped | Size | Why it doesn't (alone) work |
|--------|----------------|------|------------------------------|
| A. Final detections | "Pedestrian at GPS X,Y, confidence 0.84" | Tiny | Too thin. Car A must blindly trust Car B's model. No way to fuse evidence. |
| B. Raw camera frames | Full RGB frame | Huge | Wireless can't carry it for many cars. Latency budget blown. |
| C. **Intermediate feature maps** | FPN tensors from inside the model | KBs/frame after compression | **Sweet spot.** Rich enough to fuse, light enough to ship. |

Option C **is** split inference. The sender runs the front half locally, compresses the feature maps, ships them. The receiver fuses the received features with its own and runs the back half.

```
       Car B (sees pedestrian)            Car A (occluded)
       ───────────────────────            ───────────────
       Camera frame                       Camera frame
            │                                  │
            ▼  front half                      ▼  front half
       FPN feature maps                   FPN feature maps
            │                                  │
            ▼  quantize + compress + UDP       │
            └────► OAI 5G uplink ─────────────►│
                                                ▼
                                          fuse features
                                                │
                                                ▼  back half
                                          detection (now sees
                                          the pedestrian)
```

**Crucially:** the existing [`carla_split_inference_udp_demo.py`](../carla_split_inference_udp_demo.py) is exactly this pipeline, just generalized to "one camera → one server." My job is to extend it to "N cars → fused inference," with the network-aware and coordination layers on top.

---

## 3. The two foundational papers

### SCAN-AI ([SCAN_AI_03_13_26_2.pdf](SCAN_AI_03_13_26_2.pdf))
- *Mohanti et al., MobiCom 2026 submission.*
- Single-UE video uplink for teleoperation. Cross-layer proactive adaptation.
- Inputs fused: scene complexity (spatial entropy, edge density, temporal motion), vehicle telemetry (speed, accel, turning), real-time network statistics.
- Technique: **FiLM-style cross-layer modulation** + **SAC reinforcement learning** with continuous action space → fine-grained H.265 encoder control.
- Testbed: CARLA + H.265 + OpenAirInterface 5G + NVIDIA SionnaRT ray tracing.
- Result: **zero packet loss** vs. ~13.59% for baseline ABR; interpretable agent decisions.
- *This is the paper my work directly extends.*

### V2X / CoDriving ([V2X_for_AD.pdf](V2X_for_AD.pdf))
- *Liu et al., "Towards Collaborative Autonomous Driving: Simulation Platform and End-to-End System", arXiv 2404.09496v2.*
- Introduces **V2Xverse** (CARLA-based collaborative-AD simulation platform; open-source on github: CollaborativePerception/V2Xverse) and **CoDriving** (end-to-end driving system with shared perception).
- Key technique: **driving-oriented communication strategy** — agents request shared features only for "driving-critical regions" near planned waypoints. *Selective* feature sharing.
- Benchmark datasets cited: DAIR-V2X, V2V4Real, OPV2V, V2XSIM2.0, TUMTraf-V2X.
- Result: +62.49% driving score, -53.50% pedestrian collision rate vs. SOTA single-agent.
- *This is the perception-side reference. The network-side robustness story is where my contribution lives.*

---

## 4. Mental model: the three demos sit at different points in a design space

The three scripts my supervisor gave me are **not all the same thing applied to different problems**. They're three different points in a 3D design space:

- **What gets shipped:** raw sensor data / intermediate features (split inference) / final outputs
- **What downstream task:** detection / segmentation / multi-task / tracking / ...
- **What topology:** single UE→server / multi-UE→server / V2V / UE→RSU / drone→ground / ...

The three demos currently sit at:
- **Detection demo** = features + detection + single-UE→server
- **Segmentation demo** = features + segmentation + single-UE→server
- **Multi-sensor demo** = raw + whole-scene + single-UE→server

The "open mind" my supervisor wants is because **picking a use case = picking which point in this grid I extend, and which demo is my foundation.** Different use cases use different demos as their primary starting point:

| Use case | Primary demo |
|---|---|
| Multi-UE cooperative perception | Detection (or segmentation) |
| Drone-assisted perception | Segmentation (top-down) |
| Multi-modal split inference on single UE | Multi-sensor |
| Tele-op assistance (closest to SCAN-AI) | Multi-sensor |
| Privacy-preserving cooperative sensing | Detection / segmentation |
| Adaptive task offloading | All three (different operating points) |

So when I run the demos this week, I'm not just "running them" — I'm sampling three different research starting points and forming an instinct for which to extend.

---

## 5. Candidate use cases for the pipeline (brainstorm)

Supervisor told me to keep an open mind — multi-UE is one option, not the only one. To bring to next sync:

**Tier 1 — closest to existing team work:**
1. **Multi-UE cooperative perception (V2V or V2N).** *Lead candidate.* Cars share features so occluded cars benefit. Pedestrian-behind-truck scenario supervisor described.
2. **Roadside-infrastructure-assisted perception.** RSUs at street-lamp height share features with ground vehicles. Same pipeline, different topology.
3. **Drone-assisted perception.** Drones share top-down features with ground vehicles. Leverages Dale's AirSim work + Matteo's drone-YOLO thesis. Drones see over occlusions — uniquely valuable views.
4. **Multi-modal split inference on a single UE.** Adaptive selection of camera/radar/LiDAR features per frame.

**Tier 2 — same pipeline, different framing:**
5. **Adaptive task offloading.** Per-frame decision: local / split / fully remote. Systems paper.
6. **Multi-task split inference.** One feature stream → detection + segmentation + tracking from the same UDP transmission.
7. **Tele-operation assistance with shared features.** Continuation of SCAN-AI's tele-op framing.

**Tier 3 — stretch:**
8. **Privacy-preserving cooperative sensing.** Features reveal less than raw video — quantify the privacy gain.
9. **Heterogeneous fleet sensing.** Mixed-sensor vehicles contribute what each has.

**What to say to supervisor:** "I want to confirm multi-UE is the lead, but I sketched these alternatives so we can compare. Which resonates? Anything I haven't thought of?"

---

## 6. Strawman research plan (refine with supervisor)

> This is a starting point I built with Claude. Treat it as a sketch to discuss with my supervisor, not a commitment.

### Phase 0 — Grounding (weeks 1–2)
- Deep-read SCAN-AI; map each component to what's already in the testbed.
- Skim CoDriving paper; note feature-sharing assumptions, mark what they treat as solved (network) vs. open.
- Run [`carla_split_inference_udp_demo.py`](../carla_split_inference_udp_demo.py) end-to-end. Confirm working testbed on assigned hardware.
- Confirm the open questions in Section 6 with supervisor.

### Phase 1 — Two-UE feature sharing (weeks 3–5)
- Spawn two CARLA vehicles with deliberately overlapping FOV (occlusion scenario).
- Each runs front-half split inference locally.
- Ship features from Car B to Car A — first over localhost UDP, then over OAI 5G.
- Implement minimal fusion before the back half. **Start classical** (geometric warp + concat); learned cross-attention later.
- **Demo target:** Car A detects the occluded pedestrian *only when* it receives Car B's features.
- **Measurements:** end-to-end latency, payload bytes per frame, detection accuracy gain vs single-UE.

### Phase 2 — Scalability to N UEs (weeks 6–8)
- 4 → 8 → 16 cars. Identify what breaks first: bandwidth, latency, or coordination overhead.
- Naive baseline: everyone broadcasts. Smart variants: sender selection by view diversity, geographic anchor, or trust score.
- *This is the new contribution vs SCAN-AI* (which never dealt with multi-UE contention).

### Phase 3 — Network-aware feature transmission (weeks 9–12)
- Borrow SCAN-AI's three signals (scene complexity, vehicle telemetry, channel quality from OAI).
- Action space adapted to features: which FPN scales, which spatial regions, what bit depth.
- RL agent (SAC-style starting point, evaluate alternatives) decides per frame what to ship.
- Compare against fixed-policy baselines.

### Phase 4 — Closed-loop QoS integration (weeks 13+)
- Wire to OAI QoS framework (PCF, NEF, NWDAF) — coordinate with Subhramoy's OAI work.
- Closed loop: cars flag safety-critical content → network up-prioritizes radio resources → cars ship at higher fidelity.
- Final evaluation: collision-rate metric (CoDriving-style) under realistic 5G channel conditions.

---

## 7. The team (and who does what)

| Name | Role | Their track |
|------|------|-------------|
| Dale | IDCC project lead | OAI proposals, NLM design, AirSim |
| Subhramoy | IDCC senior IC | Owns SCAN-AI, traffic characterization, arch doc — my closest collaborator |
| Roya | IDCC | Leading Lights award |
| Quang | IDCC | Suggested navigation/path-planning enhancements |
| Michele | NEU lead | Academic framing, AI-RAN WG3 |
| **Mateo / Matteo** | **NEU intern (ends Aug 2026)** | **NLM + parking-spot demo — separate from my track** |
| MJ | NEU PhD (new) | Experiment support |

**Key clarification I had to figure out:** Mateo's parking-spot demo and NLM ("where can I park?") work is **a separate research thread** running on the same testbed. Not mine. The 4-week plan in the meeting notes was his roadmap.

---

## 8. Open questions for supervisor

1. **Scenario scope:** V2V (car-to-car direct), V2N (car-to-network fusion), or both?
2. **Headline task:** Detection (Faster R-CNN), segmentation (LR-ASPP), or something else like BEV / tracking?
3. **Publication target + rough deadline?** MobiCom follow-up? IWPC? AI-RAN Alliance work item?
4. **Deployment requirement:** Demo on real 5G hardware (NEU testbed) or OAI emulation enough?
5. **SCAN-AI reuse:** How much of SCAN-AI's RL infrastructure is reusable for multi-UE vs needs re-architecting?
6. **Subhramoy's task list:** Some items in meeting notes are "Subhramoy: ..." — which are still his vs. mine now?
7. **Ground truth pipeline:** Do we have a working pipeline measuring detection mAP / IoU vs CARLA ground truth, or is everything visual right now?

---

## 9. Editing convention (from supervisor)

**Do not edit any existing top-level script in `neu_collab/`.** When I want to change something, copy into `abiodun/` first. Example:
```bash
cp ../carla_split_inference_udp_demo.py carla_split_inference_udp_demo_abiodun.py
```

---

## 10. Command cheatsheet

### Start CARLA server
```bash
source /home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/activate
sh /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/CarlaUnreal.sh
```

### Split-inference detection (the demo most relevant to my work)
```bash
source /home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/activate
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/
python3 carla_split_inference_udp_demo.py --camera-resolution 1080p
```

### Split-inference segmentation
```bash
python3 carla_split_inference_udp_segmentation_demo.py --camera-resolution 1080p
```

### Multi-sensor raw streaming (baseline only)
```bash
# Terminal 1
python3 carla_multisensor_udp_gstreamer_receiver_v2.py

# Terminal 2
python3 carla_multisensor_udp_gstreamer_sender_v2.py
```

---

## 11. Running log

### 2026-05-18 — kickoff session
- Got initial codebase tour; misread the project as parking-spot-focused. Corrected after sharing meeting notes + papers.
- Mapped my track to **split-inference + feature-compression + network-aware pipeline, extending SCAN-AI**. Lead use case: multi-UE cooperative perception, but supervisor wants open mind.
- Got the lightbulb on how split inference enables feature sharing between cars.
- Note: I've already read SCAN-AI and presented it to supervisor — well-received. V2X paper not yet finished (skim now, deep-read alongside experiments).
- Created this working doc + alternative-use-case brainstorm + Phase 0–4 strawman plan.
- **Realization mid-session:** the three demos are not the same thing — they're three different points in a (what-gets-shipped × task × topology) design space. Each is the natural foundation for a different class of use cases. This is exactly why supervisor said keep open mind. Updated Section 4 with the mental model.
- **Next session plan:**
  1. Run **all three demos** in order — detection, segmentation, multi-sensor. Each one samples a different research starting point.
  2. Read V2X paper Section 1 + Section 3 in parallel.
  3. Sharpen the supervisor question from "which use case?" to "where do you want me positioning between perception-centric (multi-UE cooperative) and system-centric (multi-modal on one UE)?"

### 2026-05-19 — first demo runs
- **Detection demo: ran successfully at 1080p.** System got noticeably slow during the run — ask supervisor whether this is expected (likely culprits: 1080p × 5 FPN scales × full quantize-tile-flip-pickle-zlib chain per frame, plus CARLA in another process on the same box). Diagnostics to run if asked: try default resolution; watch `nvidia-shmi` + `htop` for GPU vs CPU saturation.
- **Segmentation demo: blocked.** Fails at import — `carla_split_inference_udp_data_collect.py` does not exist anywhere on this machine. Searched whole `workarea/` tree; genuinely missing. The segmentation script was updated (May 14 mtimes) to use a new `od_collect` module providing `TransportConfig`, `FeatureAutoencoder`, plus `QUANT_MODE_*`, `ENTROPY_CODER_*`, `AE_MODE_*` constants — none of which exist in the detection demo. The autoencoder pieces in particular need a trained checkpoint, so reconstruction isn't viable. **Action: ask Subhramoy for the missing module + any associated `.pt`/`.pth` checkpoints.**
- **Multi-sensor demo: not yet run.** Plan to run today/tomorrow as the third design-space sample while segmentation is blocked.

### 2026-05-19 — debugging system lag → FFmpeg/NVENC breakthrough

**Lag was three layered problems, not one:**

1. **Display side (receiver):** `autovideosink` + 200ms `rtpjitterbuffer` default + AnyDesk = perceived sluggishness. **Fixed** by `scan_receiver_v3.py` (appsink + `cv2.imshow` + `max-buffers=1 drop=true`). Pairs perfectly with sender_v1: receiver HUD shows 20-22 fps, no lag.

2. **Sender side (SI/TI in camera callback):** sender_v2's `compute_si_ti` ran inside the sync-mode CARLA callback. World.tick blocked waiting for ~90ms per frame → sim drifted. **Fixed** by `scan_sender_v2_threaded.py` (camera callback queues frame; worker thread does SI/TI). Reduced drops 44% → 19%, encoder fps 10.5 → 15.2, but encoder still couldn't hit 20 fps because **x265enc itself is CPU-bound**.

3. **Encoder side (the real bottleneck):** GStreamer's `x265enc` is software, multi-threaded but expensive. On the existing GStreamer 1.20.3 from 2022, the NVIDIA `nvcodec` plugin file (`libgstnvcodec.so`) is present but registers **0 features** because it predates Blackwell (RTX 5090). So *via GStreamer*, no hardware codec is available on this box.

**The breakthrough — FFmpeg/NVENC works on this RTX 5090 today:**
- `ffmpeg -c:v hevc_nvenc -preset p4 -tune ll` initializes cleanly
- Benchmark: 1280×720 @ 20 fps encodes at **21.5× real-time**
- Confirmed with `nvidia-smi dmon`: `enc%` lights up; `sm%` stays around 67-75% (CARLA rendering), unchanged
- PoC: [scan_sender_ffmpeg_poc.py](scan_sender_ffmpeg_poc.py) pairs with [scan_receiver_v3.py](scan_receiver_v3.py). Result: smooth video, sender CPU **240% → 50-60%** (~75% reduction), no perceived lag, no frame drops noticed in output
- **No installs needed.** FFmpeg 4.4.2 + `hevc_nvenc` + `hevc_cuvid` + `cuda` hwaccel are already on the box. Migration is purely a code change in our scripts, not a system change.

**Implications for the testbed direction:**
- Migrating sender/receiver scripts from GStreamer to FFmpeg is the path to GPU-accelerated codec on this hardware *today*. Upgrading GStreamer system-wide would be the alternative but needs IT and affects everyone.
- Receiver side is still software decode in v3. To get hardware codec end-to-end, we'd also write an FFmpeg-based receiver using `hevc_cuvid`. PoC results suggest single-stream is fine on CPU; multi-stream (multi-sensor) may need it.
- **Take to supervisor:** there's now a concrete, working demo to anchor the FFmpeg-migration proposal. Better than a hypothetical pitch.

**Multi-sensor refactor completed same day:**

Two new files in `abiodun/`:

- [carla_multisensor_udp_ffmpeg_sender_v2.py](carla_multisensor_udp_ffmpeg_sender_v2.py) — adds `FfmpegRtpH265CameraSender` class next to the existing `GstRtpH265CameraSender`. `--encoder ffmpeg|gst` flag (default ffmpeg) routes the 3 camera streams through either FFmpeg+NVENC or the original GStreamer+x265 path. Point sensors unchanged. Includes `-pkt_size 1200` to match GStreamer sender MTU, and `-rtpflags +skip_rtcp` to suppress RTCP packets that confuse the GStreamer depayloader.
- [carla_multisensor_udp_opencv_receiver_v2.py](carla_multisensor_udp_opencv_receiver_v2.py) — adds `OpencvRtpH265CameraReceiver` class. Pipeline ends in `appsink` (not `autovideosink`); no `rtpjitterbuffer`. Decoded frames stored under a lock by the GStreamer callback; main thread pulls them with `get_latest_frame()` and calls `cv2.imshow` + `cv2.waitKey(1)`. Point-sensor display (`GstBgrVideoRenderer`) left untouched — separate workstream if needed.

**End-state numbers (CARLA + 3 cameras @ 1280×720 + 10 NPCs + 10 pedestrians):**
- Sender CPU: **~50%** (was ~240% with x265enc)
- nvidia-smi `enc%` > 0 during steady state — NVENC engaged
- Encode: at 20 fps (sim rate matches camera rate)
- RTP/depayloader errors: **none** after `skip_rtcp` patch
- Video: smooth, close to local-feeling — but with AnyDesk-inherent ~100–250 ms display lag baked in

**Issues encountered along the way:**
- Initial attempt at 1080p + 30 NPCs + 30 pedestrians segfaulted CARLA (Unreal SIGSEGV). Resolved by ramping down to 720p + 10/10 NPCs. The crash boundary is independent of FFmpeg vs GStreamer; it's a CARLA stability ceiling under combined load on this hardware. Worth flagging to supervisor / Mateo as a constraint for any multi-sensor work.
- `rtpjitterbuffer` in the original receiver rejected ~constant fraction of FFmpeg-emitted RTP packets. Diagnosis confirmed by error source migrating from `gstrtpjitterbuffer.c(3309)` (pre-refactor) to `gstrtpbasedepayload.c(829)` (post-refactor receiver). Final fix: `-rtpflags +skip_rtcp` on sender (RTCP sender reports were arriving on the H.265 data port and failing depay validation).

**Remaining lag sources (all infrastructure, not code):**
1. Receiver-side software decode (3× `avdec_h265`). Modest CPU cost on this machine.
2. **AnyDesk capture latency** (~100–250 ms intrinsic, unavoidable while accessing the box remotely).
3. CARLA Unreal's own physics + render load.

The system itself is now smooth at the source. What's seen over AnyDesk is "smooth + AnyDesk lag." If perfect smoothness becomes a requirement (e.g., for a demo at the team's desk), physical access would resolve it. If we need to reduce receiver CPU further, the next refactor is FFmpeg + `hevc_cuvid` for hardware decode on the receive side.

**Take to supervisor:** updated proposal text above (Section 6 already drafted). The story now spans the full pipeline — sender, receiver, MTU, RTCP suppression — and has measured CPU + smoothness numbers at every stage. It's a complete migration proposal, not just a PoC.

**Refined contribution attribution (after A/B testing — corrected from earlier overstatement):**

| Configuration | Perceived lag | Notes |
|---|---|---|
| Original multi-sensor (gst sender + gst receiver) | severe + RTP errors | starting state |
| New receiver + `--encoder gst` (CPU x265) | ~30% laggy | receiver refactor is doing the heavy lifting |
| New receiver + `--encoder ffmpeg` (NVENC) | ~25-27% laggy | only ~3-5% additional smoothness over gst |

So the **smoothness improvement** is dominantly from the **receiver refactor** (eliminates rtpjitterbuffer, autovideosink, and the RTP validation errors). FFmpeg/NVENC at the sender adds only a small marginal smoothness gain at this scale.

**The real value of FFmpeg/NVENC is CPU savings**, not smoothness. Sender process CPU drops from ~240% (x265enc) to ~50% (push to subprocess pipe). That ~190% CPU headroom is what matters when AI workloads (split inference, RL agents, SI/TI at scale) get layered on later. For SCAN-AI extension work, that headroom is the real prize.

The remaining ~25% perceived lag is infrastructure (3× software H.265 decode on receiver + AnyDesk + CARLA's own sim load), not script logic. Code-side, there's nothing left to tune meaningfully at this scale.

**SI/TI sampling caveat in `scan_sender_v2_threaded.py`:** the threaded architecture means video stream is decoupled from SI/TI worker — video stays smooth even when the worker is overloaded. But the worker dropped ~43% of frames at full 1280x720 SI/TI compute on this hardware (1500+ drops in a ~175s run). For research output, this means **SI/TI is effectively sampled at ~11 fps instead of 20 fps** at full resolution. Two ways to fix if needed:
- Use [scan_sender_v2_threaded_lowres_siti.py](scan_sender_v2_threaded_lowres_siti.py) (320x180 SI/TI) — worker keeps up easily but absolute SI/TI values differ from full-res
- Reduce CARLA scene load (fewer NPCs / lower camera res) so the worker has more CPU headroom

Either is a conversation to have with supervisor about what SI/TI sampling rate the SCAN-AI extension actually needs.

### Next session candidates
1. Build FFmpeg/CUVID receiver for full hardware codec end-to-end (only if receiver CPU becomes a problem at scale).
2. Migrate point-sensor display (`GstBgrVideoRenderer`) from `autovideosink` to `cv2.imshow` (only if those windows become a lag bottleneck).
3. **Investigate the CARLA crash at high load** — under what NPC count / resolution combo does it stabilize? This affects any multi-sensor work.
4. Resume the SCAN-AI / CoDriving research direction proper (this whole session was infra debugging — important but tangential to the main research thread).

### 2026-05-19 (end of day) — supervisor sync, official verdict on the lag

Supervisor ran the same scripts (split inference detection, segmentation, multi-sensor streaming) on **his** machine. All smooth. Verdict: **it's a system-specific issue on this box**, not code. Don't sink more time into optimizing it. If the lag becomes blocking for real experiments, supervisor offered access to his machine.

**Don'ts** (per supervisor):
- Don't spend more time on lag debugging on this box
- Don't worry about achieving perfectly smooth visualization here
- Don't get blocked by visualization at all — *"you don't even need to see the streaming. Results can just be logged for offline analysis."*

**Do's** (per supervisor — the real work):
- **Digest the slides** (both decks — testbed implementation, split inferencing walkthrough)
- **Understand the codebases deeply** — split inference (detection + segmentation), multi-sensor streaming, the frameworks they sit in
- **Then** discuss the actual research problem statement, specifically:
  - **How will the model be trained to select which tensors are important to send over the network?**
  - **How does the network influence that selection — channel quality, available resources?**
  - **Latency-aware decisions**
- CARLA is just visualization. The frameworks underneath are what need digesting.

**What I've kept from today's work (still useful on supervisor's machine or any healthy box):**
- [scan_receiver_v3.py](scan_receiver_v3.py) — OpenCV display pattern
- [scan_sender_v2_threaded.py](scan_sender_v2_threaded.py) — threaded SI/TI worker, proves the decoupling pattern
- [scan_sender_v2_threaded_lowres_siti.py](scan_sender_v2_threaded_lowres_siti.py) — same with downsampled SI/TI
- [scan_sender_ffmpeg_poc.py](scan_sender_ffmpeg_poc.py) — FFmpeg/NVENC PoC
- [carla_multisensor_udp_ffmpeg_sender_v2.py](carla_multisensor_udp_ffmpeg_sender_v2.py) — multi-sensor FFmpeg/NVENC sender, `--encoder gst|ffmpeg` flag
- [carla_multisensor_udp_opencv_receiver_v2.py](carla_multisensor_udp_opencv_receiver_v2.py) — multi-sensor OpenCV display receiver

These are valid contributions; just not the priority right now.

### Next session plan
**REFOCUS on understanding.** Pick up where we were before the lag rabbit hole:
1. Walk through **`Split_inferencing_walkthrough.pptx`** (the feature-compression *theory* deck — we did the testbed *implementation* deck earlier)
2. Walk through **`carla_split_inference_udp_segmentation_demo.py`** now that the data_collect dependency is resolved
3. (Optional) Deep dive on the multi-sensor sender/receiver internals
4. Then start framing the actual research problem: tensor-selection, network-aware, latency-aware

### 2026-05-20 — research framing day
- Walked through `Split_inferencing_walkthrough.pptx` (the theory deck, all 13 slides).
- Discovered that `saliency_drop_masks()` in the segmentation demo **already implements feature-importance-based dropping** using L2 norm of channel activations. Research = upgrading the existing static gate to a dynamic, network-aware, scene-aware policy. Not building from scratch.
- Created two new docs to keep work organized:
  - **[research_direction.md](research_direction.md)** — clean, structured, MobiSys-paper-shaped. Phased plan, control knobs, open design questions, related reading, differentiation table.
  - **[brainstorm_log.md](brainstorm_log.md)** — raw idea capture, dated entries, running follow-up checklist.
- Going forward: keep `research_direction.md` clean; dump raw ideas into `brainstorm_log.md`; promote ideas to the main doc as they mature.

### Project documents — index
- [research_direction.md](research_direction.md) — *the* paper-shaped doc. Read this when you need to know where the research is going.
- [brainstorm_log.md](brainstorm_log.md) — raw idea capture. Read this when you need to remember what you were thinking.
- This file (`notes.md`) — session-by-session running log of what was done.

### 2026-05-21 (end of day) — V2Xverse install hit a wall; supervisor redirected to OAI

**V2Xverse install status:** blocked. Cumulative 4 patches landed (PyTorch 2.x nightly, MKL pin, spconv 2.x, carla 0.9.16 client). The final blocker: no Python 3.10 wheel exists for CARLA 0.9.10.1, and pip-installable newer clients (0.9.15, 0.9.16) fail `client.get_world()` with `rpc::rpc_error in function get_sensor_token` against the 0.9.10.1 server. Full details + escape paths in [installation.md](installation.md) §12.3.

**Supervisor's direction:**
- He'll look into providing access to another machine (pre-Blackwell GPU, V2Xverse-compatible). TBD when.
- **In the meantime:** install OAI in `/abiodun/` (full control), then run **segmentation, OD, and multi-sensor demos over the OAI 5G network** instead of localhost UDP. Goal: have the pipeline working over real 5G transport before the V2Xverse arc resumes.

**Why this is good direction:**
- We need OAI anyway for the network-aware part of the research. This isn't a detour.
- The split-inference + multi-sensor scripts already exist and work; only the UDP destinations need to point at OAI-allocated IPs instead of 127.0.0.1.
- Produces real-network measurements (latency, payload, throughput under 5G channel) — closer to research-grade data than the localhost runs.

### Next session plan (2026-05-22)
1. **Run `/compact`** at the start to free up conversation room before OAI install fills it.
2. **Run the OAI discovery checklist** (in the chat history at end of 2026-05-21 — Docker containers, existing OAI installs on box, etc.) to see if any existing OAI is reusable.
3. **Ask Subhramoy / supervisor:**
   - "Is there an existing OAI on this box I should reuse, or should I install fresh?" (kickoff notes mention a dockerized OAI + CARLA setup that already worked.)
   - "What IP scheme do OAI UEs get, so I know what to point the demo scripts at?"
4. **Install OAI in `/abiodun/`** if no existing one is reusable. Use OAI's official Docker setup.
5. **Migrate first demo** (probably `scan_sender_v1.py` + `scan_receiver_v3.py` — the smoothest pair we built) to talk over OAI instead of localhost. Single IP change.
6. Then segmentation, then multi-sensor.

### (next session) — TBD
