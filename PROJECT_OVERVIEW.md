# Project overview — cooperative-perception split-inference RL controller

**Read this first.** It's the sequential story of how we got here, so you don't have to reverse-engineer it
from the folder (which holds many similar experiments/runs). For terse pointers + conventions see `CLAUDE.md`;
for the locked agent design see `rl_agent/AGENT_CONSTRAINTS.md §9`; to start the current work see
`rl_agent/POLICY_KICKOFF.md`.

**What this is.** Abiodun's research thread (IDCC × NEU): a UE-side, **network-aware split-inference RL
controller** for RGB+radar fusion cooperative perception. A car runs a front backbone, compresses features,
sends them over a real 5G uplink (OAI) to an edge that fuses them and publishes a shared spatial map. The
controller decides, per frame, *how much to compress / what FPS / whether to send* so the map stays fresh and
accurate without congesting the uplink.

> ⚠️ This codebase is **shared** with a separate thread (Mateo's parking-spot/NLM demo). Not everything here
> is this project. The dirs that matter for THIS work are listed under "Navigating the folder" below.

## How we got here (sequential)

**Phase 0 — Foundation & first 5G demo (May 2026).** Built on SCAN-AI (single-UE) + V2X CoDriving. First
working stack: CARLA → H.265/RTP → 5G (OAI) → receiver container. (`project_first_demo_over_oai` notes.)

**Phase 1 — The cooperative-perception model M′.** A standalone detection head was an architectural dead-end
(F1 stuck ~0.35); two-view triangulation worked (~1.40 m) → the productive direction is **feature-sharing +
edge fusion (M′)**. Built a drop-aware 2-stage ROI-robust fine-tune, a feature-AE compressor, and a GATE-A
acceptance check. Model is **range-gated to 40 m** (`max_gt_distance_m=40` — beyond that, localization is
unreliable and was never optimized). A GT-convention bug (origin vs bbox-center) once inflated live loc error
to ~2–3 m; fixed 2026-07-15, after which **live ≈ offline** and the offline knob matrix became the accuracy
anchor. Localization floor **ε ≈ 1.1 m** (model-limited).

**Phase 2 — Compression & the knob matrix.** Levers: quantization (u8→u4), an **AE bottleneck** (the main
accuracy↔bytes dial), **ROI drop**, and a **zstd** entropy codec (lossless; replaced zlib — faster + better
delivery). Compression erases the ~5× OAI transport penalty. Result: `rl_agent/PERMODEL_KNOB_MATRIX_ZSTD.md`
— accuracy↔knob↔payload, and it is **transport-invariant** (lossless codec → holds byte-for-byte over OAI).
Key refinement (2026-07-31): once the map carries **segmentation**, ROI-drop destroys it (veh IoU 0.92→0.11),
so the seg-safe operating point is **`ae32/u4/ROI0 ≈ 90 KB` and is density-invariant** — density was dropped
as a control variable. (`density_knob/DENSITY_KNOB_RESULTS.md`.)

**Phase 3 — The staleness / FPS requirement.** Localization error is bounded by a **master inequality**
`v·total_staleness ≤ √(ε² − floor²)` (floor ≈ 1.1 m); fast objects blow up under latency (e.g. 32 mph @
267 ms → 4.4 m) unless FPS is high enough. Corrected the architecture to **uplink-only** (car → features →
edge → edge publishes the map; **no downlink detection return**), and fixed a radar-rasterizer frontend
bottleneck (139→33 ms). Capture→map budget measured at 67–93 ms p50 offline. (`staleness/STALENESS_RESULTS.md`,
`uplink_only_spatial_map_pipeline/`.)

**Phase 4 — OAI transport characterization.** A config sweep found TDD/5QI/PRB levers were near-no-ops in a
clean single-UE channel. The real uplink bottleneck is **UE RLC queue-wait driven by the QPSK MCS cap**
(root-caused in `gNB_scheduler_ulsch.c` via a per-layer T-tracer). Fix: a **SINR-driven UL MCS policy**
(`SCENESENSE_MCS_POLICY=sinr`, Track 2, 2026-08-03) that follows the injected SNR, keeps HARQ on, and gives
~0 retransmissions — and it removes a clean-channel artifact where reactive BLER-OLLA collapsed MCS to ~7.
(`oai_mcs_policy_track2/`, `oai_layer_latency/`.)

**Phase 5 — RL design locked (2026-07-31).** State / action / reward synthesis frozen in
`AGENT_CONSTRAINTS.md §9`: speed sets the freshness budget; channel sets the affordable payload; knobs spend
payload for accuracy; scene-emptiness is a send-gate; ROI is a last-resort escalation.

**Phase 6 — The channel sweep (2026-08-04) — the last missing piece.** Measured the transport function
`delivery/latency(payload × SNR)` over OAI, uplink-only, SINR. Clean 12-cell grid → a sharp, payload-ordered
**congestion knee**: **1 MB survives only a clear channel; 400 KB holds to 15.6 dB; the 90 KB seg-safe floor
delivers 100 % at every rung.** Collapse is pure congestion (BSR pins ~48 MiB), retx ≈ 0. Gives the agent's
control law **`payload_budget(SNR) = capacity(SNR) / fps`**. (`channel_condition_sweep/CHANNEL_SWEEP_RESULTS.md`
+ `combined_surface.csv` + `plots/`.)

**Phase 7 — NOW: policy formulation.** Build a surrogate env from the three measured tables → bandit baseline
→ constrained RL → live validation. **Start at `rl_agent/POLICY_KICKOFF.md`.**

## Dead-ends / superseded (do NOT revisit or be confused by)
- Standalone **detection head** — dead end (F1 ~0.35). Feature-sharing + fusion is the path.
- **Density-adaptive ROI** knob — an artifact of scoring detection only; superseded by the seg-safe,
  density-invariant 90 KB floor.
- **Closed-loop downlink pipeline** (`downlink_latency_fps/`, `oai_mcs_policy_track2/run_awgn_106prb_ladder.sh`)
  — self-throttles; **never** use it to characterize the channel knee. Uplink-only only.
- **zlib** codec, **273PRB** config — retired (now zstd, 106PRB).
- Any `chsweep_clean_*` / degraded-CARLA grid or 2×2 draft — **deleted** (a CARLA-render artifact ran them at
  2.5 fps). The authoritative sweep is `chsweep_full_*` → `CHANNEL_SWEEP_RESULTS.md`.

## Navigating the folder (what matters for THIS project)
- `PROJECT_OVERVIEW.md` (this), `CLAUDE.md` — orientation.
- `rl_agent/` — the agent: `AGENT_CONSTRAINTS.md` (§9 locked design), `POLICY_KICKOFF.md` (start here),
  `PERMODEL_KNOB_MATRIX_ZSTD.md`, `density_knob/`, `feature_ae/` (AE checkpoints).
- `channel_condition_sweep/` — the transport surface: `CHANNEL_SWEEP_RESULTS.md`, `combined_surface.csv`,
  `plots/`, `CHANNEL_SWEEP_PLAN.md` (guardrails).
- `staleness/` — `STALENESS_RESULTS.md`, `uplink_only_latency_budget/`.
- `uplink_only_spatial_map_pipeline/` — the uplink-only runners + edge back-half (the agent's real data path).
- `oai_mcs_policy_track2/`, `oai_layer_latency/`, `OAI/` — 5G stack + MCS policy + T-tracer (only for live runs).
- `experiments/`, `metrics_logs/` — historical runs; mostly not needed. The current model checkpoint is
  `experiments/ae_integrated_20260710/noae_baseline/checkpoints/mprime_joint_noae/best.pt`.

## Conventions (from CLAUDE.md — don't violate)
Work only in `abiodun/`; GT = actor origin offline; don't export `PYTHONPATH` for CARLA clients; rebuild both
softmodems after any `T_messages.txt` edit; policy work is **table-driven — no OAI/CARLA needed to start**.
