# Month-2 summary — static split-inference knob characterization (2026-07-09)

**North Star:** learn a network-aware split-inference control policy that cuts payload/latency while
preserving task utility. **Month-2 DoD:** offline controller harness + static knob sweeps.

## Status: static-knob DoD ✅ done (controller harness = remaining M2 item)

### Delivered
1. **M′ — one ROI-robust fusion model** (drop-aware fine-tune from the 200k model; objectness-guided
   feature-dropout q~U(0,0.8), rank-based; maximin clean/robust selection). GATE A vs the 200k baseline:
   seg + localization preserved (mIoU 0.841, veh-IoU 0.933, loc 1.21m = baseline; person-loc/dim BEAT
   baseline); object recall within ~1.4% (residual cost of robustness, covered by the RL guardrail).
2. **COMPLETE_KNOB_MATRIX.md** — 19 action profiles over {quantization, entropy coder, ROI drop, AE
   bottleneck} × {mIoU, IoU, recall, localization, **payload bytes**, **front latency, RTT, delivery**}.
   Payload measured offline (entropy-coded); latency/reliability measured on the CARLA loopback transport.
3. **Reusable infra** — drop-aware trainer, task-aware feature-AE trainer, offline eval with quant/ROI/AE
   + payload, loopback sweep runner, aggregators, GATE-A checker. All resumable + logged.

### Key findings (these motivate the RL agent)
- **No single static config wins.** High accuracy (clean/u8/u6) needs ~1MB payloads that deliver only
  ~11–32% over the loopback UDP; reliable delivery (100%) needs ≤~400KB.
- **Quantization is cheap:** uint6 ≈ lossless (mIoU 0.841, recall 0.790, loc 1.22m) at 51% payload;
  uint4 near-lossless (slight recall cost 0.790→0.755) at 25% payload + **100% delivery** — the best
  static all-rounder.
- **Entropy coder is free and zstd ≥ zlib** (same accuracy, smaller payload, lower front latency).
- **ROI drop trades SEG for payload while preserving detection:** at q=0.7, payload 27%, recall 0.778 /
  loc 1.30m preserved, but mIoU 0.779 (drops low-objectness background). Graceful, importance-ranked.
- **Payload→reliability cliff (~700KB):** below → 100% delivery; above → 10–30%. Strongest single
  argument for compression under load.
- ⚠ **AE (as trained) preserves segmentation but COLLAPSES object detection** (mIoU 0.83 / veh 0.93 kept,
  but recall 0.83→0.05–0.28, loc 1.2→2.5–3.0m). The object head is far more sensitive to channel-
  bottleneck compression than seg. Needs a follow-up (upweight object/heatmap distill terms, gentler
  bottleneck, or object-feature-preserving AE) before AE is a usable action. **Flagged, not yet resolved.**

### Remaining (Month-2)
- **Offline controller harness (B):** baselines + LinUCB contextual bandit over the matrix as the action-
  cost table; guardrails (pedestrian-recall / mIoU floor, vulnerable-object clamp).
- Optional: fix the AE so it's a usable action (currently only quant/ROI are viable knobs).

## [ENG] vs [RES] split
- **[ENG]** drop-aware trainer + AE trainer + eval/payload instrumentation + loopback sweep + aggregators +
  orchestration/CARLA automation. (Done.)
- **[RES]** knob-cost relationships + the no-single-winner / payload-reliability-cliff findings + the AE
  detection-collapse result + deriving the RL state/action/reward from the matrix. (Matrix done; harness next.)

## Next (Month-3 / tomorrow): network-state phase
Replace the loopback transport column with **OAI + Sionna ray-traced channel** (OpenStreetMap geometry):
real latency/loss under varying channel conditions → the true reward's network term → train + evaluate the
policy against static baselines under channel stress.
