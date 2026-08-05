# Channel-condition sweep — RESULTS (uplink-only, SINR, 106PRB)

**Date:** 2026-08-04. **Batches:** `chsweep_full_{p1u8,p2u4,p3ae}` × `{clear,mild,mid15,strong}`.
**Pipeline:** uplink-only (`run_track2_sinr_uplink_only_ladder.sh` → `carla_fusion_staleness_scenario_uplink_only.py`,
edge publishes to spatial map, **no downlink return**). **MCS:** `SCENESENSE_MCS_POLICY=sinr` (retx ≈ 0 across
the grid). **12 cells, serial, on a freshly-restarted (healthy) CARLA**, long 120 s window (~1,200 frames/cell).
Every cell passed the health gate (eff 5.9–7.5 fps, camera_wait 29–41 ms). One cell (p3ae/mid15) hit a
transient back-half-container failure and was re-run clean. This run supersedes all earlier drafts (the
degraded-CARLA grid and the 2×2 draft, both deleted).

Plots: `plots/fig1_latency_knee`, `plots/fig2_delivery_heatmap`, `plots/fig3_payload_budget`,
`plots/fig4_sweep_summary_bars`, `plots/fig5_latency_breakdown_by_payload_channel` (PNG + PDF).
Raw: `combined_surface.csv`, per-payload `results/chsweep_full_p*_.csv`.

## The surface (offered ~6–8 fps)

`sched UL` = MAC-scheduled/served uplink rate; `app off.` = application on-wire offered rate over the
first-send→last-send window (payload × send fps). On collapse cells **app-offered ≫ served** — that gap is
the congestion (the excess piles into the UE buffer → BSR pins at the ~48 MiB ceiling).

| payload | SNR dB | MCS | delivery % | capture→map p50 | app off. Mbps | sched UL Mbps | BSR p95 | retx |
|---|---|---|---|---|---|---|---|---|
| **1 MB**   | 50.3 (clear) | 28 | **97.5** | 138 ms | 50.2 | 36.7 | 1.0 MiB | 0 |
| **1 MB**   | 19.5 (mild)  | 24 | **22.2** | 6.1 s  | 63.6 | 27.8 | 47.7 MiB | 0 |
| **1 MB**   | 15.6 (mid)   | 19 | **10.7** | 10.7 s | 62.4 | 19.7 | 47.7 MiB | 0 |
| **1 MB**   | 8.2 (strong) | 9  | **4.6**  | 15.5 s | 68.5 | 9.2  | 47.7 MiB | 0 |
| **400 KB** | 50.3 | 28 | **100** | 112 ms | 20.0 | 15.5 | 0.4 MiB | 0 |
| **400 KB** | 19.5 | 24 | **100** | 209 ms | 21.7 | 15.6 | 0.4 MiB | 0 |
| **400 KB** | 15.6 | 19 | **100** | 251 ms | 22.3 | 15.2 | 3.0 MiB | 0 |
| **400 KB** | 8.2  | 9  | **31.5**| 11.6 s | 23.4 | 10.4 | 47.7 MiB | 0 |
| **90 KB**  | 50.4 | 28 | **100** | 94 ms  | 4.8 | 4.8 | 0.1 MiB | 0 |
| **90 KB**  | 19.6 | 24 | **99.7**| 130 ms | 5.2 | 4.3 | 0.1 MiB | 0 |
| **90 KB**  | 15.7 | 19 | **100** | 146 ms | 5.3 | 4.1 | 0.1 MiB | 0 |
| **90 KB**  | 8.2  | 9  | **100** | 175 ms | 5.2 | 3.8 | 0.1 MiB | 0 |

## The knee — sharp, payload-ordered (fig 1 & 2)
- **1 MB** survives **only the clear channel** (97.5%). Collapses at 19.5 dB and below (22 % → 4.6 %, 6–15 s).
- **400 KB** holds down to **15.6 dB** (100 %, ≤251 ms); collapses at 8.2 dB (31.5 %, 11.6 s).
- **90 KB (seg-safe AE-32)** is **100 % at every rung**, ≤175 ms — robust to 8.2 dB with large margin.

Collapse mechanism is pure congestion: offered rate > channel capacity → UE buffer saturates (**BSR pins at
the ~48 MiB ceiling**), latency runs to seconds, delivery cliffs — all at **retx ≈ 0** (SINR keeps essentially
all transmissions first-try; the loss is queueing, not radio errors).

## Agent rule — `payload_budget(SNR) = capacity(SNR) / fps` (fig 3)
Capacity rises with SNR (delivered ceiling in this grid: strong ≈10, mid ≈20, mild ≈28, clear ≈37 Mbps). Projected
to the agent's target **10 fps**, the affordable payload is ~127 KB (strong) / 241 KB (mid) / 339 KB (mild) /
448 KB (clear). So:
- **90 KB seg-safe floor is deliverable at the tested/projected 10 fps target across every SNR rung** — the
  robust invariant for this sweep.
- At ~8 dB the 10 fps budget forces the floor; even 400 KB won't fit. **ROI-escalation region = only here**,
  and only if 90 KB still can't meet freshness — not seen (90 KB fit everywhere).
- accuracy is transport-invariant (retx ≈ 0, lossless zstd, AE-32 accuracy from the offline knob matrix) —
  compose with staleness from these latencies; do not re-measure.

## Honest caveats
- **Offered fps was ~6–8 (live-front, CARLA-render limited), not the target 10.** The empirical knee above is at
  ~6–8 fps; the fig-3 budget projects to 10 fps via `capacity/fps`. Capacity is estimated from delivered
  ceilings (±~30 %); a **shaped-burst run at a fixed 10 fps** (no CARLA in the loop) would pin the absolute
  knee precisely — recommended follow-up, not blocking for the agent.
- **SNR rungs cluster low** (50 / 19.5 / 15.6 / 8.2); thin between 20–50 dB — add tuned mid rungs later.
- **clear 1 MB = 97.5 %** (≈30/1200 frames lost to UDP fragmentation of the large payload, not congestion —
  BSR 1 MiB, 138 ms). Smaller payloads fragment less → 100 %.
