# OAI single-UE A/B — compression over real 5G (2026-07-13)

Single-UE OAI (rfsim, 106 PRB / 40 MHz / band78 / eMBB, **no channel impairment**), Town10HD_Opt,
pole @ TL-14, 300 frames each. Same fusion pipeline; only the transmitted representation changes.
Transport metrics = live (this run); **accuracy = offline per-model sweep** (model-determined, identical operating point).
Idle-link ping RTT = **3.85 ms** (so loaded transport is queueing, not propagation).

| config | payload | frag/frame | RTT mean | RTT p95 | transport | delivery | mIoU | ped-rec | loc |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **no-AE · u8 · ROI0** (baseline) | 1141 KB | 19.5 | 209 ms | 271 ms | 200 ms | **75.0%** | 0.840 | 0.855 | 0.95 |
| **AE-128 · u8 · ROI0** | 346 KB | 6.0 | 118 ms | 174 ms | 107 ms | **99.3%** | 0.819 | 0.883 | 0.87 |
| **AE-128 · u4 · ROI0** | 142 KB | 3.0 | **77 ms** | 139 ms | 67 ms | **99.0%** | 0.819 | 0.887 | 0.88 |
| *loopback ref (no-AE u8)* | 1160 KB | — | *53 ms* | — | *47 ms* | *100%* | 0.840 | 0.855 | 0.95 |

## Headline
- **Payload 1141 → 142 KB (8×↓)** drives **RTT 209 → 77 ms (2.7×↓)** and **delivery 75% → 99%**.
- AE-128 u4 over real 5G (**77 ms, 99%**) is within ~24 ms of the *loopback* baseline (53 ms) — the ~4–5× OAI penalty is essentially **erased by compression**.
- Even at matched quant (u8), the AE alone: 1141→346 KB, RTT −44%, delivery 75%→99%.
- Cost: ~2% mIoU (0.840→0.819) but **better** detection (ped-rec 0.855→0.887, loc 0.95→0.87 m).

## Accuracy vs delivery — TWO SEPARATE axes (do not multiply them)
Accuracy and the channel are orthogonal, and it matters to keep them separate:
- **Per-frame accuracy** is set by the **compression config** (quant/AE/ROI). A *delivered* frame decodes to
  exactly the model's output — **transport latency does not change it**. So delivered-frame accuracy is
  **identical on loopback and OAI**. (A frame counts as delivered only if all ~20 UDP fragments reassemble;
  partial frames are dropped, so there is no "corrupted-but-delivered" case.)
- **Delivery / availability** is set by the **network**: what fraction of ticks get *any* fresh result.

| config | mIoU | ped-rec | loc | delivery (OAI) |
|---|--:|--:|--:|--:|
| no-AE u8 | 0.840 | 0.855 | 0.95 | **75.0%** |
| AE-128 u8 | 0.819 | 0.883 | 0.87 | **99.3%** |
| AE-128 u4 | 0.819 | 0.887 | 0.88 | **99.0%** |
| no-AE u8 (loopback) | 0.840 | 0.855 | 0.95 | 100% |

**Key point:** the network does not corrupt accuracy — it costs **availability**. At the no-AE baseline, 25%
of ticks have **no fresh perception** (a continuity/safety gap); AE configs restore ~99%. Do **not** report
"accuracy x delivery" as an accuracy number — that conflates the two axes and wrongly implies the model degraded.
**Staleness** (a late-but-delivered frame is temporally misaligned -> position error) is the *only* true
latency->accuracy link and is not captured by per-frame mIoU; measuring it is a Month-3 item. Plot:
`plots/oai_accuracy_delivery.pdf`.

## Why
Transport is **fragmentation/queueing-bound**, not propagation (idle RTT 3.85 ms). The 1.14 MB baseline is ~20 UDP
fragments/frame saturating the ~9 Mbps uplink → queue buildup (200 ms) + 25% datagram loss. Compression cuts the
frame to 3 fragments (142 KB), which fits the uplink grant → latency collapses toward the idle floor and loss ~vanishes.
Two independent levers hit the same bottleneck: **payload (compression)** — shown here — and **uplink capacity
(OAI config: TDD UL/DL ratio, PRBs)** — next.
