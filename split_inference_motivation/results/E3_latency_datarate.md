# E3 — Per-hop end-to-end latency + uplink data rate (architectures A / B / C)

**Date:** 2026-07-27 · **Raw:** `E3_payloads.json`, `E3_latency_datarate.json`, `E3_run.log`
**Scripts:** `../e3_payloads.py` (payload measurement), `../e3_latency_datarate.py` (assembly)

## Provenance — what is new vs reused

| Quantity | Source |
|---|---|
| Payloads for A / B / C | **Measured in this study**, 25 real test-split frames (`e3_payloads.py`) |
| Car / edge compute | **Measured in this study** (E1) |
| OAI per-hop latency, delivery | **Reused, not re-measured** — prior work, cited inline below |

Cited prior sources:
- **[1]** `abiodun/downlink_latency_fps/OAI_TRANSPORT_BOTTLENECK_DISCUSSION.md` — live CARLA frontend, 10 FPS target, 1300 frames, corrected drivable route.
- **[2]** `abiodun/oai_layer_latency/README.md` — Phase-2b instrumented CARLA run, 918 409 matched SDUs; per-packet UE PDCP-ingress → gNB PDCP-deliver.
- **[3]** `abiodun/rl_agent/PERMODEL_KNOB_MATRIX_ZSTD.md` — ideal-loopback compute/payload anchors.

## 1. Measured uplink payload per frame (25 real frames)

| Architecture | payload | mean | p50 | range |
|---|---|---|---|---|
| **A** full-local | detections JSON | **2.27 KB** | 2.22 | 1.91–2.84 |
| **A** full-local | detections JSON + zstd | 0.58 KB | 0.58 | 0.52–0.67 |
| **B** full-offload | RGB JPEG q92 | 347.16 KB | 353.33 | 300.96–357.54 |
| **B** full-offload | RGB JPEG q75 | 192.10 KB | 195.81 | 166.33–198.58 |
| **B** full-offload | RGB PNG (lossless) | 1411.74 KB | 1432.55 | 1260.72–1443.01 |
| **B** full-offload | radar raster fp16+zstd | 35.98 KB | 27.37 | 25.47–66.76 |
| **B** full-offload | **JPEG q92 + radar** | **383.13 KB** | 381.96 | 366.20–417.84 |
| **C** split | features fp32 (raw) | 5670.00 KB | — | constant |
| **C** split | features fp16 (uncompressed) | 2835.00 KB | — | constant |
| **C** split | features u8 (quantized) | 1425.34 KB | — | constant |
| **C** split | **features u8 + zstd** | **1045.54 KB** | 1045.96 | 1037.38–1054.89 |
| **C′** split + AE-128 u6 ROI0.5 | features, AE-compressed | **152.70 KB** [1] | — | — |

**Cross-validation (this is why the payload numbers can be trusted):** measured independently here,
the no-AE u8+zstd payload lands at 1045.54 KB against the deployed references of **1050.3 KB** [3] and
**1055.2 KB** live over OAI [1] — agreement within 1 %. The fp16 uncompressed figure reproduces [3]'s
`uncompressed_fp16` baseline of **2835.0 KB exactly**.

> **Correction to the plan's estimates.** The plan estimated A's result payload at ~8–12 KB; measured it is
> **2.27 KB** raw (2.2–2.5 KB is also what the live OAI downlink carries [1]) — roughly 4× smaller.
> The plan estimated B at ~100–200 KB; that holds only at **JPEG q75 (192 KB)**, and only for RGB — adding
> the radar raster the edge needs to run the fusion model brings q75 to **228 KB** and q92 to **383 KB**.

## 2. Uplink data rate at the 10 FPS target

Measured OAI uplink drain under bursty CARLA load: **10.9 Mbps** (UL MCS ~4) [2].

| Architecture | KB/frame | Mbit/frame | **Mbps @ 10 FPS** | sustainable FPS at 10.9 Mbps |
|---|---|---|---|---|
| **A** full-local | 2.27 | 0.018 | **0.18** | ~616 |
| **B** full-offload (q75+radar) | 228.08 | 1.782 | **13.92** | 6.1 |
| **B** full-offload (q92+radar) | 383.13 | 2.993 | **29.93** | 3.6 |
| **C** split, no-AE | 1045.54 | 8.168 | **81.68** | **1.33** |
| **C′** split, AE-128 u6 ROI0.5 | 152.70 | 1.193 | **11.93** | 9.1 |

Consistency check: the 1.33 FPS computed here for no-AE matches [2]'s independently stated ~1.45 FPS
effective uplink throughput for the same configuration.

> ### Guardrail 1 confirmed — uplink does NOT favour split
> **Architecture A has by far the lowest uplink demand: 0.18 Mbps, ~450× less than split's 81.68 Mbps.**
> Split ships the *most* data of any architecture. Do not claim split reduces bandwidth. It does not.
> Even the AE-compressed variant (11.93 Mbps) is ~66× architecture A's demand.

**Important caveat on the "sustainable FPS" column.** The 10.9 Mbps drain was measured specifically under
the *no-AE 1 MB burst* regime, where the gNB pins UL MCS ≈ 4. It is **not a fixed link capacity**: with the
smaller 3-chunk AE payload, link adaptation behaves better and the measured live result was **99.8 % delivery
at 64.2 ms RTT** [1] — i.e. C′ works in practice despite 11.93 Mbps nominally exceeding 10.9. Treat that column
as an order-of-magnitude feasibility indicator, not a hard verdict. The no-AE row (7.5× oversubscribed) is
far enough over that its conclusion is safe.

## 3. Per-hop end-to-end latency

Car compute = E1 CPU @ 8 threads (vehicle-plausible operating point; GPU p50 in parentheses).
Uplink / edge / downlink from the cited live OAI runs.

| Hop | **A** full-local | **B** full-offload | **C** split, no-AE | **C′** split, AE-128 |
|---|---|---|---|---|
| Car compute | 32.84 ms (1.85) | ~0 (JPEG encode only) | **14.65 ms** (1.30) | **14.65 ms** (1.30) |
| Uplink (UE→gNB→UPF→edge) | 4.6 ms [2] | *not measured* | **151.1 ms** [1] | **52.6 ms** [1] |
| Edge compute | 0 (peer fuses lists) | 1.85 ms (GPU) | 6.9 ms [1] | 7.5 ms [1] |
| Downlink (edge→UE) | 4.6 ms [2] | 4.6 ms [2] | 3.0 ms [1] | 3.2 ms [1] |
| **Measured RTT p50** | — | — | **162.2 ms** [1] | **64.2 ms** [1] |
| **Measured capture→result** | — | — | **188.0 ms** [1] | **86.5 ms** [1] |
| **Delivery rate** | — | — | **83.6 %** [1] | **99.8 %** [1] |

Reference upper bound with the radio removed: ideal loopback, no-AE = 46.1 ms capture→result, 100 % delivery [1].
So **of split's 188 ms, roughly 142 ms is the radio**, not compute.

**Two latencies matter for architecture A, and conflating them would overstate its cost:**
- **Own-perception latency: 32.84 ms.** A vehicle running the full model locally can act on its *own*
  perception immediately — no network in the loop at all.
- **Cooperative-sharing latency: ~42 ms** (32.84 local + 4.6 up + 4.6 down) for its detections to reach a peer.

Architecture A wins the latency axis outright, on both readings.

## 4. Conclusion — latency is split's cost, stated plainly

| | car compute | uplink Mbps | E2E to a shared/cooperative result | delivery |
|---|---|---|---|---|
| **A** full-local | 32.84 ms (highest) | **0.18 (lowest)** | **~42 ms (lowest)** | — |
| **B** full-offload | ~0 (lowest) | 13.9–29.9 | not measured | — |
| **C** split no-AE | 14.65 ms | 81.68 (highest) | 188.0 ms (highest) | 83.6 % |
| **C′** split AE-128 | 14.65 ms | 11.93 | 86.5 ms | 99.8 % |

**Architecture A is the best choice on both network axes — lowest uplink demand and lowest latency — and split
is the worst on both.** That is the honest position and the paper should state it. Split's case rests entirely
on the other three columns: it roughly halves on-car compute (E1/E2), it enables intermediate feature fusion
that late fusion cannot do (E4), and it never puts raw pixels on the air (E5).

A secondary but practically important result: **feature compression is not optional for split.** At no-AE
1 MB/frame the architecture cannot reach the 10 FPS target over this link at all (1.33 FPS sustainable,
83.6 % delivery, 188 ms). With AE-128/uint6/ROI0.5 it becomes viable (9.1 FPS sustainable, 99.8 % delivery,
86.5 ms) — a 2.2× end-to-end latency reduction for a 6.8× payload reduction.

## Gaps / not done
- **Architecture B was never run over OAI**, so its uplink latency is genuinely unmeasured — the table says
  *not measured* rather than interpolating a number. Its data rate (13.9–29.9 Mbps) sits between C′ and C,
  so its latency plausibly does too, but that is an inference, not a measurement. Closing it would need one
  OAI run shipping JPEG frames.
- Car-compute figures are host-CPU (Core Ultra 9 285K @ 8 threads), not a vehicle SoC — see E1's caveat.
