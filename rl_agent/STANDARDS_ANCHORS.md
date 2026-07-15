# Standards anchors — latency / FPS / accuracy for V2X cooperative perception & tele-op

Deep-research lookup (2026-07-14, adversarially verified, primary sources). Anchors for the SceneSense
requirement (see `REQUIREMENTS_AND_RL_DESIGN.md`). Confidence noted per row.

## Summary table (by use case)
| Use case | E2E latency | Reliability | Data rate / notes | Source |
|---|--:|--:|---|---|
| Extended Sensors (sensor sharing), **low automation** | **100 ms** | 99% | 1600 B, ~10 msg/s, 1000 m | 3GPP TS 22.186 T5.4-1 R.5.4-001 (high) |
| Extended Sensors, high automation | 3 ms | 99.999% | 50 Mbps, 200 m | 3GPP 22.186 R.5.4-003 (high) |
| Extended Sensors, imminent collision | 10 ms | 99.99% | 1000 Mbps, 50 m | 3GPP 22.186 R.5.4-006 (high) |
| **Advanced Driving incl. cooperative perception**, higher auto | **100 ms** | — | 53 Mbps UE-UE / 50 UE-RSU | 3GPP 22.186 T5.3-1 R.5.3-003/005 + NOTE 1 (med, 2-1) |
| Emergency trajectory alignment | 3 ms | 99.999% | — | 3GPP 22.186 R.5.3-006 (high) |
| Remote/Tele-op Driving (3GPP target) | 5 ms | 99.999% | UL 25 / DL 1 Mbps, ≤250 km/h | 3GPP 22.186 T5.5-1 R.5.5-002 (high) |
| **Tele-op Driving (practical)** | **~120 ms** (100 UL + 20 DL) | — | ~25 Mbps UL; net-level 40-45 ms UL | 5GAA ToD WP / 5G-MOBIX (high) |
| Cooperative Sensing (survey) | 3 ms–1 s (event-dep) | >95% | 5–25000 kbps | Boban et al. IEEE 2017 (high) |
| CPM update (ETSI CPS) | check ≤10 Hz (T_GenCpm 100 ms–1 s) | event-triggered | emit if Δpos>4 m, Δspeed>0.5 m/s, or 1 s | ETSI TR 103 562 / TS 103 324 (high) |
| AD perception frame rate (de-facto) | — | — | **~10 Hz** (Waymo, KITTI); nuScenes cam 12 / lidar 20 / radar 13 Hz; object-update 2 Hz → interp 10–20 Hz | datasets (high) |
| Tele-op camera FPS | — | — | 30–60 fps (5G-MOBIX 720p/60; 30 fps supervision) | 5G-MOBIX (high) |
| Positional accuracy | — | — | **lane-level ~0.5 m** de-facto; ETSI 4 m/0.5 m·s⁻¹/1 s = age-of-information triggers | ETSI / de-facto (LOW conf) |

## Anchors we adopt (cooperative perception → shared spatial map, low/mid automation)
- **Latency budget Y ≈ 100 ms E2E** (3GPP Advanced-Driving/Extended-Sensors low-auto; 5GAA ToD 100 ms uplink).
  Tightens to **3–10 ms** for high-automation / imminent-collision (a different regime — sidelink/direct,
  not our edge-map use case).
- **Reliability ≈ 99%** (delivery).
- **Update rate ≈ 10 Hz** (de-facto AD perception; ETSI CPM check rate). Raw capture can be 10–20 Hz; object
  update as low as 2 Hz interpolated. (Tele-op *video* is 30–60 fps, but object/feature sharing — our case — is ~10 Hz.)
- **Positional tolerance ε ≈ 0.5 m** (lane-level) — the anchor for the staleness experiment; low-confidence as a
  hard standard, so we DERIVE our own via Analysis #1 and cross-check against 0.5 m.

## Our measured configs vs the anchor (single-UE OAI, no impairment)
| config | Y | delivery | vs ~100 ms / 99% cooperative-perception budget |
|---|--:|--:|---|
| no-AE u8 (baseline) | 267 ms | 75% | **FAILS** (2.7× latency, well below 99%) |
| AE-128 u8 | 152 ms | 99.3% | latency over budget; reliability ✓ |
| **AE-128 u4** | **105 ms** | **99%** | **MEETS** ~100 ms / 99% (cooperative perception + ToD uplink) |

**Headline:** feature compression moves the system from **standards-noncompliant** (no-AE 267 ms / 75%) to
**compliant** with 3GPP cooperative-perception (100 ms / 99%) and 5GAA ToD (~120 ms) budgets at low/mid
automation. High-automation (3–10 ms) is out of reach for any edge-relayed config — that regime is direct
sidelink. Full sources + per-claim votes: research output archived in session task `wcizs0tol`.
