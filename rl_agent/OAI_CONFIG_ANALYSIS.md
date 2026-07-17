# OAI config — current settings, why latency is high, and what to tune next

> **Pre-sweep analysis (historical).** The configuration description remains useful, but its tuning
> expectations have now been tested. Current conclusion: TDD 7:2→4:5 and 5QI 9→1 barely move single-UE
> RFsim transport; compression is the effective lever. See `../oai_config_sweep/OAI_CONFIG_FINDINGS.md`.

Single-UE OAI 5G, rfsim, **no channel impairment**. This is the "network side" the UE agent will eventually
negotiate against. Companion to `OAI_AB_RESULTS.md` (the compression A/B).

## Current configuration (highlight for slides)
| parameter | value | notes |
|---|---|---|
| band | n78 (TDD, FR1) | 3.5 GHz |
| bandwidth | **106 PRB = 40 MHz** | 30 kHz SCS (numerology μ=1), 10 slots / 5 ms |
| **TDD pattern** | **7 DL : 1 special : 2 UL** | `dl_UL_TransmissionPeriodicity=6` (5 ms); special = 6 DL + 4 UL symbols |
| → uplink airtime | **~20–24%** | only ~2 of every 10 slots carry uplink |
| QoS flow | **5QI = 9** | best-effort, **non-GBR**, no delay bound, no priority |
| scheduler | OAI default MAC | UL grants driven by SR/BSR loop |
| MIMO | 2×2 | |
| achieved uplink | **~9 Mbps** | at this TDD split |

## Why transport latency is high (75% delivery, ~200 ms transport at 1.14 MB)
Idle-link ping RTT is **3.85 ms** — so the ~200 ms is *not* propagation. Root causes, in order:

1. **TDD is DL-favored (7:2), traffic is UL-heavy.** Features flow UE→edge (uplink), but only ~2/10 slots
   are uplink. The uplink is starved → each 1.14 MB frame (~20 UDP fragments) waits many periods to drain.
   **This is the dominant bottleneck.**
2. **MAC buffer / BSR–grant loop.** UE reports buffer status (BSR), gNB grants UL; with only 2 UL slots the
   grants are small/infrequent, so the UE UL buffer fills faster than it drains → **buffer overflow → the 25%
   UDP loss**. Delivery and latency are the *same* bottleneck seen two ways.
3. **5QI = 9 (best-effort).** No guaranteed bitrate, no delay budget, no scheduling priority for our flow.
4. **Bandwidth cap.** 106 PRB @ 40 MHz bounds peak rate; combined with the 20% UL airtime → ~9 Mbps uplink.

Compression attacks (1)+(2) from the UE side (smaller payload = fewer fragments = fits the grant); config
tuning attacks them from the network side. **Two independent levers on the same bottleneck.**

## Config options to explore (= what the UE/network agent can request)
**A. TDD DL/UL slot ratio (biggest lever for our uplink-heavy traffic).** Change `nrofDownlinkSlots` /
`nrofUplinkSlots` (+ special-slot symbols), keep the 5 ms period.
| pattern | DL:UL slots | UL airtime | expectation for our traffic |
|---|---|---|---|
| **7-2 (current)** | 7 : 2 | ~20% | baseline: ~200 ms transport, 75% delivery |
| **5-5** | ~4 : 4 (+special) | ~45% | ~2× uplink → large drop in transport + loss |
| **3-7** | ~3 : 6 (+special) | ~65% | uplink-favored → best case for UE→edge features |
*(Trade-off: more UL slots steal DL capacity — fine here since the result/downlink is tiny.)*

**B. Bandwidth / PRB.** n78 @ 30 kHz supports 24…273 PRB. Options above 106: **133 (≈50 MHz), 162 (≈60),
217 (≈80), 273 (≈100 MHz)** → more PRBs per UL slot = higher uplink capacity. (rfsim has no RF cost to widening.)

**C. QoS / 5QI.** Move the flow off best-effort 5QI=9 to a **GBR** (e.g. 5QI 1/2/3) or **delay-critical**
(5QI 82–85) profile to get scheduling priority / a delay budget on the uplink.

**D. Scheduler knobs (finer).** SR/BSR periodicity, `min_rxtxtime` (K2), max UL MCS / grant size.

## Experiment plan (next)
Fix the model at the **no-AE u8 baseline (1141 KB)** so the *config* is the only variable, and sweep:
TDD {7-2, 5-5, 3-7} × PRB {106, 217, 273}, no impairment, 300 frames each → RTT / delivery / goodput curve.
Then confirm the winning config **also** helps a compressed model (compression × config compose).
Workflow: edit a **copy** of the gNB conf per variant (never the original), user restarts gNB, rerun the
fixed harness. Deliverable: an RTT-vs-config table + the "agent's network action menu" (TDD / PRB / 5QI).
