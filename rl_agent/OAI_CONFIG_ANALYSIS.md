# OAI config — current settings, why latency is high, and what to tune next

> **Current note.** The original 2026-07-16 live-stream sweep found that TDD 7:2→4:5 and 5QI 9→1 barely
> moved single-UE RFsim transport, so compression was the strongest lever. A 2026-07-20 true-paced replay
> diagnostic did show network config sensitivity under forced ~92 Mbps offered load, but the corrected
> 2026-07-20 live CARLA frontend runs are more conservative: UL-heavy TDD modestly improves latency but not
> delivery, while the manual validated 273PRB run slightly improves delivery but worsens p50 latency. Important:
> the manual `bw273_mu1` 273PRB result is reportable; the failed automated `prb_273` sweep is a separate
> non-reportable bring-up failure caused by mismatched center-frequency/SSB parameters. So the current reading is:
> compression/payload reduction remains the primary UE-side lever; OAI config is a secondary/contextual
> network-side lever, not a standalone fix for no-AE 10 FPS deployment.

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

**B. Bandwidth / PRB.** n78 @ 30 kHz supports wider PRB settings. The manual `bw273_mu1` setup is a validated
273PRB condition (`-r 273 -C 3649260000 --ssb 516` with the 273PRB gNB config), but the result is mixed:
delivery improves slightly and p50 latency worsens. Treat 273PRB as a reportable diagnostic/network-side action
candidate, not as a solved latency fix. The automated `prb_273` sweep remains invalid because it launched the UE
with mismatched center-frequency/SSB settings.

**C. QoS / 5QI.** Move the flow off best-effort 5QI=9 to a **GBR** (e.g. 5QI 1/2/3) or **delay-critical**
(5QI 82–85) profile to get scheduling priority / a delay budget on the uplink.

**D. Scheduler knobs (finer).** SR/BSR periodicity, `min_rxtxtime` (K2), max UL MCS / grant size.

## Experiment plan (next)
Fix the model at the **no-AE u8 baseline (1141 KB)** so the *config* is the only variable. For reportable
conditions, keep default 106PRB, selected UL-heavy TDD/QoS variants, and the manual validated 273PRB recipe.
Do **not** include automated/generated 273PRB configs unless the UE/gNB logs prove the same working center
frequency/SSB/tunnel path. Then confirm any winning validated config **also** helps a compressed model
(compression × config compose).

## 2026-07-20 update — true-paced no-AE payload replay

This was a diagnostic, not the final Step-1 CARLA deployment result. The replay removed live CARLA frame generation
and offered the same synthetic no-AE zlib/per-channel-u8 payload at a clean 10 FPS:

- payload p50: ~1.15 MB/frame
- chunks p50: 20 chunks/frame
- offered load: ~92 Mbps
- frames: 400
- back-half and checkpoint fixed to the no-AE baseline

| Condition | RAN config | Delivery | RTT p50 | RTT p95 | Back/server p50 | Downlink p50 |
|---|---|---:|---:|---:|---:|---:|
| Default OAI replay | 106 PRB, mu=1, default TDD 7 DL / 2 UL | 50.5% | 103.8 ms | 172.0 ms | 10.4 ms | 7.2 ms |
| Wider bandwidth replay | 273 PRB, mu=1, default TDD 7 DL / 2 UL | 76.0% | 90.6 ms | 128.6 ms | 10.7 ms | 7.5 ms |
| UL-heavy TDD | 106 PRB, mu=1, TDD 4 DL / 5 UL | 83.5% | 111.0 ms | 154.5 ms | 11.2 ms | 7.5 ms |

Replay interpretation:

- Back/server compute and downlink stay nearly fixed, so this is not an inference or result-return bottleneck.
- Default OAI had bursty dead windows; UL-heavy TDD stayed around 88-94% delivery after the first warmup window.
- The manual validated 273PRB replay is reportable as a replay/transport diagnostic; do not mix it with the failed automated `prb_273` sweep.
- The remaining failure mode still looks like uplink frame-completion reliability for 20 UDP chunks/frame: frames
  either reconstruct and return quickly, or enough chunks miss their useful window that the application drops them.

The important caveat: replay is open-loop/fixed-rate, while the normal CARLA frontend is closed-loop and waits for
result/timeout before advancing. That difference changes what the experiment measures.

## 2026-07-20 corrected update — live CARLA frontend

The same two OAI config variants were rerun using the actual CARLA frontend path at 10 FPS / 1300 frames, matching
the Step-1 deployment harness.

| Condition | RAN config | Delivery | RTT p50 | RTT p95 | Feature/uplink handling p50 | Downlink p50 |
|---|---|---:|---:|---:|---:|---:|
| Default OAI | 106 PRB, mu=1, default TDD 7 DL / 2 UL | 72.6% | 209.0 ms | 279.5 ms | 187.4 ms | 11.1 ms |
| UL-heavy TDD | 106 PRB, mu=1, TDD 4 DL / 5 UL | 71.9% | 200.1 ms | 249.5 ms | 177.0 ms | 9.8 ms |
| Wider bandwidth | 273 PRB, mu=1, default TDD 7 DL / 2 UL | 74.9% | 235.2 ms | 263.7 ms | 216.1 ms | 9.6 ms |

Corrected interpretation:

- In the actual closed-loop CARLA deployment, simple OAI config changes do **not** solve no-AE delivery.
- UL-heavy TDD is the better latency variant: lower RTT p50/p95 and lower uplink-handling p50 than default.
- 273PRB is valid/reportable but mixed: slightly better delivery than default, worse RTT p50 and uplink-handling p50.
- Downlink remains cheap in all cases (~10 ms p50 for returned boxes).
- The replay result should be reported as an open-loop transport stress diagnostic, not as the main deployment result.

Current action from this corrected update: keep default OAI as the baseline, report UL-heavy as the only validated
low-latency improvement, report 273PRB as a validated but mixed wider-bandwidth diagnostic, and return focus to
payload reduction / compression and reliability characterization. If a CARLA-based open-loop offered-load result is
needed, use `--queue-probe-mode` rather than synthetic replay.
