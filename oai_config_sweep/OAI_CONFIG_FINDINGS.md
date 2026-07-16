# OAI config sweep — findings (2026-07-16)

Model FIXED at no-AE u8 (1141 KB) so the network config is the only variable. Single-UE, rfsim, no channel
impairment. 300 frames/config. Metrics from the front stream CSV. Raw: `oai_config_results.tsv`.

## Results (what actually completed)
| config | RTT mean | RTT p95 | delivery | payload | frag/frame |
|---|--:|--:|--:|--:|--:|
| **TDD 7:2** (baseline, 5QI=9, K2=6) | 173.8 ms | 207 ms | **78.0%** | 1115 KB | 19.6 |
| **TDD 4:5** (K2=6, ~60% UL airtime) | 181.8 ms | 200 ms | **80.7%** | 1115 KB | 19.6 |
| **5QI=1 (GBR)** on TDD 7:2 | 191.1 ms | 211 ms | **79.3%** | 1113 KB | 19.3 |

## Headline finding
**In this single-UE rfsim setup, the network-config levers barely move transport.** Shifting TDD from 20% → 60%
uplink airtime changed delivery only 78% → 81% (RTT flat within noise); a GBR 5QI changed nothing (79%).
→ The **payload is the bottleneck**, so **compression (AE) is the effective lever** — consistent with the A/B
(AE-128 u4: delivery 75%→99%, RTT 209→77 ms). Config tuning in this setup does not.

Why config barely helps (reasoned + evidenced):
- **TDD:** even with 3× the uplink airtime, a 1.14 MB / ~20-fragment frame still can't drain fast — the frame
  size overwhelms the uplink regardless of slot count. Cutting the frame (compression) is what fits the grant.
- **5QI:** priority/GBR only matters with *competing* flows. With a single UE + single flow there's nothing to
  prioritize against, and OAI's rfsim scheduler doesn't enforce a GBR reservation that helps here.

## What did NOT run, and why (all diagnosed — not silent failures)
- **Extreme uplink TDD (2:7 / 3:6) — not achievable in this OAI build.** The gNB asserts `N_dl1 >= k2-1` (fails
  at K2=6 with 2 DL slots); lowering `min_rxtxtime` to 2 lets the *gNB* start (3 DL : 8 UL), but then the *UE*
  crashes in `nr_ue_process_dci_dl_10` (`_Assert_Exit_`). So the analysis's "best-case UL-favored TDD" is blocked
  by both a gNB (K2) and a UE (DCI) constraint — would need deeper protocol work.
- **Bandwidth / PRB 162/217/273 — deferred.** RIV auto-corrected (275*(L-1) for L≤138, else 275*(276-L)+274),
  but wide PRB also needs SSB/PointA/coreset0 re-derivation (not automated) → UE won't attach. A focused pass.
- **The overnight run only yielded 2 points** because (a) 2:7 crashed the gNB and (b) I'd mistakenly based the
  5QI phase on that broken 2:7 conf. Both fixed in the re-run; the re-run also exposed that repeated automated
  rfsim gNB↔UE restarts are flaky (UE sometimes can't reconnect to :4043), so the clean points above were taken
  with careful one-at-a-time manual bring-ups.

## Takeaway for the supervisor / next steps
- **Message:** we tested the two network-side knobs the analysis proposed (TDD, 5QI); in this single-UE, no-impairment
  setup they don't meaningfully change delivery or latency — **compression remains the effective transport lever.**
- Config tuning may matter more (a) under a **realistic channel** (impairment/fading) or (b) with **multiple UEs
  contending** (then 5QI priority + TDD split bite) — that's the SIONA-RT stress-test phase, where these knobs
  become the RL controller's "network action menu."
- If a clean bandwidth sweep is wanted, derive SSB/PointA/coreset0 per PRB (or use OAI's shipped wider-PRB templates).
