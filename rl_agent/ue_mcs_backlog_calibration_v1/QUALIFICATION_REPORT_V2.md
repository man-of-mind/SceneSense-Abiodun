# SUPERSEDED — UE-local `[previous UL MCS, pre-enqueue backlog]` qualification

> **Do not cite the numerical analysis below.** This report describes the
> rejected v1-derived analysis and predates corrective commit `8f1b5da`.
> The authoritative create-only result is
> `experiments/ue_mcs_backlog_calibration_v1/20260924_131015/analysis_v2.json`,
> bound by `VERIFIER_INPUT_MANIFEST_V2.json`. Its verdict is `INCONCLUSIVE`:
> **4/7 scientific checks and 12/13 structural gates**. The sole structural
> failure is incomplete verifier-only gNB provenance for 4 of 5,400 decisions;
> UE-side MCS and backlog coverage remain 100%, with no MCS mismatch among
> matched records. See `INTEGRITY_AMENDMENT_DERIVED_V1_SUPERSEDED.md` for why
> the v1 derivatives below were rejected and retained only as history.

## Historical superseded report (retained verbatim below)

**Verdict: `INCONCLUSIVE`** — 5 of 7 registered checks, **12 of 12 structural gates**.

Evidence: `rl_agent/experiments/ue_mcs_backlog_calibration_v1/20260924_131015`
(12/12 cells, **5,400 decisions**, `manifest.json` with SHA-256 for every file, `analysis_v1.json`,
`figures/`). Preregistration: `PREREGISTRATION_V2.md`, committed before collection.

`INCONCLUSIVE` here is not "the data was bad". Every gate passed and both features were measured
cleanly. It means the **pair** was not validated as a state, for two specific and opposite reasons:
MCS is impeccably behaved but adds nothing predictive, and backlog carries all the prediction but is
history-dependent rather than a summary of present demand.

---

## 1. Headline

| Feature | Verdict on its own terms |
|---|---|
| `previous_ul_mcs` | Behaves **exactly** as claimed: load-invariant, channel-discriminating, perfectly repeatable, 100% available. **But near-chance as a predictor** (AUC 0.53). |
| `pre_enqueue_backlog_bytes` | **Dominant predictor** (AUC 0.96–0.99) and responds strongly to load. **But strongly order-dependent** — it reflects queue *history*, not current demand. |

Adding MCS to backlog does not help and marginally hurts: AUC **0.964 → 0.947** (`y_complete`) and
**0.995 → 0.982** (`y_in_budget`).

## 2. Steady state (last 80 decisions of each block)

| Channel | Tier | backlog P50 | MCS P50 (sd) | complete | latency P50 |
|---|---|---:|---:|---:|---:|
| ADVERSE_STABLE | low | 912,456 | **9.0** (1.86) | 0.667 | 26.0 ms |
| ADVERSE_STABLE | medium | 15,099,772 | **9.0** (2.07) | 0.598 | 6,068 ms |
| ADVERSE_STABLE | high | 49,916,953 | **9.0** (2.08) | 0.000 | — |
| FAVORABLE_STABLE | low | 0 | **25.0** (2.93) | 1.000 | 23.0 ms |
| FAVORABLE_STABLE | medium | 0 | **25.0** (3.28) | 1.000 | 91.8 ms |
| FAVORABLE_STABLE | high | 49,005,015 | **25.0** (3.34) | 0.496 | 8,431 ms |

**Q1 — is MCS stable across load at a fixed channel? Yes, exactly.** The median is *identical* across
all three tiers within each channel: max gap **0.0 index units** in both. This is the strongest
possible form of the claimed semantics — the scheduler's MCS tracks the channel, not the demand.

**Q2 — does MCS separate the channels at fixed load? Yes.** 9 vs 25, Cliff's δ = **0.998 / 0.992 /
0.992** (low/medium/high), all LARGE.

**Q3 — does backlog respond to load? Yes.** high-vs-low Cliff's δ = **0.990** (FAVORABLE) and
**1.000** (ADVERSE).

**Channel × load interaction is real and visible.** At medium (21.1 Mbps offered) the favorable
channel drains cleanly (complete 1.000, 91.8 ms) while the adverse channel collapses (0.598,
6,068 ms). Capacity is channel-dependent, and that is precisely the regime where an exogenous
channel signal ought to matter.

## 3. Q4 — prediction (leave-one-cell-out, folds are whole cells)

| Target | A: MCS only | B: backlog only | C: both | usable rows |
|---|---:|---:|---:|---:|
| next-frame complete reassembly | 0.5257 ± 0.0961 | **0.9643 ± 0.0456** | 0.9466 ± 0.0484 | 5,388 |
| next-frame within transport budget | 0.5542 ± 0.1367 | **0.9947 ± 0.0019** | 0.9816 ± 0.0180 | 3,943 |

MCS alone is **indistinguishable from chance**. That is not a contradiction of §2: within a cell the
channel is fixed, so MCS is nearly constant and cannot rank frames; across cells both channels
produce good and bad outcomes depending on load. Backlog already encodes the joint effect of load
*and* channel, because it is the queue those two jointly produced.

**Important distinction.** This tests **prediction given the current state**, not **control**.
Backlog is a *consequence* of past actions; MCS is *exogenous*. A controller choosing payload size
cannot treat them as interchangeable just because backlog dominates a one-step predictor. This
experiment does not license dropping MCS from an action-selection state.

## 4. Repeatability — the finding the counterbalancing bought

| Channel | Tier | backlog P50 rep0 | backlog P50 rep1 | effect | MCS effect |
|---|---|---:|---:|---|---|
| ADVERSE | low | 27,874,302 | 0 | MEDIUM | NEGLIGIBLE |
| ADVERSE | medium | 12,186,796 | 49,898,919 | **LARGE** | NEGLIGIBLE |
| ADVERSE | high | 49,915,807 | 49,917,665 | NEGLIGIBLE | NEGLIGIBLE |
| FAVORABLE | low | 0 | 0 | MEDIUM | NEGLIGIBLE |
| FAVORABLE | medium | 0 | 22,577,087 | **LARGE** | NEGLIGIBLE |
| FAVORABLE | high | 49,203,111 | 48,772,068 | NEGLIGIBLE | NEGLIGIBLE |

**MCS is perfectly repeatable** — every effect NEGLIGIBLE, medians identical (9.0/9.0, 25.0/25.0).

**Backlog is not.** Repetition 1 runs the exact reverse order, and at low and medium the same tier
under the same channel yields wildly different backlog depending on what preceded it — 0 vs 27.9 MB,
0 vs 22.6 MB. Only `high` repeats, because it saturates regardless of history.

This is **hysteresis, not noise**: backlog at tier *T* is dominated by the block before *T*. It is
therefore **not a Markov summary of current demand**. The withdrawn v1 design — one fixed payload per
cell — could not have detected this at all, which is the concrete payoff of the amended matrix.

## 5. Transients and saturation

Block 0 of every cell starts from an empty queue (RAN rebuilt cold), and the position-balanced design
puts each tier there equally often, giving a clean from-empty transient per tier:

| Tier at position 0 | n | backlog P50 over first 30 decisions | at ceiling |
|---|---:|---:|---:|
| low | 120 | 0 | 0.000 |
| medium | 120 | 0 | 0.000 |
| high | 120 | 6,851,329 | 0.000 |

Saturation over the whole campaign: ceiling **49,984,583 B**, overall **23.3%** of decisions at ≥95%
of it (high 44.4%, medium 22.2%, low 3.3%). Backlog is **not** degenerate — 830 / 1,337 / 1,788
distinct values by tier — but a quarter of the campaign sits against the buffer limit, which bounds
how much the feature can discriminate there.

The pinned tiers offer **0.50 / 21.08 / 70.45 Mbps** at 10 fps against a ~6 Mbps uplink, so medium and
high are 3.5× and 12× over capacity. That is consistent with the project's own load-shaping frontier
(400 KiB → 7/200 feasible cells). The tiers were pinned by the study design and were not tuned away.

## 6. Measurement quality

| Item | Result |
|---|---|
| Cells | **12/12** captured |
| Decisions | **5,400** (12 × 450) |
| Clock bridge | `NR_PDCP_TX_SDU` **same-event** in all 12 cells; residual P95 ≤ **0.48 µs**; window audit 1.000 |
| MCS coverage | **1.0000** in every cell (no stale, no missing) |
| Backlog coverage | **1.0000** in every cell |
| Grants used | 482,852 new-data; **6,406 retransmissions excluded**; 29,641 non-UL excluded |
| MCS table | **0** everywhere (as `get_mcs_from_SINRx10` requires) |
| Terminal accounting | 3,955 complete + 627 incomplete + 818 no-arrival-in-window = **5,400**, exact |
| Upper anchor | −12.5 dB → **25.0 dB** achieved median PUSCH SNR (412 samples), measured not extrapolated |

All **12/12 structural gates** pass: cell identity and profile read-back, exact accounting, no
negative interval, no cross-cell queue contamination, every observation precedes its decision,
retransmissions excluded, missing MCS never zero, raw backlog retained, MCS table constant, RF
restored to −50 with read-back, no orphan process, CARLA and CUDA untouched.

## 7. Measured normalization bounds

Reported, **not applied**; no agent was modified and no training was run.

```
pre_enqueue_backlog_bytes : min 0, P50 20,931,256, P95 49,925,882, P99 49,943,990, max 49,984,583
                            transform  log1p(bytes) / log1p(P99)
                            WARNING    the deployed log1p_scale=1.0 maps every backlog above
                                       1 byte to 1.0 and must not be reused
previous_ul_mcs           : min 8, P50 16, max 28
                            transform  mcs / 28   (MCS table 0 upper index)
                            missing    must be signalled explicitly, never encoded as 0,
                                       which is a real modulation index
```

### Causal runtime sampling contract

1. Read the backlog sample from the **last** `NRUE_MAC_RLC_BUFFER_STATUS` tick **strictly before** the
   payload is handed to PDCP. The tagged payload's own bytes are then excluded structurally.
2. Read MCS from the **latest strictly prior** UE-decoded UL DCI with **HARQ round 0**. Retransmission
   grants repeat an older decision and must be skipped.
3. Reject an MCS older than **200 ms**; emit an explicit missing signal. Never forward-fill, never 0.
4. Carry age as a **validity gate only**. It is not a policy feature.
5. Both are UE-local. No SRS, PUSCH SNR, RSRP, CQI or gNB telemetry is read at runtime.

## 8. Limitations

1. **Backlog is order-dependent** (§4). Any state using it inherits queue history, which is a
   consequence of the policy's own past actions.
2. **MCS's incremental value was not tested where it should matter.** Two of three tiers sit far
   beyond capacity and the third far below; the interesting regime is *near* capacity, where the
   channel decides feasibility (visible at medium: 1.000 vs 0.598 complete).
3. **Prediction, not control.** One-step next-frame prediction says nothing about a sequential
   controller choosing payload.
4. Simulated radio (RFsim), single UE, one host, 2 repetitions, ~45 s of traffic per cell.
5. 818 of 5,400 frames had no arrival inside the observation window; under a saturated uplink a
   datagram may still have been queued when the receiver closed, so that is a bounded observation,
   not proven loss.
6. Loss appears largely at the tun qdisc (qlen 500) rather than as socket backpressure — the sender
   recorded 0 socket drops — so "chunks handed to socket" overstates what entered RLC.

## 9. Required statements

- **UL MCS is a delayed, quantized scheduler decision** derived from gNB-measured uplink SNR in this
  custom build (`get_mcs_from_SINRx10`, `gNB_scheduler_ulsch.c:2028`, gated at `:2027`).
- **It reaches the UE through standard DCI** with no added controller.
- **Backlog represents demand/queue pressure, not physical channel.**
- **This is a bounded engineering qualification, not repeated publication-level evidence.**

## 10. Recommendation

Do not adopt or reject the pair on this evidence. The productive next step is a bounded **near-capacity**
sweep — offered loads bracketing the channel-dependent capacity, where §2 already shows the two
channels diverge — testing whether MCS adds incremental value once backlog is no longer pinned by
gross over-offer, and whether an explicit history term removes backlog's order dependence. No agent
change and no training is authorized by this study.
