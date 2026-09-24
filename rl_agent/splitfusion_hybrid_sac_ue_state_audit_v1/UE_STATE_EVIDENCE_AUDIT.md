# UE-state evidence audit — pre-action BSR feature for SplitFusion Hybrid-SAC

**Status:** `INSUFFICIENT_OR_CAUSALLY_UNRESOLVED` / `PRE_ACTION_BSR_SOURCE_UNRESOLVED`
**Phase:** offline, read-only evidence audit. No policy, environment, reward, runtime, trainer or
frozen contract was modified. No CARLA, OAI, RFsim, Docker, map server, CUDA or model inference was
launched.
**Date:** 2026-09-23
**Tooling:** `audit_ue_state_evidence.py` (this directory), 77 unit tests in
`test_audit_ue_state_evidence.py`.

---

## 1. Executive decision

Retained OAI traces **cannot** support a causal, action-conditioned BSR feature in the Hybrid-SAC
state. Two independent blockers, either of which is sufficient:

1. **No retained source proves the pre-action read ordering.** `NRUE_MAC_BSR_STATUS` is emitted
   *after* uplink logical-channel multiplexing, so its byte counts are what the current grant left
   behind — it is contaminated by the action. `NRUE_MAC_RLC_BUFFER_STATUS` is emitted *before*
   multiplexing and is the right *kind* of source, but "pre-multiplex" is a weaker property than
   "pre-enqueue". Proving the policy read older backlog requires knowing when the application
   enqueued the payload, and **no retained run contains an application-enqueue timestamp**
   (`NR_PDCP_TX_SDU` and `NR_RLC_TX_SDU` are defined in `T_messages.txt` but extracted in 0 of 15
   runs). `NR_RLC_TX_DEQUEUE` is **not** an enqueue candidate: it fires when a PDU is handed down to
   MAC for a grant, i.e. it *ends* the RLC queue wait, so treating it as an enqueue instant would
   date the payload to when it was served rather than when it arrived.
2. **Offered load does not *span* the action range, and no run varies payload at all.** All 15 runs
   use a fixed-rate generator. Across every run there are exactly **two** distinct offered payload
   sizes, 12 500 B and 25 000 B, both at a fixed 0.1 s period, with **no within-run variation**.
   Against the frozen 72-action FCOS catalogue (**6 229 B – 3 568 326 B**) both values sit *inside*
   the low end of the action range — they are **not** below it — and both also fall inside the Run-3
   modeled support (**6 423 B – 427 605 B**). They are insufficient for a different reason than
   coverage of the low end: there are only **two** sizes, payload is **constant within each run**,
   and no run pairs a changed payload with an observed queue transition, so **no action-conditioned
   queue transition can be inferred**.

**Recommended pre-action source: `UNRESOLVED`.** `NRUE_MAC_RLC_BUFFER_STATUS` is the correct
*candidate* and the only one that survives the action-leakage check, but it is not yet qualified.

**A bounded calibration is required** (§11). A new 288-cell campaign is **not** required and is not
recommended.

**Separately and independently confirmed:** the currently frozen normalization
`clip(log1p(bsr_bytes), 0, 1)` destroys **100.0000 %** of nonzero queue information across all
324 790 nonzero backlog samples in the evidence (§9). That defect is real regardless of how the
causal question resolves.

---

## 2. Exact meaning of zero

There are **three different zeros** in this evidence and they are not interchangeable.

| Zero | What it actually asserts |
|---|---|
| `NRUE_MAC_RLC_BUFFER_STATUS.bytes_in_buffer == 0` | No bytes were waiting in *that RLC transmit buffer* at *that MAC tick*. |
| `NRUE_MAC_BSR_STATUS.lcg*_bytes == 0` | Nothing was left to report *after the current grant was filled*. The weakest zero: it is fully consistent with a large backlog that this grant happened to drain. |
| `GNB_MAC_UL_MCS_DECISION.estimated_ul_buffer == 0` | The *gNB believes* the UE has nothing pending, decoded from a BSR table index. |

**A zero at any of these layers does not mean zero network delay.** It is silent about all of:
PDUs already handed to lower layers, HARQ retransmissions and round-trips, scheduler and grant wait,
gNB / core / edge queueing, and — decisively for a pre-action feature — **delay from the payload that
is about to be enqueued**. The retained runs make this concrete: in
`20260821_clean_control_03` the RLC buffer reads zero on 83.22 % of ticks while the link is
continuously carrying 1 Mbps of uplink traffic.

A related trap: the current environment always supplies `bsr_bytes = 0`
(`empirical_contextual_environment.py:309`, provenance `INERT_FOR_EXACT_GENESIS_ZERO_ONLY`). That
zero is a *placeholder*, not a measurement, and must not be read as evidence of an idle queue.

---

## 3. RLC-buffer versus MAC-BSR, side by side

| Question | `NRUE_MAC_RLC_BUFFER_STATUS` | `NRUE_MAC_BSR_STATUS` |
|---|---|---|
| **What object is measured?** | UE MAC's read of RLC transmit-buffer occupancy, per logical channel, via `nr_mac_rlc_status_ind` | The BSR MAC-CE about to be written into the current grant, plus the LCG counts it encodes |
| **Are byte fields occupancy, indices, reported values, or residual?** | **True queue occupancy in bytes**, unquantized | **Post-multiplex residual bytes** (`lcg*_bytes`); `bsr_index` / `bsr_long*_index` are **coarse BSR table indices**, not bytes |
| **All logical channel groups represented?** | One row per active LCID per tick, each carrying its LCGID. Total backlog = plain sum over the tick's rows | All 8 LCGs in one row, but only the groups the encoder populated are meaningful (a short BSR reports one LCG) |
| **At what event is the row emitted?** | Once per active LCID in `nr_update_rlc_buffers_status`, **every UL MAC tick while CONNECTED**, grant or no grant | Once **per filled UL grant**, after the multiplexing loop has drained the buffer into the PDU |
| **Samplable at the RL decision point before current-payload enqueue?** | Available at the right point in the MAC, but the MAC tick is **asynchronous to the SplitFusion frame**; retained evidence cannot place it relative to enqueue | **No.** It only exists *because* a grant was filled with the current payload |
| **What does zero mean?** | Empty RLC buffer at that tick (see §2) | Nothing left after this grant (see §2) |
| **Directly available inside the UE runtime?** | Yes (UE MAC) | Yes (UE MAC) |
| **Could it leak the action into its own state?** | **Yes, if sampled after enqueue** — this is exactly what the calibration must pin down | **Yes, unavoidably** — it is a function of how the current payload was multiplexed |
| **Sample population** | 792 868 ticks across 15 runs | 104 381 rows — a *grant-conditioned subset*, ~7.6× sparser |

The last row matters and is easy to miss: the two sources do **not** describe the same population.
Pooled zero fraction is 59.04 % for RLC ticks versus 37.59 % for BSR rows, because BSR rows only
exist when a grant existed. Swapping one for the other silently changes the conditioning.

---

## 4. Source-code provenance and timing order

All citations are from the worktree OAI copy that built these softmodems.

| Fact | Location |
|---|---|
| T-event timestamp is `CLOCK_REALTIME` **at the `T()` call site** | `OAI/openairinterface5g/common/utils/T/T.h:176,195,204` |
| CSV sink renders it as `HH:MM:SS.microseconds`, **dropping date and timezone** | `OAI/openairinterface5g/common/utils/T/tracer/csv.c:32` |
| `NRUE_MAC_RLC_BUFFER_STATUS` declared "*before BSR update*" | `common/utils/T/T_messages.txt:248-251` |
| `NRUE_MAC_BSR_STATUS` declared "*after uplink logical-channel multiplexing*" | `common/utils/T/T_messages.txt:252-255` |
| RLC-buffer event emitted | `openair2/LAYER2/NR_MAC_UE/nr_ue_scheduler.c:1462` |
| …its caller `nr_update_rlc_buffers_status` runs | `nr_ue_scheduler.c:2611` |
| `nr_ue_get_sdu` runs (only when a grant exists) | `nr_ue_scheduler.c:2669` |
| `nr_update_bsr` accumulates `LCG_bytes += LCID_buffer_remain` | `nr_ue_scheduler.c:1511` |
| **`fill_mac_sdu` decrements `LCG_bytes -= sdu_length`** | `nr_ue_scheduler.c:2414` |
| BSR event emitted, logging the **decremented** `LCG_bytes` | `nr_ue_scheduler.c:2182` |
| …reached from the end of `nr_ue_get_sdu` | `nr_ue_scheduler.c:2561` |
| BSR indices encoded via `nr_locate_BsrIndexByBufferSize` | `nr_ue_scheduler.c:2093,2108` |

**Established order within one UL MAC tick:**

```
nr_ue_ul_scheduler
  └─ nr_update_rlc_buffers_status        (:2611)
       └─ T_NRUE_MAC_RLC_BUFFER_STATUS   (:1462)   ← PRE-multiplex occupancy
  └─ [only if a grant exists] nr_ue_get_sdu        (:2669)
       ├─ nr_update_bsr: LCG_bytes += remain       (:1511)
       ├─ fill_mac_sdu: LCG_bytes -= sdu_length    (:2414)   ← the action drains it
       └─ nr_ue_get_sdu_mac_ce_post                (:2561)
            └─ T_NRUE_MAC_BSR_STATUS              (:2182)   ← POST-multiplex residual
```

This is confirmed empirically, not only read from source: the BSR↔RLC one-to-one join matches
**100 %** of BSR rows in all 15 runs at a median skew of **−3 µs** (BSR follows RLC by 3 µs), with
maximum |skew| 22 µs.

**Classification is therefore resolved for multiplex ordering and unresolved for enqueue ordering.**
The chain a pre-action feature needs is:

```
read older UE backlog → construct policy state → select (mode, q) → construct/enqueue payload
```

Code settles only that the RLC read precedes *multiplexing*. The application enqueue happens on the
SplitFusion frame clock, asynchronously to the MAC tick, and no retained trace timestamps it.

---

## 5. Same-run alignment

Logical run identity is taken from **directory structure** — the parent of the run's `ttracer/`
directory — never from filename similarity. Cross-run joins raise `RunIsolationError`.

Join keys are `(frame, slot)` **plus** a bounded time window. The key alone is never trusted:
`(frame, slot)` recurs, so a key-only join is many-to-many by construction. Every window is checked
against the **measured** smallest same-key recurrence in the right-hand source and refused if it
could reach two counterparts.

| Left | Right | Domains | Max skew — justification | Coverage | Median skew |
|---|---|---|---|---:|---:|
| `NRUE_MAC_BSR_STATUS` | `NRUE_MAC_RLC_BUFFER_STATUS` (tick totals) | identical | **166–189 µs** = ½ the measured RLC tick-interval P50 (331–377 µs) in each run | **1.000** (15/15 runs) | −3 µs (−4 in one) |
| `NRUE_MAC_BSR_STATUS` | `NRUE_MAC_DCI_GRANT` (UL) | identical | ~3.3 s = just under ½ the measured same-key recurrence. Deliberately *not* a same-tick bound: the DCI is logged at grant reception, `sched_frame/sched_slot` names a later slot | **1.000** (15/15) | **−1004 to −1136 µs** (the K2 offset) |
| `NRUE_MAC_BSR_STATUS` | `UE_PHY_UL_PAYLOAD_TX_BITS` | identical | same-tick bound as row 1 | **1.000** (14/15) | +3 to +5 µs |
| `NRUE_MAC_BSR_STATUS` (UE) | `GNB_MAC_PUSCH_POWER_CONTROL` (gNB) | identical *(same host, both `CLOCK_REALTIME`)* | ~3.3 s, recurrence-derived; residual skew is the physical offset and is **reported, not constrained** | 0.876–1.000 | **+1464 to +2700 µs** |
| `NRUE_MAC_RLC_BUFFER_STATUS` | `traffic/sender.csv` | **`T_TRACER_REALTIME_LOCAL` vs `EPOCH_WALL_SECONDS`** | — | **REFUSED** | — |

No join reused a right-hand row in any run (`reused_right_rows = 0` everywhere). Missing values are
preserved as missing, never filled with zero.

**Refusals and partial coverage, explained rather than papered over:**

- **`traffic/sender.csv` join is refused.** The sender writes epoch seconds; the tracer renders
  `CLOCK_REALTIME` as *date-less local* `HH:MM:SS` (`tracer/csv.c:32`). The underlying clock is the
  same, but recovering the offset requires reconstructing the trace date and UTC offset. That is an
  assumption, not a measured bridge, so the join is refused. Offered load is summarized from
  `sender.csv` *on its own*, which needs no join.
- **`20260821_meeting_smoke_04` has 0.000 TX-bits coverage.** `UE_PHY_UL_PAYLOAD_TX_BITS.csv` in
  that run is **header-only, 0 data rows**. This is missing evidence, not a join defect.
- **The three `minus2p0` sustain runs have 0.876–0.880 gNB coverage.** The UE emits ~11 100 BSR
  events but the gNB logs only ~10 800 PUSCH rows: under the degraded channel, ~12 % of UE
  transmissions were never observed gNB-side. This is a finding, and it reinforces §7 — **UE-side and
  gNB-side evidence are not in one-to-one correspondence exactly when the channel is interesting.**

---

## 6. Distributions

### 6.1 Per-run (the primary table — deliberately not pooled)

RLC total backlog is summed across all LCIDs per tick. Bytes.

| Run | ticks | zero frac | P50 | P90 | P95 | P99 | max | tick P50 (µs) | gNB SNR P50 (dB) | dominant grant MCS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `ue_n2…/20260821_meeting_smoke_04` | 12 279 | 0.7552 | 0 | 25 425 | 25 467 | 25 467 | 25 467 | 377 | 10.0 | 9 (33 %) |
| `ue_n3…command_search_02/rung_00_minus4p0` | 27 712 | 0.8204 | 0 | 0 | 12 751 | 12 751 | 12 751 | 340 | 8.5 | 9 (63 %) |
| `…/rung_01_minus3p5` | 27 146 | 0.8098 | 0 | 0 | 12 751 | 12 751 | 37 061 | 351 | 7.5 | 9 (64 %) |
| `…/rung_02_minus3p0` | 27 273 | 0.7708 | 0 | 12 741 | 12 751 | 37 861 | 74 880 | 342 | 6.5 | 9 (64 %) |
| `…/rung_03_minus2p5` | 27 002 | 0.6825 | 0 | 12 751 | 12 751 | 148 238 | 186 203 | 348 | 6.0 | 8 (61 %) |
| `…/rung_04_minus2p0` | 26 305 | 0.8368 | 0 | 0 | 12 751 | 12 751 | 27 019 | 331 | 50.5 | 28 (98 %) |
| `ue_n3…/20260821_clean_control_03` | 62 100 | 0.8322 | 0 | 0 | 12 751 | 12 751 | 12 751 | 332 | 50.5 | 28 (100 %) |
| `ue_n3a…/sequence_00_rep_01_minus2p5` | 75 623 | 0.5899 | 0 | 12 751 | 12 751 | 103 027 | 186 289 | 347 | 6.0 | 8 (89 %) |
| `ue_n3a…/sequence_01_rep_01_minus2p0` | 74 748 | 0.3306 | 490 | 12 751 | 13 581 | 159 734 | 247 362 | 345 | 5.0 | 7 (92 %) |
| `ue_n3a…/sequence_02_rep_02_minus2p5` | 75 072 | 0.5853 | 0 | 12 751 | 12 751 | 12 751 | 25 354 | 344 | 6.0 | 8 (89 %) |
| `ue_n3a…/sequence_03_rep_02_minus2p0` | 74 847 | 0.3310 | 830 | 12 751 | 13 581 | 159 951 | 247 827 | 343 | 5.0 | 7 (92 %) |
| `ue_n3a…/sequence_04_rep_03_minus2p5` | 75 122 | 0.5880 | 0 | 12 751 | 12 751 | 87 246 | 186 884 | 345 | 6.0 | 8 (89 %) |
| `ue_n3a…/sequence_05_rep_03_minus2p0` | 74 598 | 0.3335 | 493 | 12 751 | 13 581 | 160 106 | 247 710 | 346 | 5.0 | 7 (92 %) |
| `ue_n3c…/rep_01_minus3p0_cold` | 66 858 | 0.7083 | 0 | 12 741 | 12 751 | 12 751 | 12 751 | 342 | 6.5 | 9 (94 %) |
| `ue_n3c…/rep_03_minus3p0_cold` | 66 183 | 0.7025 | 0 | 12 741 | 12 751 | 12 751 | 12 751 | 343 | 6.5 | 9 (94 %) |

Per-run reporting is not cosmetic. Zero fraction ranges **0.331 → 0.837** — a 2.5× spread driven by
the commanded SNR rung. Pooling would report a single 0.590 that describes no run.

Note `rung_04_minus2p0` reads SNR P50 50.5 dB and MCS 28, unlike the other `minus2p0` runs. Its
commanded degradation did not take effect for the bulk of the run. It is reported as measured and
**should not be treated as a degraded-channel sample.**

### 6.2 Pooled (stated separately, never substituted for the above)

| Quantity | Value |
|---|---|
| RLC ticks, all runs | 792 868 |
| RLC zero ticks | 468 078 (**0.5904**) |
| BSR rows, all runs | 104 381 |
| BSR zero rows | 39 235 (**0.3759**) |
| Nonzero RLC backlog samples | 324 790 |

### 6.3 Lagged descriptive association — **not causal**

Pearson *r*, per run, on a **fixed-rate generator with no action variation**. Three quantities are
kept apart because they mean different things: `served_sdu_bytes` is data the MAC actually
multiplexed; `granted_tb_bytes` is the transport block **including padding**, so it is the *grant*,
not data sent; `next backlog` is the following tick's pre-multiplex total.

| Run group | r(backlog, served) | r(backlog, granted TB) | r(backlog ₜ, backlog ₜ₊₁) | r(served ₜ, backlog ₜ₊₁) |
|---|---:|---:|---:|---:|
| clean control | +0.395 | −0.352 | +0.938 | −0.140 |
| `minus2p5` sustain | +0.168 … +0.328 | −0.205 … −0.056 | +0.971 … +0.996 | +0.072 … +0.076 |
| `minus2p0` sustain | +0.129 … +0.134 | −0.007 … −0.006 | **+0.998** | +0.072 … +0.076 |
| `minus3p0` cold | +0.365 / +0.366 | −0.139 / −0.137 | +0.965 | +0.134 |

Backlog is **strongly autocorrelated** (r up to 0.998), which is what a queue does and is not
evidence of anything controllable. The negative r(backlog, granted TB) is a property of a padded
transport block under a fixed generator and must not be read as "more backlog causes smaller
grants". **None of these numbers identify a response to an action that was never varied.**

---

## 7. UE-visible versus gNB-only

| Metric | Source | Visible to | Usable in UE policy state? |
|---|---|---|---|
| RLC buffer occupancy, per LCID | `NRUE_MAC_RLC_BUFFER_STATUS` | **UE** | Candidate, pending §11 |
| Post-multiplex LCG residual, BSR indices | `NRUE_MAC_BSR_STATUS` | **UE** | **No** — action leakage |
| Granted MCS, TBS, RB allocation, HARQ | `NRUE_MAC_DCI_GRANT` | **UE** (decoded DCI) | Yes in principle |
| Transmitted transport-block bits | `UE_PHY_UL_PAYLOAD_TX_BITS` | **UE** | Yes, but it is grant size incl. padding |
| **PUSCH SNR, RSSI, PHR, TPC** | `GNB_MAC_PUSCH_POWER_CONTROL` | **gNB only** | **No** — not observable at the UE |
| **Scheduler UL MCS decision, `avg_snr_x10`** | `GNB_MAC_UL_MCS_DECISION` | **gNB only** | **No** |
| **`estimated_ul_buffer`** | `GNB_MAC_UL_MCS_DECISION` | **gNB only** | **No** — and doubly disqualified (below) |

Two claims that must not be made:

- **gNB-side PUSCH/SNR is not UE visibility.** The UE does not observe uplink SNR. Any state feature
  built from `GNB_MAC_PUSCH_POWER_CONTROL` or `avg_snr_x10` is an oracle unless a downlink feedback
  path is built and measured. The three `minus2p0` runs make the gap concrete: 12 % of UE
  transmissions have no gNB record at all.
- **`estimated_ul_buffer` is not an independent backlog measurement.** Observed values in
  `20260821_clean_control_03` are exactly `{0, 276, 3909, 20516}` — decoded BSR table buckets. It
  inherits both the post-multiplex semantics *and* the BSR table's coarse quantization of the UE's
  own report. It is a lossy echo of a contaminated source.

**SNR, MCS and BSR are complementary, not interchangeable.** SNR is a *channel* quality the UE
cannot see uplink-side. MCS is the scheduler's *rate decision* given SNR and headroom, UE-visible via
DCI. BSR/RLC backlog is *demand* — how much work is waiting. A policy needs demand and service
separately; substituting one for another changes what is being controlled.

---

## 8. Action-conditioned coverage verdict

### `INSUFFICIENT_OR_CAUSALLY_UNRESOLVED`

| Criterion | Finding |
|---|---|
| Within-run payload variation | **None.** Every run has exactly one `frame_bytes`, one `period_s`, one `chunk_bytes`. |
| Across-run payload variation | **Two values total**: 12 500 B (14 runs) and 25 000 B (1 run). |
| Offered period | Fixed 0.1 s everywhere. |
| Frozen action payload range | **6 229 B – 3 568 326 B** (72-action FCOS catalogue, `rl_agent/splitfusion_action_catalog_v1/splitfusion_72_action_catalog.json`, `zstd_median_bytes`; action 71 `split_ae32_uint4_q9800` to action 0 `split_noae_uint8_q0000`). |
| Run-3 modeled support | **6 423 B – 427 605 B** (`rl_agent/splitfusion_hybrid_sac_v1/modeled_smoke_support.py:60`). |
| Retained offered payload vs range | 12 500 B and 25 000 B are **inside** the low end of both the action range and the Run-3 support. |
| Why still insufficient | Only **two** sizes; payload **fixed within every run**; no action-conditioned queue transition inferable. |
| Radio-condition variation | **Yes** — commanded noise −4.0 → −2.0 dB, gNB SNR P50 5.0 → 50.5 dB, MCS 7 → 28. |
| Enqueue-instant evidence | **Absent in 15/15 runs.** |

The runs *do* vary radio conditions well, and they *do* show real backlog build-up (P99 up to
160 KB, max 247 KB under `minus2p0`). What they never vary is the **payload**. A fixed-rate generator
cannot identify how 12 modes and a continuous `q` affect future backlog. The retained traffic is
*inside* the low end of the action range rather than below it, so the blocker is not that the
payloads are unreachably small — it is that **payload never moves**: two fixed sizes, each constant
within its run, never paired with an observed queue transition. These runs characterize BSR under a
fixed traffic generator; they do not identify

```
(previous backlog, SplitFusion action/payload, radio service) → (next backlog, delay/failure)
```

Had the ordering question been settled, these runs would have qualified as
`ADEQUATE_FOR_CARRIER_AND_NORMALIZATION_AUDIT_ONLY`. Both blockers hold, so the stricter status
applies.

---

## 9. Normalization saturation

The frozen spec is `bsr_log1p_scale = 1.0` (`empirical_contextual_environment.py:309`), applied by
`state_reward_transition_contract.py:5827` as `clip(log1p(bsr_bytes) / scale, 0, 1)`.

**With scale 1.0 the feature saturates at 2 bytes.** `log1p(x) ≥ 1` for `x ≥ e − 1 ≈ 1.718`, so every
integer byte count from 2 upwards maps to exactly 1.0. Only `0 → 0.0` and `1 → 0.693` are
distinguishable.

Measured over all 15 runs:

| | |
|---|---:|
| Nonzero backlog samples | 324 790 |
| Of those, mapped to exactly 1.0 | **324 790** |
| **Saturated fraction of nonzero** | **1.000000** |
| Distinct output values, every run | **2** |

The feature is not a queue measurement. It is a **zero / nonzero indicator**, and the 12 751-byte
queue in the clean run is indistinguishable from the 247 827-byte queue in `sequence_03`.

### Candidate diagnostics — all `PROVISIONAL_NOT_FROZEN`

Shown for the most congested run, `sequence_01_rep_01_minus2p0` (74 748 ticks, 66.9 % nonzero):

| Candidate | scale | saturated fraction of nonzero | distinct outputs | zero preserved |
|---|---:|---:|---:|---|
| `DEPLOYED_bsr_log1p_scale_1.0` *(currently frozen)* | 1.0000 | **1.0000** | **2** | yes |
| `CANDIDATE_log1p_engineering_bound_8192B` | 9.0110 | 0.5949 | 1 220 | yes |
| `CANDIDATE_log1p_nonzero_P95_13583B` | 9.5166 | 0.0618 | 1 688 | yes |
| `CANDIDATE_log1p_nonzero_P99_184759B` | 12.1268 | 0.0109 | 1 927 | yes |
| `CANDIDATE_log1p_run3_support_max_427605B` | 12.9660 | **0.0000** | 1 996 | yes |
| `CANDIDATE_log1p_action_max_payload_3568326B` | 15.0876 | **0.0000** | 1 996 | yes |

Every candidate maps 0 → 0.0 exactly, so zero is preserved throughout. Percentile anchors are
computed over **nonzero** values only, so a 99 %-zero run cannot produce a degenerate scale of 0.

**No scaler is selected.** Freezing one requires a resolved causal source, a declared train-only
split, and payload coverage that *spans* the frozen action range — **none of which hold**. Note that
the per-run P95/P99 anchors differ by an order of magnitude across runs (13 581 B vs 12 751 B vs
184 759 B), so a scale fitted on today's evidence would be fitted to the *generator*, not to
SplitFusion. `synthetic_contract_environment.py:905` already uses a different, unreconciled scale of
`log1p(8192)`; that divergence should be resolved at the same time, not before.

---

## 10. Retraining implication

**May measured BSR enter the learned Hybrid-SAC state now? No.**

| Item | Decision |
|---|---|
| `NRUE_MAC_BSR_STATUS` bytes or indices as policy state | **Prohibited.** Post-multiplex residual; using it leaks the action into its own observation. Enforced by `assert_no_action_leakage`. |
| `GNB_MAC_UL_MCS_DECISION.estimated_ul_buffer` as policy state | **Prohibited.** gNB-only, and a quantized echo of the same contaminated report. |
| `NRUE_MAC_RLC_BUFFER_STATUS` as policy state | **Not yet.** Correct source type and passes the leakage check, but pre-enqueue ordering is unproven. Admissible only after §11. |
| BSR as an **external guard** (outside the learned state, e.g. a shield or admission threshold) | **Permissible**, with the zero semantics of §2 stated at the call site and no claim that zero means zero delay. This is the recommended interim position. |
| Re-freezing `bsr_log1p_scale` | **Not now.** Fix only after the source is resolved. |
| gNB SNR / PUSCH / `avg_snr_x10` in UE state | **Prohibited** as UE-observable. |

**Fields that must not be fabricated under any circumstance:** `bsr_bytes` (no synthetic or random
values, no model-generated backlog), the pre-enqueue ordering flag, any application-enqueue
timestamp, any UE-side uplink SNR, any clock bridge between the sender log and the tracer, and any
normalization constant presented as fitted.

**Recommended interim posture for the next retraining:** leave the learned state unchanged. Keep
`bsr_bytes = 0` with its existing `INERT_FOR_EXACT_GENESIS_ZERO_ONLY` provenance — it is honest about
being a placeholder. Do **not** substitute a measured-looking value from these traces; that would
import the generator's queue behaviour and the action leakage in one step.

**This audit does not show that BSR evidence would improve the policy.** It shows only that the
currently available evidence cannot establish the question either way.

---

## 11. Minimal future calibration design (specified, **not run**)

Smallest experiment that answers what this audit could not. A new 288-cell campaign is **not**
justified: the open questions are ordering and payload span, both reachable with a handful of cells.

**Scale.** A few registered SplitFusion actions spanning small / medium / large payload, under at
least two contrasting radio conditions. Concretely, 3 payload rungs × 2 channel conditions × 2
repetitions = **12 cells**, comparable in cost to one existing rung sweep.

**Payload rungs.** Chosen from the registered catalogue so they span the action range rather than
sitting below it — e.g. the 49.4 KB floor, the ~90 KB seg-safe knob, and a ~400 KB rung. Offered load
must **vary within a run**, not only across runs, so backlog evolution is observed through a payload
change.

**Channel conditions.** Reuse the already-calibrated commanded-noise rungs, e.g. a clean control and
a `minus2p5`-class degraded condition, both of which this evidence shows produce distinguishable
backlog regimes.

**Required additional instrumentation** — this is the part that does not exist today:

| Timestamp | How |
|---|---|
| Policy-state sampling instant | Application-side, monotonic |
| **Pre-enqueue RLC/BSR read** | Explicit UE-runtime read *before* the payload is handed down |
| **Payload enqueue instant** | Enable `NR_PDCP_TX_SDU` / `NR_RLC_TX_SDU` extraction (already defined in `T_messages.txt` with monotonic timestamps — extraction just needs turning on) |
| RLC→MAC dequeue | `NR_RLC_TX_DEQUEUE` |
| Grants and transmitted bytes | `NRUE_MAC_DCI_GRANT`, `UE_PHY_UL_PAYLOAD_TX_BITS` (already retained) |
| Next backlog | `NRUE_MAC_RLC_BUFFER_STATUS` (already retained) |
| Feature completion or failure | Existing SplitFusion result/deadline accounting |

**Clock discipline.** Record a **measured** bridge between the application clock and the tracer
clock, rather than reconstructing a date. The monotonic-timestamp T events exist precisely for this.

**Exit criteria.** The calibration succeeds if it establishes (a) that a UE-runtime backlog read can
be placed strictly before payload enqueue for the same decision, and (b) that backlog evolution
responds measurably to payload across the registered range. Only then may a scaler be fitted, on a
declared train-only split.

**Note the T-tracer rebuild constraint:** per `CLAUDE.md`, the tracer byte-compares `T_messages.txt`
against the compiled copy, so **both softmodems must be rebuilt** if the message file is edited.
Enabling extraction of already-declared events should avoid an edit; verify before launching.

---

## 12. Limitations and explicit non-claims

**Limitations**

1. All 15 runs are from a single day (2026-08-21) and five closely related experiment families. No
   independent replication across environments.
2. Every run is RFsim on one host. Nothing here bounds over-the-air behaviour.
3. Single UE. No contention.
4. The UE↔gNB shared clock domain rests on both softmodems running on one host with `CLOCK_REALTIME`
   at the `T()` call site. Correct here; it does **not** generalise to a distributed deployment.
5. Tracer timestamps are `CLOCK_REALTIME`, which is not monotonic. No NTP step was observed, but none
   was excluded either.
6. `20260821_meeting_smoke_04` has a header-only TX-bits file; `rung_04_minus2p0` did not reach its
   commanded degradation.
7. The SplitFusion payload range is taken from the knob matrix as an authority; this audit did not
   re-derive the catalogue.
8. Correlations in §6.3 are descriptive, on a fixed generator, and are reported to bound what the
   evidence can say, not to support a mechanism.

**Explicit non-claims**

- **Not claimed:** that existing BSR evidence improves, or would improve, the policy.
- **Not claimed:** that `NRUE_MAC_RLC_BUFFER_STATUS` *is* a valid pre-action source. It is the
  surviving candidate; its qualification is `UNRESOLVED`.
- **Not claimed:** that zero BSR means zero network delay, an idle link, or an unloaded scheduler.
- **Not claimed:** that SNR, MCS and BSR are interchangeable, or that gNB-observed SNR is UE-visible.
- **Not claimed:** that a specific normalization scale is correct. Every candidate is
  `PROVISIONAL_NOT_FROZEN`.
- **Not claimed:** that the retained runs falsify an action-conditioned queue model. They cannot
  test it.
- **Not claimed:** any result about multi-UE contention, over-the-air transport, or Phase-2 work.

---

## 13. Artifact and hash inventory

**Deliverables** (this directory, additive; no existing file edited):

| File | Lines | SHA-256 |
|---|---:|---|
| `audit_ue_state_evidence.py` | 1 921 | `cd233e0e2a350d64caf487f2973033b1d989a1c41b3a3088e82859620c8a1191` |
| `test_audit_ue_state_evidence.py` | 883 | `6a2806bcad4e92e05718aafc592b56daa772fadaf2493aa602e1ac330e8b7817` |
| `UE_STATE_EVIDENCE_AUDIT.md` | — | this document |

**Evidence read (read-only): 15 logical runs, 90 files.** The two decision-bearing traces per run:

| Run | `NRUE_MAC_RLC_BUFFER_STATUS.csv` rows / SHA-256 | `NRUE_MAC_BSR_STATUS.csv` rows / SHA-256 |
|---|---|---|
| `ue_n2_oai_ul_calibration_smoke_v1/20260821_meeting_smoke_04` | 36837 / `9599ee2fd3eae74d77b234321f3aa3e36a3e59359f6c383bd98230b3d2a96b47` | 2093 / `6c70bce3232e6d5c673b4812e9df22d6e3785d44c2914841055b5b82b56ff968` |
| `ue_n3_oai_ul_command_calibration_v1/20260821_command_search_02/rungs/rung_00_minus4p0` | 83136 / `ecd821dd7d2dfdd36d8c95d80d7eb39b033667cbb6b834e31c22a807b0608389` | 3125 / `7f7067efcb855ad504f29d5a26e1643fb184630d69ca65a5950537dd03720a43` |
| `ue_n3_oai_ul_command_calibration_v1/20260821_command_search_02/rungs/rung_01_minus3p5` | 81438 / `6d2bb69fa7b50d328697faeead9d5caedc80ea701dd037f90a907329abc203a6` | 3065 / `135a704bdd672af05223bc5ab1ac371a782b7be34161d29cb70f9e67487bba78` |
| `ue_n3_oai_ul_command_calibration_v1/20260821_command_search_02/rungs/rung_02_minus3p0` | 81819 / `b9fd2e402de9fd1c7922cc4a1ac959f7439f9759c9feeb0b4be1b949a40ee189` | 2962 / `da0d2c69fc0297cf793498a861a7810364d75ad1c11ecceb3cf49ed9ab9c9aa3` |
| `ue_n3_oai_ul_command_calibration_v1/20260821_command_search_02/rungs/rung_03_minus2p5` | 81006 / `efd5cf7f5ac975948b3c8a055e7f6317839a6af3a021addf5463b7dd90ae172d` | 2901 / `d6e5b2b9731fe72e38a5b81ce0adcbb60350450644d22c82b950ffd6d98cd47a` |
| `ue_n3_oai_ul_command_calibration_v1/20260821_command_search_02/rungs/rung_04_minus2p0` | 78891 / `1d328dadf3ff3ede6f8935e211e32c23a7574a582fc4298b09c84b10c4d99404` | 2136 / `a84b4f0f08cd95bfc3c485499e10300fad52a2de4fe416e015217446a08818c7` |
| `ue_n3_oai_ul_live_stage_v1/20260821_clean_control_03` | 186300 / `2eabc56aebdd9bd38390e232310295747e3a15ba065fac2f931ea49222cb2094` | 5415 / `ab62ca5ae54313df578b1a912a3a64f7dbc5205ce032536175d606b1f4169e2e` |
| `ue_n3a_oai_ul_sustain_replication_v1/20260821_live_02/repetitions/sequence_00_rep_01_minus2p5` | 226869 / `f8d86454b5dbc49831fc9268d2df2d025454097808fca536ce289d912742dd3d` | 10231 / `2e75f1218ad87d9edaeb122b66cbf79c7a6080b3271a09f1a71fb7f84356560a` |
| `ue_n3a_oai_ul_sustain_replication_v1/20260821_live_02/repetitions/sequence_01_rep_01_minus2p0` | 224244 / `00dbfd9718c4ff74d7d7c3d74e643c5ce39c201a766121e0de40ccf8a14f8951` | 11111 / `73af68c50912fa890f6ee2e6fea5d9fc9e3332bad264bb54a72a9338e29de6af` |
| `ue_n3a_oai_ul_sustain_replication_v1/20260821_live_02/repetitions/sequence_02_rep_02_minus2p5` | 225216 / `fb7462447180da7a5a889ab2089463a596a7e1cffd3d54bc1b354706b9a15d25` | 10320 / `cb57439fd32cf9ec43eb26b18a59ad5c7e45c4bcab183768093c501b367b4d8a` |
| `ue_n3a_oai_ul_sustain_replication_v1/20260821_live_02/repetitions/sequence_03_rep_02_minus2p0` | 224541 / `f5babe85ce1b60c266627f745dd55880c745a1bc99ab4d1b4941281eed267a34` | 11182 / `37eddeb4921fb0a203d985680796edbd523b7d66dcad624df4ffa94ef90ae1ab` |
| `ue_n3a_oai_ul_sustain_replication_v1/20260821_live_02/repetitions/sequence_04_rep_03_minus2p5` | 225366 / `166e4cebf1a9fc789c8a03d0fe05227f6a5a44ed5f8e0b6c70d04b3e74b0c4c3` | 10238 / `c5edc633528e4b926c9500e180447837872a84e1dca1ab27b2db60b11a62a047` |
| `ue_n3a_oai_ul_sustain_replication_v1/20260821_live_02/repetitions/sequence_05_rep_03_minus2p0` | 223794 / `0134cda5ad06096cd0d24619d1f3251dbf1d0cb9db05fc70b8cb76a828b0cd03` | 11007 / `5db9b2eb55e8a6c322743675f85f44c33ba43d605c3358a3b9ce6e866bab3ca2` |
| `ue_n3c_oai_ul_cold_attach_refinement_live_v1/20260821_live_01/repetitions/rep_01_minus3p0_cold` | 200574 / `1e146a9063c8036b31e633f07117b3d0d273306bd7388a6c22c500a8dc4e2196` | 9312 / `928bd8dee4de8e83d02e4dd5e25a9437febc9098fbfc51137bb7b0bf8193e1a5` |
| `ue_n3c_oai_ul_cold_attach_refinement_live_v1/20260821_live_01/repetitions/rep_03_minus3p0_cold` | 198549 / `a6e65dc7451cfcb0e08c6176be1a0cb6240533b54dfe17775acbd65584b091da` | 9283 / `18bec01a6957c76ca159a5b518e13e1fbacf894bc5a50634af9db3948b7ac0f8` |

The **complete 90-file inventory** — relative path, size, SHA-256, row count and exact header for
every file used, including the gNB and traffic companions — is emitted by the tool and is
regenerable deterministically:

```bash
python3 rl_agent/splitfusion_hybrid_sac_ue_state_audit_v1/audit_ue_state_evidence.py \
    --evidence-root rl_agent/experiments \
    --json-out <path outside the evidence tree>
```

Repeated runs over unchanged bytes produce byte-identical JSON (asserted by
`DeterminismTests.test_repeated_audits_of_the_same_bytes_agree_exactly`). The JSON report was written
to a session scratch directory, never into `rl_agent/experiments/`.

**Source files read but not modified:** `nr_ue_scheduler.c`, `T_messages.txt`, `T.h`, `tracer/csv.c`
(OAI submodule); `state_reward_transition_contract.py`, `empirical_contextual_environment.py`,
`synthetic_contract_environment.py` (Hybrid-SAC); `splitfusion_action_catalog_v1/splitfusion_72_action_catalog.json`;
`splitfusion_hybrid_sac_v1/modeled_smoke_support.py`.

---

## 14. Worktree and execution safety

**Worktree**

- Starting HEAD `e1f600ac4a2794dbfd22700eb156d6e2c56b6f28`, branch `master`.
- All work confined to the new directory `rl_agent/splitfusion_hybrid_sac_ue_state_audit_v1/`, which
  did not previously exist.
- No existing file edited; `__init__.py` untouched; no file staged that was already dirty.
- The 12 pre-existing dirty and untracked paths were hashed at start and re-verified at finish, all
  unchanged. The dirty `OAI/openairinterface5g` submodule pointer was left as found.
- No `git reset`, `clean`, `stash`, checkout restoration, `pull`, `merge`, `rebase` or `push`.

**Execution safety**

- No CARLA, OAI, RFsim, Docker, map server or model inference launched.
- No CUDA context initialised. `nvidia-smi` shows no compute process belonging to this session; the
  test suite asserts `torch` is never imported.
- No RL training launched or resumed.
- No experiment evidence altered: `RealEvidenceSmokeTests.test_auditing_a_run_leaves_the_evidence_byte_identical`
  hashes every CSV in a run before and after auditing it and requires exact equality.
- The audit module is import-side-effect free. `ImportPurityTests` imports it in a subprocess with
  `open`, `os.scandir`, `os.listdir`, `os.walk`, `os.system`, `glob`, the `pathlib.Path` I/O methods,
  `subprocess.Popen/run` and `socket` all replaced by raising guards, and asserts none fires.
- Standard library only.

**Verification run**

```
python3 -m py_compile audit_ue_state_evidence.py test_audit_ue_state_evidence.py   → clean
python3 -m unittest test_audit_ue_state_evidence                                  → 77 tests, OK
git diff --check                                                                  → clean
audit --evidence-root rl_agent/experiments                                        → 15 runs, 11.9 s
```

---

## 15. Corrections applied (audit revision 2)

This revision narrowly corrects four defects in the first release of this audit. The **verdict is
unchanged** — `INSUFFICIENT_OR_CAUSALLY_UNRESOLVED`, recommended source `UNRESOLVED`, 0 of 15 runs
qualified — but three of the four defects were reasons the audit could have been *wrong in the
permissive direction*, and one was a factual error about the payload range.

### 15.1 Payload authority rebound to the frozen FCOS catalogue (A1)

The audit quoted the legacy PERMODEL knob-matrix range **49 400 B – 2 835 000 B**. That range is
retired. The current authority is the frozen 72-action FCOS catalogue, **6 229 B – 3 568 326 B**
(`zstd_median_bytes`, action 71 `split_ae32_uint4_q9800` → action 0 `split_noae_uint8_q0000`), with
the Run-3 contextual surrogate fitted on the narrower support **6 423 B – 427 605 B**.

The practical consequence is a reversed factual claim. Under the legacy constants the retained
12 500 B and 25 000 B traffic points were reported as sitting **below** the action range; under the
correct authority they sit **inside its low end**. The runs remain insufficient, but for the honest
reason: only two sizes, fixed within each run, never paired with an observed queue transition.

`verify_payload_authority()` re-reads both authoritative files and refuses to proceed if either has
drifted, so these constants are checked rather than trusted. The retired pair is retained only as
`RETIRED_PERMODEL_PAYLOAD_BYTES` so that a test can assert it never returns.

### 15.2 Future-qualification logic repaired (A2)

The previous implementation qualified a causal queue source from **filename existence**
(`{path.stem for path in ue_csv_dir.glob("*.csv")}`). A zero-row, wrong-schema, wrong-run or
dequeue-only file would have qualified it, and a single qualified run qualified **all** runs
globally via `any(...)`.

Qualification now requires, per run and in order: file present; exact schema; non-empty parsed
records; ingress (not dequeue) semantics; same-run identity; UE/RNTI identity that actually
intersects the backlog trace; bearer/LCID identity; decision and frame identity; a source timestamp
*and* an availability timestamp; and a demonstrated ordering

```
t_measure <= t_available_to_agent <= t_state_commit < t_action < t_current_payload_enqueue
```

for **every admitted decision**, not on average. A missing instant fails closed — it is never read
as zero and never forward-filled. `CoverageVerdict.ACTION_CONDITIONED` is now unreachable while any
run's ordering is unresolved, so payload variation alone can no longer promote the verdict. The
global pre-action status is promoted only when **every** run qualifies.

`NR_RLC_TX_DEQUEUE` has been moved out of the enqueue candidates into `SERVICE_DEQUEUE_EVENTS`.

### 15.3 The post-multiplex BSR statement made precise (A3)

The earlier text implied `NRUE_MAC_BSR_STATUS` is *always* invalid. The precise position:

* A post-multiplex BSR generated **after** the current action **must not** be aligned as that
  action's own pre-action state — its bytes are what remained once the current payload had already
  been multiplexed into the current grant, so the action would enter its own observation.
* A **correctly lagged** BSR **may** legitimately describe a *previous* action. That use is not
  refused here; it requires the lag to be proven rather than assumed.
* Either way, pre-multiplex `NRUE_MAC_RLC_BUFFER_STATUS` remains the **preferred** candidate: it is
  more direct (true RLC occupancy in bytes, read before the grant is filled) and less quantized
  (unquantized bytes, against coarse BSR-table indices).

`assert_no_action_leakage(source, aligned_to_same_action=...)` now encodes exactly this distinction.

### 15.4 UE-side SNR candidate registered (A4)

Four quantities are routinely conflated; they are now kept apart by link direction, observer, and
whether the number is a standardized index.

| Quantity | Direction | UE-visible at runtime | Units | Standardized index |
|---|---|---|---|---|
| `UE_PHY_MEAS.snr` | **UE downlink receive** | yes | integer dB (**not** ×10) | no |
| `UE_PHY_MEAS.w_cqi` | UE downlink receive | yes | integer dB | **no** |
| CSI-RS CQI | UE downlink receive | only if CSI-RS configured | index 0–15 | **yes** |
| `GNB_MAC_PUSCH_POWER_CONTROL.snrx10` | **gNB uplink receive** | **no** | dB ×10 | no |

Source-code provenance, cited to exact lines in this worktree:

| Fact | Citation |
|---|---|
| `UE_PHY_MEAS` message and field order | `common/utils/T/T_messages.txt:1510-1513` |
| Event emission | `openair1/SCHED_NR_UE/phy_procedures_nr_ue.c:411` (guarded by `#if T_TRACER`) |
| Emission gate | same file `:403,409` — only when `l == 2` **and** `nr_slot_rx == 0` |
| Only call site | same file `:529`, inside the PDSCH channel-estimation path |
| `snr` expression | same file `:415` — `rx_power_avg_dB[0] - n0_power_avg_dB` |
| `rx_power_avg_dB` / `n0_power_avg_dB` | `openair1/PHY/NR_UE_ESTIMATION/nr_ue_measurements.c:99-100` |
| `w_cqi` expression | same file `:102` |
| RSSI | same file `:103-106` |
| CSI-RS CQI table | `openair1/PHY/NR_UE_TRANSPORT/csi_rx.c:666-695` |
| CSI-RS CQI population | same file `:926,958` |

Three findings follow directly from those lines and matter for any later use:

1. **`w_cqi` is not a CQI index.** `nr_ue_measurements.c:102` computes
   `wideband_cqi_avg = rx_power_avg_dB - n0_power_avg_dB` — the *identical* expression to the `snr`
   field emitted at `phy_procedures_nr_ue.c:415`. In this build the two fields therefore carry the
   **same number**. `w_cqi` is diagnostic only. The only standardized 0–15 downlink CQI is the
   CSI-RS path, which is conditional on CSI-RS measurement/reporting actually being configured and
   has no T-tracer event of its own.
2. **Cadence is at most one sample per 10 ms frame**, because the emit is gated on `nr_slot_rx == 0`.
   That is 10× the 10 Hz policy rate, so it is not *a priori* too slow — but it is an upper bound.
3. **Availability is conditional on downlink traffic.** The only call site sits in the PDSCH
   channel-estimation path, so with no PDSCH allocation to this UE in slot 0 the event simply does
   not fire. Any qualification of this signal must therefore run a sustained downlink stream, and
   must report coverage rather than assume it.

**`UE_PHY_MEAS.snr` is a UE receive-side *downlink* measurement derived from the downlink channel
estimates. It must never be called "uplink SNR."** The gNB's PUSCH SNR is an uplink measurement that
the UE cannot observe at runtime. `assert_direction_not_mislabelled()` and `assert_ue_observable()`
enforce both statements, and no candidate may be rendered with the bare label "SNR".

**Retained evidence contains no `UE_PHY_MEAS` rows.** Verified: all 15 retained runs hold exactly
four UE CSVs each (`NRUE_MAC_BSR_STATUS`, `NRUE_MAC_RLC_BUFFER_STATUS`, `NRUE_MAC_DCI_GRANT`,
`UE_PHY_UL_PAYLOAD_TX_BITS`); `**/ttracer/ue/csv/UE_PHY_MEAS.csv` matches nothing. The expectation
held. Qualifying this signal therefore requires a **new** bounded measurement, not a re-read.
