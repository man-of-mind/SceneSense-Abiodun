# Preregistration — UE-local `[previous UL MCS, pre-enqueue backlog]` qualification (v2, amended)

**Status: amended design, not yet launched.** Version 1 is withdrawn; see §1.

## 1. Why v1 was withdrawn

v1 ran 4 profiles × 3 fixed load tiers = 12 cells, with `tagged_sender.py` holding **one payload
constant for every frame of a cell**. Load was therefore a purely *between-cell* factor. Because the
runner also rebuilds the RAN from cold between cells, any backlog difference across tiers was
confounded with everything else that differs between two radio instances — attach state, scheduler
warm-up, host scheduling. **Backlog could not have been causally attributed to a load change.** That
is the same across-run confounding the project has hit before, and the plan is withdrawn.

## 2. Amended design

Load is now a **within-cell** factor. Every cell runs all three tiers as back-to-back blocks on one
continuous 10 Hz monotonic timeline, inside a single radio instance.

```
3 block orders  ×  2 contrasting channels  ×  2 repetitions  =  12 cells
```

| Factor | Levels |
|---|---|
| Block order | cyclic Latin square: `(L,M,H)`, `(M,H,L)`, `(H,L,M)` |
| Channel | `FAVORABLE_STABLE`, `ADVERSE_STABLE` (both *stable*, so the channel is held while load moves) |
| Repetition | rep 0 = the order; rep 1 = its **exact reverse** |

- 150 decisions per block, 3 blocks, **450 decisions per cell**, 45 s of traffic.
- 12 cells × 450 = **5,400 tagged decisions**.
- Cell *execution* order is shuffled from recorded seed `20260924`, so host/radio drift cannot be
  mistaken for a design effect. Shuffling changes only the order, never the design.

### Balance, proven not asserted

`contract.audit_cell_plan()` computes these, and `DesignMatrixTests` asserts them:

| Property | Result |
|---|---|
| Cells | 12 = 3 × 2 × 2 |
| Every cell contains all three tiers | **yes** (load is within-cell) |
| Position balance per channel | each (tier, position) appears exactly **2×** (9 combinations) |
| Ordered transitions per channel | all **6** of `L→M, M→H, H→L, H→M, M→L, L→H`, each exactly **2×** |
| Repetition 1 reverses repetition 0 | **yes**, for every (channel, order) |
| Latin square | each tier appears once in each position across the 3 orders |

Repetition 0 supplies `{L→M, M→H, H→L}`; its reverse supplies `{H→M, M→L, L→H}`. Transition
*direction* is therefore counterbalanced, so a transient measured after `M→H` is not confounded with
the direction of the only transition that produced it.

### Transitions are sharp

One sender process walks the whole 450-decision timeline. At a block boundary the payload size and
destination port change on the very next scheduled frame — no process restart, no gap. Each block
has its own receiver **only** because the production receiver is constructed with a single
`expected_chunks_per_frame`; all three receivers are up before the first datagram.
`LoopbackBlockTransitionTests` asserts the inter-decision step across a boundary is not larger than
a step inside a block, which is what a restart would betray.

## 3. Pinned quantities

| Tier | Action | Catalogue profile | Payload | Chunks | Offered @10 fps |
|---|---:|---|---:|---:|---:|
| low | 71 | `split_ae32_uint4_q9800` | 6,229 B | 1 | 0.50 Mbps |
| medium | 50 | `split_ae64_uint4_q5000` | 263,507 B | 5 | 21.08 Mbps |
| high | 30 | `split_ae128_uint4_q0000` | 880,567 B | 15 | 70.45 Mbps |

Resolved from `splitfusion_72_action_catalog.json` (sha256 `07e0690f…`), and refused rather than
guessed if the remembered action ids do not reconcile. Against a ~6 Mbps uplink, medium and high are
**deliberately saturating**: the design needs a backlog range, and the queue response to entering and
leaving saturation is the effect under test.

The same tier uses byte-identical payload in every block and every cell (seeded per action id,
sha256 recorded), so payload is never a hidden variable.

Channels and their 450-sample trace prefixes resolve from the registered binding and trace table;
every value inside the prefix is the registered value. Only the prefix length is bounded.

## 4. State under test

| Feature | Definition | Source |
|---|---|---|
| `previous_ul_mcs` | latest **strictly prior** UE-decoded **round-0** UL DCI MCS | `NRUE_MAC_DCI_GRANT`, `direction=1`, `round=0` |
| `pre_enqueue_backlog_bytes` | raw bytes from the last `NRUE_MAC_RLC_BUFFER_STATUS` tick **strictly before** the decision | pre-multiplex, per-LCID sum |

Because the backlog sample precedes the enqueue, the tagged payload's own bytes are **structurally**
excluded. Observation **age is not a policy feature** — it is an external validity gate only.

Missing stays missing: an MCS older than **200 ms** is recorded `MISSING_STALE`, never forward-filled;
"no prior grant" is `MISSING_NO_PRIOR_GRANT`. Neither is ever coerced to 0, which is a real
modulation index (asserted by `test_mcs_zero_is_preserved_as_a_real_observation`).

## 5. Clock domain

`NR_PDCP_TX_SDU` carries **both** the tracer's `CLOCK_REALTIME` header and an in-payload
`CLOCK_MONOTONIC` pair taken at the same call site (`nr_pdcp_oai_api.c:941-944`). Those rows are a
**measured same-event bridge** from tracer time to monotonic, not a reconstructed date. The sender
and the production receiver both stamp `CLOCK_MONOTONIC`, which is system-wide on Linux. The bridge's
residual spread is reported so join precision is visible rather than trusted.

## 6. Provenance to verify offline

gNB traces are used **only** to confirm the UE-observed MCS has the claimed provenance, never as a
runtime input. The binding is to `final_mcs`, because OAI may legitimately adjust the initial
`selected_mcs` through later scheduling/PHR constraints. The UE DCI MCS must match `final_mcs`;
`selected_mcs != final_mcs` is retained and reported as a diagnostic, not treated as a failure.
`SCENESENSE_MCS_POLICY=sinr` gates `get_mcs_from_SINRx10` at `gNB_scheduler_ulsch.c:2027-2028`;
`get_mcs_from_SINRx10` supports MCS table 0 only (`gNB_scheduler_primitives.c:238-241`), and the
table id is recorded per decision and asserted constant across cells.

## 7. Analysis plan

1. **Transient response** — first `30` decisions (3 s) after each transition, by transition type and
   channel; backlog rise/decay and MCS response. Direction is counterbalanced, so `L→H` and `H→L` are
   separately estimable.
2. **Steady-state separation** — last `80` decisions of each block (disjoint from the transient
   window: 30 + 80 ≤ 150). Tier separation in backlog; MCS stability across tiers at fixed channel.
3. **Repeatability** — rep 0 vs rep 1 agreement per (channel, order), and across the two orders that
   share a transition.
4. **Age / missingness** — per cell and per block: MCS observed / stale / no-prior counts, backlog
   coverage, age distributions.
5. **Prediction** — feature sets **A** (MCS only), **B** (backlog only), **C** (both) against
   next-frame complete reassembly, uplink latency, and the transport portion of the 170 ms budget.
   Simple interpretable models only; **no neural policy is trained**. Splits are blocked by
   (cell, block) — never random individual frames — and uncertainty is reported across blocks and cells.

Only the uplink transport segment is charged against the budget. No perception model runs here, and
folding in an unrelated constant would make the outcome a statement about that constant.

## 8. Gates

12/12 cell identity and profile read-back · exact sent-to-terminal accounting · no negative interval ·
no cross-cell queue contamination (RAN rebuilt from cold per cell, so the RLC queue cannot survive) ·
every joined observation precedes its decision · retransmissions excluded · missing MCS never zero ·
raw backlog retained with no `log1p` saturation · MCS table constant across cells · RF restored to
`noise_power_dB=-50` and read back after failure and final teardown · no orphan process · CARLA and
CUDA untouched.

## 9. Verdict

Exactly one of `ACCEPT_MCS_BACKLOG_STATE`, `REJECT_MCS_BACKLOG_STATE`, `INCONCLUSIVE`. If accepted:
measured normalization bounds and a causal runtime sampling contract — **no agent change, no training.**

UL MCS is a delayed, quantized scheduler decision derived from gNB-measured uplink SNR in this custom
build, reaching the UE through standard DCI with no added controller. Backlog is demand/queue
pressure, not physical channel. This is a bounded engineering qualification, not publication evidence.
