# Preregistration — Run-4 near-capacity UE MCS / backlog sweep

**Status: registered, not launched.** Committed before any collection. Phase B
requires explicit authorization.

Contract id `ue_mcs_backlog_near_capacity_v1`. Claim boundary:
`BOUNDED_NEAR_CAPACITY_QUEUE_TRANSITION_CHARACTERIZATION_WITH_BLOCKED_VALIDATION_NOT_KERNEL_ACCEPTANCE_AND_NOT_PUBLICATION_EVIDENCE`.

## 1. Relationship to Run 3

This is a **follow-up to** `experiments/ue_mcs_backlog_calibration_v1/20260924_131015`,
never a replacement, rerun or reinterpretation of it.

- Run 3's verdict stays **`INCONCLUSIVE`**. Its bound corrected-v2 result is
  **4/7 scientific checks and 12/13 structural gates**. The earlier "5/7 and
  12/12" claim is superseded and is not repeated anywhere in this run.
- Five Run-3 files are protected and hash-guarded before and after Run 4:
  `manifest.json`, `INTEGRITY_AMENDMENT_DERIVED_V1_SUPERSEDED.md`,
  `VERIFIER_INPUT_MANIFEST_V2.json`, `analysis_v2.json`, `decisions_v2.csv`.
  Digests are pinned in `protected_evidence.py`; a difference is a stop
  condition, never something to repair.
- Run 3's own report asked for exactly this experiment in §10: *"a bounded
  near-capacity sweep — offered loads bracketing the channel-dependent
  capacity ... testing whether MCS adds incremental value once backlog is no
  longer pinned by gross over-offer."*

Nothing in the Run-3 package is edited. Its runner, sender, receiver, causal
join, route proof and teardown are **imported unchanged**; their digests are
recorded in this run's manifest.

## 2. Immutable action authority — verified, not assumed

| Source | SHA-256 | Verified |
|---|---|---|
| `splitfusion_72_action_catalog.json` | `07e0690f…cac54c3` | yes |
| `splitfusion_72_action_catalog.csv` | `0512cb39…6bb6fbb` | yes |
| shared AE32 decoder checkpoint | `e2f86775…0693d1f271` | yes, all three actions |

| Tier | Action | Profile | Payload | Chunks @60,000 B | Offered @10 fps |
|---|---:|---|---:|---:|---:|
| low | **70** | `split_ae32_uint4_q9000` | 28,109 B | 1 | 2.25 Mbps |
| medium | **69** | `split_ae32_uint4_q7000` | 81,087 B | 2 | 6.49 Mbps |
| high | **68** | `split_ae32_uint4_q5000` | 129,707 B | 3 | 10.38 Mbps |

Bound by **action id** against the frozen catalogue, then cross-checked against
independently recorded profile ids, payload bytes, derived chunk counts,
offered rates and the shared checkpoint. Any one mismatch refuses the run
rather than guessing which side is right (`contract.resolve_load_tiers`; eight
refusal paths are unit-tested).

All three actions are `transport_valid` and `agent_action_enabled` in the
catalogue. All three share one AE32 decoder, so payload is varied by quantizer
`q` alone and decoder identity is not a hidden variable.

## 3. Capacity premise — corrected before collection, not after

The Run-3 preregistration sited its tiers against "a ~6 Mbps uplink". **Run 3's
own data contradicts that number**, and this must be settled before tiers are
defended as "near capacity".

Re-derived read-only from `decisions_v2.csv` (the protected file is unchanged),
taking service = (payload enqueued − backlog delta) / Δt over intervals where
the queue was already deeper than 2 MB and below 85% of the 49,984,583 B
ceiling — i.e. continuously backlogged, so the measured rate is capacity, not
offered load:

| Channel | P10 | **P50** | P90 | n |
|---|---:|---:|---:|---:|
| `ADVERSE_STABLE` | 10.49 | **12.05** | 22.57 | 1,200 |
| `FAVORABLE_STABLE` | 18.11 | **42.13** | 46.50 | 998 |

Independent corroboration that does not use the slope estimate at all:
`FAVORABLE_STABLE` sustained a **21.08 Mbps** offered load with **median backlog
0** and 0.960 complete reassembly. A 6 Mbps link cannot do that.

**Consequence for this design, stated in advance:**

| Channel | low | medium | high |
|---|---:|---:|---:|
| `ADVERSE_STABLE` (P50 12.05 Mbps) | 0.19× | 0.54× | **0.86×** |
| `FAVORABLE_STABLE` (P50 42.13 Mbps) | 0.05× | 0.15× | 0.25× |

- The **`ADVERSE_STABLE` arm is the near-capacity arm.** It spans 0.19→0.86 of
  capacity and crosses the knee; action 68 at 10.38 Mbps sits inside the P10
  fluctuation band of adverse capacity (10.49 Mbps). This is the arm the
  experiment is actually about.
- The **`FAVORABLE_STABLE` arm is declared sub-capacity in advance.** All three
  tiers sit at 0.05–0.25 of capacity and are expected to drain with backlog at
  or near zero. It is retained as the channel contrast — the same offered load
  with one channel near its knee and the other idle is precisely where an
  exogenous channel signal should earn its place — but it is **not** expected to
  exercise the queue recurrence.

The degenerate-arm rule in §9 exists because of this, and is registered now so
the outcome cannot be rationalised later.

## 4. Design

Six payload permutations × two channels = **12 cells**, 3 blocks of 150
decisions at 10 fps, **450 decisions per cell**, **5,400 tagged decisions**.

Load is a **within-cell** factor, as amended in Run 3: one sender process walks
the whole 450-decision timeline inside a single radio instance, so a backlog
change is attributable to a load change that happened in the same radio
instance rather than across two cold rebuilds.

| Partition | Permutations | Transitions covered (per channel, ×2 each) |
|---|---|---|
| **FIT** | `L-M-H`, `M-H-L`, `H-L-M` | `L→M`, `M→H`, `H→L` |
| **VALIDATION** (blocked, held out) | `H-M-L`, `L-H-M`, `M-L-H` | `H→M`, `M→L`, `L→H` |

Proven by `contract.audit_cell_plan` and asserted in `test_near_capacity_v1.py`:
12 cells; 5,400 decisions; every cell contains all three tiers; each
(tier, position) exactly 2× per channel; all six ordered transitions exactly 2×
per channel; FIT and VALIDATION are 3+3 per channel; each partition is itself a
Latin square; every VALIDATION permutation is the exact reverse of its paired
FIT permutation.

**Declared limitation, not a later discovery:** the three cyclic orders and
their reverses partition the six ordered transitions into two *disjoint* sets.
Validation is therefore an **extrapolation across transition direction**, not an
interpolation. That is a deliberately conservative test; it also means a gate-4/5/6
failure may indicate direction extrapolation rather than an unusable kernel,
and the report must say which. `test_validation_extrapolates_across_transition_direction`
pins this property so it cannot be quietly forgotten.

Cell **execution** order is shuffled from pinned seed `2026092401`, recorded in
`plan.json`. Shuffling changes only execution order. **Cells never move between
partitions**, before or after seeing results (asserted).

Same tier ⇒ byte-identical payload in every block and every cell (seeded per
action id, digest recorded), so payload is never a hidden variable.

## 5. State under test

| Feature | Definition | Source |
|---|---|---|
| `previous_ul_mcs` | latest **strictly prior** UE-decoded **round-0 / new-data** granted/final table-0 MCS from UL DCI | `NRUE_MAC_DCI_GRANT`, `direction=1`, `round=0` |
| `pre_enqueue_backlog_bytes` | raw UE RLC bytes from the last `NRUE_MAC_RLC_BUFFER_STATUS` tick **strictly before** this payload's enqueue | pre-multiplex, per-LCID sum |

Because the backlog sample precedes the enqueue, the tagged payload's own bytes
are **structurally** excluded. Both are UE-local: no SRS, PUSCH SNR, RSRP, CQI
or gNB telemetry is read at runtime.

- **gNB selected/final MCS is verifier-only provenance and never an actor
  input.**
- **Real MCS 0 and real backlog 0 are preserved.** 0 is a real modulation index
  and an empty queue is a real state.
- **Missing stays missing.** No prior grant, a stale grant, or a missing backlog
  tick is recorded with an explicit marker, never forward-filled and never
  coerced to 0.
- Observation **age is a validity gate only**, never a policy feature.
- The hidden profile identity is **never exposed to the actor**.

## 6. Clock domain

`NR_PDCP_TX_SDU` carries both the tracer's `CLOCK_REALTIME` header and an
in-payload `CLOCK_MONOTONIC` pair taken at the same call site
(`nr_pdcp_oai_api.c:941-944`) — a **measured same-event bridge**, not a
reconstructed date. Sender and production receiver both stamp
`CLOCK_MONOTONIC`. **The bridge residual P95 must be ≤ 1 µs**; above that the
run stops rather than joining through a degraded bridge.

## 7. Preflight and live requirements

- **No CARLA, no CUDA, no perception model, no checkpoint load, no map server.**
  The checkpoint digest is an *identity assertion only*; the file is never
  loaded.
- **OAI is not edited or rebuilt.** The T-tracer byte-compare constraint is
  therefore not engaged.
- Destination must be **non-host-local** and route from the UE namespace
  through `oaitun_ue1`. A host-local destination is matched by `ip rule 0
  (from all lookup local)` and delivered without entering the tunnel, which
  would bypass the radio entirely; receivers therefore run inside the
  `oai-ext-dn` network namespace via `nsenter` (inherited Run-3 behaviour).
  Any host-local or IP-rule bypass is refused.
- Traffic must be proven to reach UE PDCP/RLC: **`NR_PDCP_TX_SDU` nonzero**.
- Per-100-ms new-data TBS service, next backlog, every enqueue and sender
  terminal, every chunk arrival, and complete-reassembly latency are collected.
- Exact decision, terminal and chunk accounting is mandatory.
- Radio state is read back **before and after every cell**.
- Teardown of every task-created process, container, namespace/tunnel, and
  restoration of `noise_power_dB=-50` with read-back, after **every** failure and
  at final completion. The campaign finishes with a **cold-host proof**.
- `PYTHONPATH` is not exported for any client (no CARLA client runs here).
- Other users' CARLA/OAI are not killed; host load and `docker ps` are checked
  first. At registration time the host was cold: load 0.10, no containers, no
  softmodem, no `oaitun_ue1`.

## 8. Stop conditions — no criterion is changed and no evidence is silently repaired

Stop if: any authority/hash/action identity differs; traffic bypasses the UE
radio path; an observation occurs at or after its own action enqueue; missing is
converted to zero; UE-side MCS is missing; the verifier-only gNB match falls
below 99% in any cell; any matched grant is ambiguous or disagrees with gNB
final MCS; terminal/chunk accounting fails; an output would overwrite an
existing path; any protected `20260924_131015` file changes; or teardown or RF
restoration fails.

**Retry policy (predeclared).** One attempt per campaign. A failed attempt is
preserved in place under its own create-only timestamp. Only a *proven*
engineering defect may be repaired, and the repaired campaign runs as a **new**
timestamped root under a **single** code revision. Cells are never spliced
across revisions and no cell is individually rerun into an existing root.
Repair requires authorization.

## 9. Analysis — registered in full, with no post-hoc freedom

Whole cells are the unit of fitting and validation. **Individual rows are never
randomly split.** The model is fitted on FIT cells only and evaluated on blocked
VALIDATION cells only.

**Estimator.** A binned conditional median / rate: assumption-light,
interpretable, and the form that makes monotonicity checkable by construction.
Bins are predeclared in `analysis_spec.py`: payload is the three exact tier
values; backlog is `{zero, <1 KB, 1–10 KB, 10–100 KB, 100 KB–1 MB, 1–10 MB,
≥10 MB}` with **zero kept as its own bin**; MCS is table-0 width-4 bins over
0–28. Minimum support 20 FIT observations per bin; under-supported bins back off
along the predeclared order (payload+backlog+MCS → payload+backlog → payload →
global FIT marginal), **never** to an extrapolation and never to a fit borrowed
from VALIDATION.

**Queue recurrence.** `B_{t+1} = min(B_max, max(0, B_t + P_t − S_t))` with `B_t`
strictly pre-enqueue backlog, `P_t` the actual requested or held-frame payload,
and `S_t` the measured new-data service over the following 100 ms. **Overflow is
explicit, never silently clipped.**

**Reported:** backlog/service/transient trajectories by action, channel, block
position and transition direction; MCS load invariance and channel separation;
pre-enqueue backlog, per-interval service bytes, next backlog, complete
reassembly, uplink latency and 170-ms transport outcomes; payload+backlog vs
payload+backlog+MCS on blocked validation cells; next-backlog error, latency
error, Brier score, false-success rate and monotonicity violations.

### Gates (prospective)

| # | Gate | Threshold |
|---|---|---|
| 1 | `COMPLETE_CAPTURE` | 12/12 cells, exactly 5,400 terminal outcomes, exact accounting |
| 2 | `MCS_COVERAGE_AND_PROVENANCE` | UE-side MCS coverage **100%**; verifier-only gNB match **≥99% in every cell**; 0 ambiguous; 0 UE-vs-final mismatch |
| 3 | `CENSORING_BELOW_1_PERCENT` | ceiling censoring **<1%**; otherwise stop and redesign, do not fit through it |
| 4 | `VALIDATION_NEXT_BACKLOG_ERROR` | normalized median absolute error **≤10%** **and** **≥20%** better than backlog persistence |
| 5 | `VALIDATION_LATENCY_ERROR` | per-cell P50 error **≤17 ms**, P95 error **≤34 ms** |
| 6 | `VALIDATION_TRANSPORT_OUTCOME` | 170-ms false-success rate **≤5%**, Brier **≤0.15** |
| 7 | `MCS_DOES_NOT_HURT…` | adding MCS worsens backlog-only Brier by **≤0.01**; matched near-boundary MCS contrasts point the physically correct way |
| 8 | `MONOTONICITY` | **0** violations inside measured support |

Gate 2's 99% floor is calibrated to measured reality, not aspiration: Run 3's
per-cell verifier coverage was 1.0 in ten cells and 0.99556 in two, so the floor
is achievable and the two-cell shortfall that failed Run 3's stricter 100% rule
would pass here. **Unmatched verifier rows are excluded from fitting and from
gate 7's contrasts; their identities and distribution are reported. They are
never imputed, forward-filled, or reinterpreted as mismatches.**

### Degenerate-arm rule — registered before collection

Gates 4–7 are computed **per channel arm** and reported per arm plus pooled. Per
§3 the `FAVORABLE_STABLE` arm is expected to be degenerate: with backlog pinned
near zero the persistence baseline has ~no error, so "≥20% better than
persistence" is not merely hard but **arithmetically unreachable**.

An arm whose persistence-baseline error is below `1e-9`, or whose VALIDATION
outcome is constant, is reported **`NON_INFORMATIVE`** for the affected
criterion. `NON_INFORMATIVE` is **neither a pass nor a fail**: it never counts
toward a pass, and the gate verdict rests on the informative arm(s). If **no**
arm is informative the gate is **`INDETERMINATE`** and the run cannot be called
a kernel acceptance. Thresholds are not relaxed to force a pass, and a
degenerate arm is never pooled into an informative one to rescue it.

## 10. What this task does not do

- **No Hybrid-SAC fitting or training.** No agent is modified.
- **Reward is not computed or calibrated.** The intended semantics
  (`r = Q_perc − 0.25·L/170` on success, `−1` on delivery failure, service
  failure or timeout, success requiring complete UDP reassembly **and** full
  action-open-to-quality-feedback `L ≤ 170 ms`; held tensors reuse the exact
  action, request no reward, and still enter the queue recurrence) is recorded
  for continuity only.
- **Reward is never multiplied by an admission probability.**
- **Passing the collection gates is not kernel acceptance** and will not be
  reported as such.
- A failed or `INCONCLUSIVE` result is preserved honestly.

## 11. Evidence handling

Raw evidence is **create-only** and hash-manifested; analysis generations are
create-only and never overwrite an earlier analysis. Run 3's integrity amendment
records that a *concurrent* process silently rewrote 15 derived files in place
while that campaign sealed its manifest, so create-only is enforced in code
(`mkdir(exist_ok=False)`, `assert_outside_protected_run`) rather than assumed.
Source commit, effective configs, commands, clock domains, route proof and
SHA-256 + size for every retained file are recorded, and a final verifier-input
manifest binds raw evidence, source commit, analysis, figures and the
protected-old-evidence digests. Large ignored raw traces are **not**
force-added without explicit authorization.

## 12. Expected Phase-B runtime

Approximately **25–40 minutes** after authorization: qualification and upper-anchor
calibration, 12 cells (each a cold RAN rebuild plus 45 s of traffic plus
teardown), final teardown and cold-host proof, and create-only analysis.
