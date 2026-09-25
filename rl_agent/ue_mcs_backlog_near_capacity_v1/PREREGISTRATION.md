# Preregistration — Run-4 near-capacity UE MCS / backlog sweep (v2)

**Status: registered, not launched.** Committed before any collection. Both live
stages require explicit, separate authorization.

Contract id `ue_mcs_backlog_near_capacity_v1`. Claim boundary:
`BOUNDED_NEAR_CAPACITY_QUEUE_TRANSITION_CHARACTERIZATION_UNDER_OAI_N78_100MHZ_273PRB_4D5U_V1_WITH_BLOCKED_VALIDATION_NOT_KERNEL_ACCEPTANCE_AND_NOT_PUBLICATION_EVIDENCE`.

## 0. What changed from v1, and why

v1 inherited Run 3's radio wholesale. That was wrong in a way that would have
invalidated the run:

| v1 | v2 |
|---|---|
| 40 MHz / 106 PRB / 7D2U, spawned by this package | **100 MHz / 273 PRB / 4D5U** via the hash-bound `run_splitfusion_oai_100mhz_4d5u_v1.sh` |
| Run-3 actuator anchors (calibrated at 106 PRB) | **Registered Phase-14a mapping measured under this exact radio** |
| UE T-port 2022 | **2023**, the port the qualified launcher actually opens |
| Actions 70/69/68 frozen from 106 PRB capacity | **No action frozen.** Tiers come from a deterministic rule applied to capacity measured under this radio |
| Clamp, skip, restore, teardown, extraction recorded | **Every one is a hard failure** |
| Routing proof only | Routing proof **plus a real UDP probe** requiring arrival *and* UE PDCP evidence |
| Retry policy in prose | **Enforced** one-attempt authorization with lineage |

The radio lock states the position plainly: the legacy target-SNR mapping is
`CALIBRATED_ON_40MHZ_106PRB_7D2U_DO_NOT_REUSE_AS_100MHZ_EVIDENCE`, and the
Phase-14a binding sets `legacy_mapping_permitted: false`.

## 1. Relationship to Run 3

A **follow-up to** `experiments/ue_mcs_backlog_calibration_v1/20260924_131015`,
never a replacement, rerun or reinterpretation.

- Run 3's verdict stays **`INCONCLUSIVE`**, bound result **4/7 scientific checks
  and 12/13 structural gates**. The superseded "5/7 / 12/12" claim is not
  repeated.
- Five Run-3 files are hash-guarded before and after every stage. A difference
  is a stop condition, never something to repair.
- Run 3's own report asked for this experiment in §10.
- Its *instrumentation* is imported unchanged (telemetry, telnet actuation,
  traffic launch, causal join, extraction). Its *radio* and its *numbers* are
  not.

## 2. Radio identity — pinned and verified three times

`OAI_N78_100MHZ_273PRB_4D5U_V1`; band 78, 100 MHz, 273 PRB, numerology 1,
4D5U (4 DL + 5 UL slots), single UE at `10.0.0.2`, 5QI 6.

Brought up **only** by `run_splitfusion_oai_100mhz_4d5u_v1.sh`
(`8e02f091…`, matching `splitfusion_phase14a_campaign_binding_v1.json:launcher.sha256`)
with execution token `SPLITFUSION_OAI_100MHZ_4D5U_ATTACH`. This package never
spawns a softmodem itself, so the radio identity is the launcher's.

23 identities are pinned in `radio_binding.py` and verified at
**`before_preflight`**, **`before_scientific_cells`** and **`final_sealing`**:
launcher, launcher runner and its config, radio lock, 273 PRB gNB config, UE
config, channel config, the Phase-14a mapping JSON/CSV/manifest, `T_messages.txt`
and **both** compiled `T_messages.txt.h` copies, the five tracer binaries
(`multi`, `record`, `csv`, `replay`, `extract_config`), the extractor script,
`nr-softmodem`, `nr-uesoftmodem` and `libtelnetsrv.so`.

Where an independent file already recorded the same digest, the pin cites it.
Two do not, and are recorded honestly rather than silently: **`nr-softmodem` and
`nr-uesoftmodem` differ from the `ue_n3_oai_ul_live_stage_v1.json` seal**
(`01489dfb…` / `7cdeee94…`). They were rebuilt on 2026-08-25, after the
2026-08-03 edit to `gNB_scheduler_ulsch.c` that added the SINR UL-MCS policy gate
this experiment depends on. The older seal predates that work and is therefore
**not** treated as corroboration.

The T-tracer byte-compare constraint is enforced as a check: both compiled
`T_messages.txt.h` copies must be identical (`e5801830…`). OAI is neither edited
nor rebuilt by this work.

**Refused:** PRB 106, 40 MHz, `gnb.sa.band78.fr1.106PRB.usrpb210.conf`, both
legacy `run_track1_oai_default106_*` launchers, the legacy mapping CSV, and the
eleven legacy environment overrides the launcher itself rejects.

## 3. Actuator mapping — measured under this radio

The registered Phase-14a mapping, twelve strictly monotonic anchors from
`experiments/splitfusion_phase14a_100mhz_calibration_v1/20260904_phase14a_live_calibration_retry3/mapping.json`:

| noise_power_dB | −13 | −12 | −11 | −10 | −9 | −8 | −7 | −6 | −5 | −4 | −3 | −2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| achieved median PUSCH SNR dB | 25.5 | 23.5 | 21.5 | 19.5 | 17.5 | 15.5 | 14.0 | 12.0 | 10.0 | 8.5 | 6.5 | 5.0 |

Measured range 5.0–25.5 dB covers the required 5.5–24.5 dB; interpolation is
`MONOTONIC_PIECEWISE_LINEAR_INVERSE` at 0.25 dB granularity.

**Not overclaimed.** The Phase-14a artifacts record
`campaign_mapping_qualified: false` and `profile_replay_performed: false`; that
gate belonged to the 288-cell campaign and additionally required a four-profile
replay. Run 4 binds the twelve **anchor measurements**, which passed all their
observation gates under this radio, and claims nothing more.

**No Run-3 106 PRB anchor survives.** A test asserts the current set is not the
legacy set.

## 4. Capacity qualification — a separate, bounded, authorized stage

Run 4 no longer sites tiers from Run-3 numbers. 273 PRB is 2.58× the legacy
bandwidth and 4D5U gives 5 uplink slots per 9 against 7D2U's 2 per 10, so the
uplink is expected to be several times faster; a tier chosen from the 106 PRB
figure would land in the wrong regime — exactly what made Run 3 inconclusive.

Stage `near_capacity_capacity_qualification`, token
`AUTHORIZE_NEAR_CAPACITY_CAPACITY_QUALIFICATION`:

- One RAN instance under the exact radio.
- Three operating points, **held constant**, at the registered `ADVERSE_STABLE`
  trace's own percentiles: **7.827 / 8.608 / 9.604 dB**. This measures a
  capacity surface; it does not replay a profile.
- A saturating probe — the largest eligible catalogue action (action 0,
  `split_noae_uint8_q0000`, 3,568,326 B, **285.46608 Mbps exactly** at 10 fps) —
  so the queue is certainly backlogged and the measured rate is capacity, not
  offered load. The runtime re-opens the pinned catalogue and requires all four
  identities to match. **The probe is never a tier.**
- The **primary service measurement** is application delivery at ext-DN: exact
  unique `SSBURST` payload bytes (the 24-byte wire header excluded), de-duplicated
  by `(frame_id, chunk_id)`, in fixed 100 ms `CLOCK_MONOTONIC` windows. PUSCH
  transport-block size is **never** the primary capacity measure because grants
  and retransmissions can count bytes that were not uniquely delivered.
- `NR_RLC_TX_DEQUEUE`, `NR_RLC_TX_SDU` queue recurrence and
  `GNB_PDCP_RX_DELIVER` are retained as independent corroboration. They cannot
  replace or rescale the ext-DN application-goodput samples.
- Settle 3 s, measure 10 s, 100 ms sampling. Each point gets a separately
  scheduled probe on a recorded future monotonic epoch. The probe stops after
  its measurement; before the next target is primed, the runner must observe
  five consecutive zero-backlog UE RLC samples **after** the latest observed
  PDCP/RLC ingress and then a 0.5 s observation-stable interval with no new
  `NR_PDCP_TX_SDU` or `NR_RLC_TX_SDU`. Failure of either the queue-zero or
  ingress-quiet proof is a hard refusal, so service from one operating point
  cannot leak into the next. The RAN remains one continuous instance. The
  bounded worst-case stage budget is 320 s (ordinary completion is expected
  sooner).

**Stage gates:** every operating point present; ≥60 service samples; ≥80% of
intervals continuously backlogged (otherwise the probe did not saturate and the
number is offered load, so the stage **refuses**); positive median service;
capacity must not fall as SNR rises beyond a 10% tolerance. Every point label is
unique and exactly one of `p25/p50/p75`; target SNRs must exactly match the
registered values; all fields must be finite; and `p10 ≤ p50 ≤ p90`. In
addition, each point needs at least **30 achieved-PUSCH SNR samples**, its
achieved median must be within **±1.0 dB** of the registered target, and the
three achieved medians must strictly satisfy `p25 < p50 < p75`. Commanded noise
alone never qualifies an operating point.

The p50 point's 100 ms service samples also receive a fixed-seed
(`2026092403`), 2,000-draw, 95% non-parametric median bootstrap. The
deterministic low/medium/high action triplet selected from the point estimate,
the lower confidence bound and the upper confidence bound must be identical.
If sampling uncertainty changes even one action, qualification refuses rather
than freezing an unstable boundary.

`C_adv` := median service at the **p50** operating point.

**Evidence and lifecycle gates.** The current source inventory is recomputed at
re-open time and must equal the captured inventory exactly. The manifest is an
exact, canonical, root-contained inventory: absolute paths, `..`, duplicate
paths, missing files and unmanifested extra files all refuse. The verifier
reconstructs all three `CapacityPoint` rows from retained samples, reruns the
complete audit and deterministic tier rule, and requires exact equality with
the sealed result. Core containers are bound at runtime to their immutable
Docker image IDs plus available RepoDigests (not merely mutable compose tags).
All launcher, Docker, process-probe, signal, route, core-down and T-tracer
extraction subprocesses have registered positive wall-clock timeouts. The final
cold-state JSON is create-only and capture additionally requires clean RAN/core
teardown, successful extraction, no teardown note and no cold-state probe error.

## 5. Deterministic tier rule — registered before the boundary is measured

Given `C_adv` and the digest-checked catalogue, the three tiers are a pure
function. For each tier take the eligible action whose offered rate is closest to
`ratio × C_adv`, ties toward the smaller action id:

| Tier | ratio |
|---|---|
| low | 0.50 |
| medium | 1.00 |
| high | 1.40 |

Then **refuse unless** the result brackets the boundary: three distinct actions,
strictly increasing payloads, low strictly **below** `C_adv`, high strictly
**above** it, and medium within **±25%** of it. Refusing is a legitimate outcome
— if the catalogue cannot bracket the measured capacity, nothing is frozen and
the decision returns to Abiodun.

The catalogue spans approximately 0.5–285.46608 Mbps across 72 eligible
actions, so the rule brackets any capacity strictly inside that range; it
refuses outside it.

Each tier's decoder digest is **recorded**, and tiers are deliberately **not**
required to share one. This experiment loads no model, runs no CUDA and decodes
nothing — a tier is a number of bytes on the wire, so decoder identity cannot
confound a queueing, MCS or backlog measurement. A blank digest is refused.

## 6. Design (unchanged from v1)

Six payload permutations × two channels = **12 cells**, 3 blocks × 150 decisions
at 10 fps, **450 per cell**, **5,400 tagged decisions**. Load is a within-cell
factor on one continuous timeline inside a single radio instance.

| Partition | Permutations | Transitions (per channel, ×2 each) |
|---|---|---|
| **FIT** | `L-M-H`, `M-H-L`, `H-L-M` | `L→M`, `M→H`, `H→L` |
| **VALIDATION** (blocked) | `H-M-L`, `L-H-M`, `M-L-H` | `H→M`, `M→L`, `L→H` |

Proven by `audit_cell_plan` and asserted in tests. Execution order is shuffled
from pinned seed `2026092401`; **cells never move between partitions**.

**Declared limitation:** the cyclic orders and their reverses partition the six
ordered transitions into *disjoint* sets, so validation is an **extrapolation
across transition direction**. A gate 4/5/6 failure may indicate that rather than
an unusable kernel, and the report must say which.

## 7. State under test

| Feature | Definition | Source |
|---|---|---|
| `previous_ul_mcs` | latest **strictly prior** UE-decoded **round-0 / new-data** granted/final table-0 MCS from UL DCI | `NRUE_MAC_DCI_GRANT`, `direction=1`, `round=0` |
| `pre_enqueue_backlog_bytes` | raw UE RLC bytes from the last `NRUE_MAC_RLC_BUFFER_STATUS` tick **strictly before** this payload's enqueue | pre-multiplex, per-LCID sum |

- gNB selected/final MCS is **verifier-only**, never an actor input.
- Real MCS 0 and real backlog 0 are preserved.
- **Exact MCS freshness limit: `MCS_MAX_AGE_MS = 200.0`.** One pinned value, not
  a candidate list. Two decision periods at 10 fps, and the bound under which
  Run 3 measured 100% coverage in all twelve cells. 100/150/200/250 ms are
  reported as a sensitivity **diagnostic only** and never select the operative
  limit. If coverage at 200 ms is below 100%, **gate 2 fails**; the limit is not
  relaxed to rescue it.
- Missing stays missing; never forward-filled, never coerced to 0.
- Age is a validity gate only. The hidden profile identity is never exposed to
  the actor.

## 8. Clock domain

`NR_PDCP_TX_SDU` carries both the tracer's `CLOCK_REALTIME` header and an
in-payload `CLOCK_MONOTONIC` pair from the same call site
(`nr_pdcp_oai_api.c:941-944`) — a measured same-event bridge.

**Refusal is implemented, not described:** `require_clock_bridge` raises when the
residual P95 exceeds **1.0 µs**, and also when it is undefined (no same-event
rows). The join is not attempted through a degraded bridge and the limit is not
widened.

## 9. Preflight, radio-path proof and the UDP probe

- **No CARLA, no CUDA, no perception model, no checkpoint load, no map server.**
  Decoder digests are identity assertions only; no file is loaded.
- **OAI is not edited or rebuilt.**
- Destination must be non-host-local and route from the UE namespace through
  `oaitun_ue1`; receivers run inside the `oai-ext-dn` namespace via `nsenter`.
- **A real pre-scientific UDP probe runs in every cell**, before any tagged
  traffic. It sends 5 × 1,200 B datagrams from the UE address to the ext-DN
  receiver and requires **both**:
  1. at least one datagram arriving at the ext-DN receiver, **and**
  2. **nonzero `NR_PDCP_TX_SDU` events while they were in flight**, counted live.

  Either alone is insufficient: arrival without PDCP evidence is exactly what a
  host-local shortcut would produce. Failure of either aborts the cell.
- After the measured receivers are ready, but immediately before the first
  scientific decision, a separate **target-channel primer** sends 5 × 1,200 B
  datagrams on port 5411. It is infrastructure only—not an action, transition,
  reward or measured payload. All five datagrams must reach ext-DN; the UE must
  retain fresh PDCP ingress and a fresh table-0, round-0 UL grant; and the queue
  must then show three complete zero-RLC ticks observed after the latest primer
  PDCP receipt plus a 20 ms ingress-quiet interval. The first decision must
  follow that drain proof and occur no more than 100 ms after the retained grant
  receipt. This prevents the first state of a cell from fabricating MCS=0 or
  inheriting an unrelated old grant while keeping the primer out of the
  scientific workload.
- Radio state is read back before and after every cell.
- Teardown covers our processes **and the launcher's detached, root-owned
  softmodems** — the launcher leaves the RAN up by design, so inheriting Run-3
  teardown unchanged would leak a gNB and a UE between cells.
- `noise_power_dB` is restored to −50 with read-back after every failure and at
  final completion; the campaign ends with a cold-host proof.

## 10. Failure policy — recorded is not enough

Run 3 recorded clamps, skips, restore results, teardown notes and extraction
failures and let the run continue. **Every one is a hard, nonzero failure here:**

`UNRESOLVED_CALIBRATION` · `UNREGISTERED_CLAMP_OR_SKIP` ·
`RF_RESTORE_OR_READBACK_FAILURE` · `SENDER_OR_RECEIVER_FAILURE` ·
`TTRACER_EXTRACTION_FAILURE` · `ACCOUNTING_FAILURE` · `ANY_TEARDOWN_NOTE` ·
`NON_COLD_FINAL_STATE` · `RADIO_BINDING_DRIFT` · `UDP_PROBE_FAILURE` ·
`PROTECTED_EVIDENCE_CHANGED`

A cell that clamps a single target, skips a single 100 ms actuator command, fails
read-back, emits a teardown note, or whose extraction fails, is `FAILED` — not
`CAPTURED` with a note. The campaign exits nonzero unless 12/12 are `CAPTURED`,
the final state is cold, there are no teardown notes, and `final_sealing`
verification passes.

Additional stop conditions: authority/hash/action-identity drift; traffic
bypassing the UE radio path; an observation at or after its own enqueue; missing
converted to zero; UE-side MCS missing; verifier-only gNB match below 99% in any
cell; ambiguity or UE-vs-final disagreement; an output that would overwrite an
existing path; any protected Run-3 file changing.

## 11. Authorization and lineage — enforced

Two stages, two tokens, each requiring an operator-written `AUTHORIZATION.json`
**beside** the campaign root, so a run cannot authorize itself. Unknown fields
are refused rather than ignored.

**One attempt per campaign.** A second attempt is refused unless the
authorization names the attempt it supersedes (which must exist) *and* states the
proven engineering defect that was repaired. Prior attempts are listed, never
removed. A scientific run additionally refuses to start if this task's own
sources are uncommitted, so a run always executes a single committed revision.
Each run writes `lineage.json` naming its parent, its authorization and the
source commit.

## 12. Analysis — registered in full

Whole cells are the unit; individual rows are never randomly split. Fitted on
FIT cells only, evaluated on blocked VALIDATION cells only.

**Estimator.** Binned conditional median / rate. Payload is the three selected
tier values; backlog is `{zero, <1 KB, 1–10 KB, 10–100 KB, 100 KB–1 MB, 1–10 MB,
≥10 MB}` with **zero as its own bin**; MCS is table-0 width-4 bins over 0–28.
Minimum support 20 FIT observations; under-supported bins back off along the
registered order (payload+backlog+MCS → payload+backlog → payload → global FIT
marginal), never to an extrapolation and never to a fit borrowed from VALIDATION.

**Queue recurrence.** `B_{t+1} = min(B_max, max(0, B_t + P_t − S_t))`, overflow
explicit and never silently clipped.

**Gate-5 P95 estimator, completed.** A binned conditional *median* cannot produce
a P95, so v1 left that arm unspecified. It is now a **mixture estimator**: each
FIT bin retains its full empirical latency sample, a validation cell's predicted
distribution is the mixture weighted by that cell's bin occupancy, and the
predicted P95 is the mixture's P95. It uses the **weighted Type-7 convention**,
which reduces *exactly* to the observed-side `percentile` under equal weights —
asserted at nine quantiles — so the two arms of gate 5 cannot disagree about what
a percentile means.

**Gate-7 matching and effect rule, completed.** v1 had a sentence; this is a
procedure. **Match** exactly on (payload level, backlog bin) within one channel
arm — exact strata, never a propensity score. **Restrict** to strata whose
payload is within **±25%** of `C_adv`, the only region where the channel is
supposed to decide feasibility. **Contrast** the low-MCS group (bin ≤ low)
against the high-MCS group (bin ≥ high), requiring ≥30 observations per side and
a gap of ≥2 MCS bins. **Effect** = success-rate difference (high − low). **Rule:**
an effect below −0.05 is a `WRONG_DIRECTION` violation; within ±0.05 it is
`NULL_WITHIN_TOLERANCE`, reported as neither pass nor violation; above it is
`CORRECT_DIRECTION`. Any violation fails gate 7; **no evaluable contrast is
`INDETERMINATE`, not a pass.**

### Gates (prospective)

| # | Gate | Threshold |
|---|---|---|
| 1 | `COMPLETE_CAPTURE` | 12/12 cells, exactly 5,400 terminal outcomes, exact accounting |
| 2 | `MCS_COVERAGE_AND_PROVENANCE` | UE-side coverage **100%** at the pinned 200 ms limit; verifier-only gNB match **≥99% in every cell**; 0 ambiguous; 0 UE-vs-final mismatch |
| 3 | `CENSORING_BELOW_1_PERCENT` | ceiling censoring **<1%**; otherwise stop and redesign |
| 4 | `VALIDATION_NEXT_BACKLOG_ERROR` | NMAE **≤10%** **and** **≥20%** better than backlog persistence |
| 5 | `VALIDATION_LATENCY_ERROR` | per-cell P50 error **≤17 ms**, P95 error **≤34 ms** |
| 6 | `VALIDATION_TRANSPORT_OUTCOME` | 170-ms false-success rate **≤5%**, Brier **≤0.15** |
| 7 | `MCS_DOES_NOT_HURT…` | Brier degradation **≤0.01**; matched near-boundary contrasts point the right way |
| 8 | `MONOTONICITY` | **0** violations inside measured support |

Gate 2's 99% floor is calibrated to measured reality: Run-3 per-cell verifier
coverage was 1.0 in ten cells and 0.99556 in two. Unmatched verifier rows are
excluded from fitting and from gate 7, and reported; never imputed,
forward-filled, or reinterpreted as mismatches.

### Degenerate-arm rule

Tiers bracket the **ADVERSE** boundary by construction, so the **FAVORABLE** arm
sees the same offered loads on a faster channel and is expected to drain with
backlog near zero. Its persistence baseline error would then be ~0, making
"≥20% better than persistence" arithmetically unreachable.

Gates 4–7 are computed **per arm** and reported per arm plus pooled. A degenerate
arm is `NON_INFORMATIVE`: **never a pass**, never pooled into an informative arm
to rescue it, and the verdict rests on the informative arm(s). If no arm is
informative the gate is `INDETERMINATE` and the run is not a kernel acceptance.
Thresholds are not relaxed.

## 13. What this task does not do

- No Hybrid-SAC fitting or training; no agent is modified.
- Reward is not computed or calibrated. Intended semantics
  (`r = Q_perc − 0.25·L/170` on success, `−1` on delivery failure, service
  failure or timeout; success requires complete UDP reassembly **and**
  `L ≤ 170 ms`; held tensors reuse the exact action, request no reward, and still
  enter the queue recurrence) are recorded for continuity only.
- **Reward is never multiplied by an admission probability.**
- **Passing the collection gates is not kernel acceptance.**
- A failed or `INCONCLUSIVE` result is preserved honestly.

## 14. Evidence handling

Raw evidence is create-only and hash-manifested; analysis generations are
create-only and never overwrite an earlier analysis. Run 3's integrity amendment
records a concurrent process silently rewriting 15 derived files in place, so
create-only is enforced in code (`mkdir(exist_ok=False)`,
`assert_outside_protected_run`), not assumed. Source commit, effective configs,
commands, clock domains, route proof, UDP-probe outcome, the three identity
verifications and SHA-256 + size for every retained file are recorded. A final
verifier-input manifest binds raw evidence, source commit, analysis, figures and
the protected-old-evidence digests. Large ignored raw traces are not force-added
without explicit authorization.

## 15. Expected runtime

- Capacity qualification: **3–5 minutes**.
- Scientific cells: **25–40 minutes** (12 cells, each a cold RAN rebuild via the
  launcher plus 45 s of traffic plus teardown), then create-only analysis.
