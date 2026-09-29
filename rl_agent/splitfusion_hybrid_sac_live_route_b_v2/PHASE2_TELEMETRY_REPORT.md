# Run-4 live-qualification package v2 — Phase 2: bounded live UE telemetry

**Verdict:** `PHASE2C_RADIO_TELEMETRY_QUALIFICATION_PASSED` (10/10 pre-registered gates).
**Date:** 2026-09-29. Provider and gates committed at `557662f` before collection
(gates SHA-256 `8a7f3b4007794228f3a6156365ebb982289f44291549152fc066d66a2fc9dca6`).
This report and the radio evidence are a separate commit.

Scope of the run: exactly one create-only, radio-only qualification. OAI/RFsim/T-tracer
only; no CARLA, CUDA, FCOS or actor inference. It is **not** a policy or perception result.

## Implementation (Phase 2A)

`ue_telemetry_provider_v2.py`:

- **Transport reused unchanged.** The UE runs on port 2023; `multi` relays it to 2123. The
  durable `record` client writes `ue.raw`, and three extra `csv -f` clients read the same relay
  for `NRUE_MAC_DCI_GRANT`, `NRUE_MAC_RLC_BUFFER_STATUS` and `NR_PDCP_TX_SDU`. There is no new
  OAI socket, no OAI edit or rebuild, and no tailing of `ue.raw`.
- **Hot path.** Parsing, clock conversion and aggregation run in reader threads. Each accepted
  update publishes one immutable `TelemetrySnapshotV2` by reference assignment.
  `snapshot()` is a single attribute read. The caches are fixed-capacity deques (8 DCI,
  8 RLC ticks). Audit lines go through a bounded queue to a separate writer thread. The legacy
  `LiveCsv` (unbounded list, per-line flush) is not used.
- **DCI rule.** Uplink, table 0, HARQ round 0, bound RNTI. NDI 0 and 1 are both accepted.
  MCS outside [0, 28] is passed through so the registered selector refuses it.
- **RLC rule.** Rows are grouped by the complete `(rnti, ue_id, frame, slot)` tick, and the
  latest value per distinct LCID is summed. A tick is published only when the next tick's
  first row proves it complete; availability is that proof instant. A measured 0 is valid;
  missing telemetry is typed missing.
- **Clock.**
  - The T `time` field (REALTIME, local time-of-day, µs) is reconstructed against the host
    receipt REALTIME, which is midnight-safe and rejects future timestamps.
  - REALTIME→MONOTONIC uses the median of the last 32 *already received* PDCP dual-stamped
    anchors, with at least 8 required for warm-up.
  - MONOTONIC→RAW uses the latest bracketed in-process host clock pair.
  - A REALTIME step, a MONOTONIC/RAW jump or an anchor discontinuity above 1 ms resets the
    bridge and clears every cache.
- **Decision.** The sequence is `open_decision` (snapshot → commit → open), then the unchanged
  `select_prior_new_data_ul_mcs` / `select_pre_action_rlc_backlog`, then the exact training
  `FreshnessPolicyV2` (digest `6c694ebe…3d65`, the checkpoint's `freshness_policy_sha256`),
  applied with the guard's age rule. Any failure yields typed missing or stale evidence and
  `ExternalFallbackRequired`. `act_or_fallback` never calls the actor after a refusal.
- **Pins.** Tracer binaries, both softmodems, the launcher, radio configs and `T_messages.txt`
  (source and both compiled headers) are verified through the existing `radio_binding.verify`.
  The four event-emission sources and the production config are pinned in
  `phase2_telemetry_qualification.EMITTER_PINS`. They are pinned as observed: the RLC/PDCP
  T-event edits are uncommitted in the OAI submodule, and this task neither wrote nor changed
  them.

## Offline tests (Phase 2B)

- `test_ue_telemetry_provider_v2`: **35 OK**. Covers:
  - DCI acceptance and rejection: NDI 0 and 1, retransmission round, direction, table,
    out-of-range MCS (fails closed), MCS 0 valid;
  - RLC ticks: multi-LCID sums, duplicate and contradictory LCID rows, incomplete tick hidden,
    availability equals the proof instant, valid measured zero versus missing;
  - rejection and fallback: stale, cross-session, cross-UE, cross-epoch, child EOF/death,
    malformed header (fatal) and malformed rows (counted);
  - clock: warm-up, midnight crossing, REALTIME step, MONOTONIC/RAW jump, anchor
    discontinuity, future anchor;
  - causality and hot path: no future sample enters a decision, bounded cache, O(1)
    snapshot, and fault injection with zero actor calls.
- `test_phase2_telemetry_qualification`: **7 OK** (identity round trip, independent raw tick
  replay, raw comparison, gate evaluation with every gate's failure case, pins).
- Existing suites: `ue_production_queue_capture_v1` 39 OK, `run4 state_adapter` 25 OK,
  `ue_mcs_backlog_near_capacity_v1.test_capacity_runner` 42 OK.

## Radio qualification (Phase 2C)

Evidence (untracked):
`rl_agent/experiments/splitfusion_hybrid_sac_live_route_b_v2/20260929_phase2c_ue_telemetry_qualification/`
(84 files, 93 MB).

| Artifact | SHA-256 |
|---|---|
| `TERMINAL.json` | `5d30ad0345290bbc14905daf5f9a31d8711437c917b6cb29b32ee330b1196df6` |
| `QUALIFICATION_RESULT.json` | `5305904219abf8a1ce84335f726cb13b527cfa7994353ef96f9c77ed50752fbd` |
| `cells/…/decisions.json` | `880ca8010bc6cff595a7fddd06739d751bad0e49244346ea8c80c286263f858e` |
| `cells/…/ttracer/ue/ue.raw` | `f49992265a0c7fbcc2926e09c3095a4b2865a0559fe4092f187570302ff82a0b` |

Setup: 100 MHz / 273 PRB / 4D5U radio through the pinned launcher. The channel was held at the
first `FAVORABLE_STABLE` target. Traffic was the `favorable_stable__perm0` production schedule
(450 frames at 10 Hz; production sender and receivers). There were 300 decision snapshots, each
5 ms before its frame opened. The maximum schedule lateness was 1.22 ms.

| Gate | Result |
|---|---|
| G1 readers alive, schemas, pins | PASS: all three STREAMING at every decision; 0 malformed headers; pins equal before and after |
| G2 fresh coverage (decisions 10–299) | PASS: **290/290 = 100%** admitted |
| G3 causality | PASS: 0 future/negative-age, cross-UE, cross-session, partial-tick or contaminated (every sample was available before its frame's first send) |
| G4 raw replay agreement | PASS: 300/300 DCI and 300/300 RLC values and identities equal the independent `ue.raw` replay; 0 mismatches |
| G5 explicit fallback | PASS (vacuous here: 0 fallbacks; exercised offline) |
| G6 snapshot latency | PASS: P50 1.1 µs, P99 2.4 µs, max 3.7 µs (full radio-evidence assembly P99 0.80 ms) |
| G7 bridge residual | PASS: P50 0.25 µs, P95 0.55 µs, max 6.8 µs (40,997 out-of-sample anchors) |
| G8 bounded cache | PASS: 8/8 DCI and 8/8 RLC at every decision |
| G9 coexistence | PASS: `record` and the 3 live readers alive through the window; 0 unexpected EOF |
| G10 restore and cold | PASS: channel read-back equals initial; RF restored; core stopped; host cold |

Measured timings (ms). Fallback fraction was 0/300.

| Slot | Source→availability P50 / P95 / P99 (max) | Decision age P50 / P95 / P99 (max) |
|---|---|---|
| UL MCS | 0.65 / 1.18 / 1.58 (21.3) | 49.8 / 61.7 / 66.2 (78.5) |
| RLC backlog | 1.47 / 4.53 / 5.02 (5.7) | 2.6 / 5.6 / 6.2 (6.8) |

## Limitations (stated, not rescued)

- **MCS freshness margin.** UL grants arrive only while a frame is being sent, so the selected
  grant is 50–78 ms old at the decision. The worst case is 21.5 ms inside the 100-ms bound, on
  a favorable channel. Longer idle gaps (smaller payloads, SKIP-like actions, lower cadence)
  would push decisions into explicit fallback. That outcome would be correct, but it is not
  measured here.
- **Backlog was 0 at every decision.** The queue drained between frames: 54,851 ticks were
  published and 64,607 complete ticks were replayed from `ue.raw`, but none of the 300 selected
  ticks was positive. The same happens in training (98.7% zero). Positive-backlog agreement at a
  decision instant is therefore not demonstrated live.
- **NDI and retransmission coverage.** All 11,812 round-0 UL grants in `ue.raw` had NDI = 1, and
  no round > 0 UL grant occurred. The NDI-0 and retransmission rules are proven offline only.
  On this trace the corrected rule and the old `ndi == 1` filter select the same grants.
- **Channel coverage.** One held channel condition, and MCS values 26–28 only.
- **ACK path.** The UE→map feedback path is host-local (CLAUDE.md), so it is not exercised here.

## Files

Phase 2A/B (commit `557662f`): `ue_telemetry_provider_v2.py`, `test_ue_telemetry_provider_v2.py`,
`phase2_telemetry_qualification.py`, `test_phase2_telemetry_qualification.py`,
`PHASE2_BLOCKER_CORRECTION.md`, `PHASE0_ADDENDUM_2026-09-29.md`.
Phase 2C evidence commit: this report only. Radio evidence is untracked and pinned above.
`PHASE2_BLOCKER.md` is unchanged.
