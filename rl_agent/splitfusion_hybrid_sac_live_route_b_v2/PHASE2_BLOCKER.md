# Run-4 live-qualification package v2 — Phase 2 blocker

**Verdict:** `BLOCKED__NO_CAUSAL_ONLINE_UE_TTRACER_FEED`
**Date:** 2026-09-29. **Scope:** read-only adjudication. No CARLA, OAI, RFsim, Docker,
softmodem, map server, CUDA or network service was started. No code was written for Phase 2.

## Question

Phase 2 is allowed to implement "only the narrow provider that translates the already-proven
UE T-tracer stream" into `state_adapter.RawUeUlDciGrantCandidateV1` and
`RawUeRlcBacklogSampleV1` records **at decision time**. The prompt defines the stop condition:
*"If retained code supports only post-run CSV parsing and there is genuinely no causal online
event feed to bind, stop with exact evidence. Do not fabricate an online source."*

## Evidence

1. **The UE emits these quantities only as T events.** The only UE-side OAI change is in
   `openair2/LAYER2/NR_MAC_UE/nr_ue_scheduler.c` (+55 lines, uncommitted in the OAI submodule).
   It adds `T(T_NRUE_MAC_RLC_BUFFER_STATUS, …)` and `T(T_NRUE_MAC_BSR_STATUS, …)`.
   `NRUE_MAC_DCI_GRANT` is declared at `common/utils/T/T_messages.txt:244`, and
   `NRUE_MAC_RLC_BUFFER_STATUS` at `:248`. No socket, shared-memory or UDP exporter exists in
   the UE MAC, RLC or PDCP code.

2. **The only retained consumer is a recorder that writes a file during the run.** The UE is
   launched with `--T_stdout 2 --T_nowait --T_port 2023` (`scripts/ue_multi_start_ttracer.sh`,
   `ue_mcs_backlog_near_capacity_v1/radio_binding.py:244-249`). The pinned consumer is
   `OAI/openairinterface5g/common/utils/T/tracer/record` (`radio_binding.py:130-131`), which
   writes `ttracer/ue/ue.raw`.

3. **Decoding happens only after teardown.**
   `ue_mcs_backlog_near_capacity_v1/capacity_runner.py:1180` (`extract_ttracer`) and
   `ue_mcs_backlog_calibration_v1/runner.py:678` run `scripts/ttracer_extract_csv_smoke.sh`
   on the finished `.raw` file. That script replays the file through the `replay` and `csv`
   tools into `NRUE_MAC_DCI_GRANT.csv` / `NRUE_MAC_RLC_BUFFER_STATUS.csv`. In
   `ue_production_queue_capture_v1/runner.py:584-588` it runs in the cell's `finally` block,
   after `teardown_ran()` has stopped both softmodems.

4. **The causal timestamp requires the whole finished cell.** Both events carry only a wall
   `time` field. `ue_production_queue_capture_v1/parse.py:load_ue_traces` converts it to
   monotonic through a `ClockBridge` whose offset is the **median** over at least 100
   dual-stamped `NR_PDCP_TX_SDU` / `NR_RLC_TX_SDU` / `NR_RLC_TX_DEQUEUE` events from the whole
   cell (`parse.py:85-95`). A decision at time *t* cannot use a median taken over events after
   *t*. The retained bridge is therefore post-hoc by construction.

5. **No online follower exists anywhere.** Across `rl_agent/`, `scripts/`,
   `oai_layer_latency/` and `uplink_only_spatial_map_pipeline/` (excluding evidence
   directories), nothing reads T port 2023 live, tails `ue.raw` while it is written, runs
   `textlog`/`multi` against a live softmodem, or builds the wall→monotonic bridge causally.
   `textlog` is only built (`scripts/ttracer_build_tools.sh:6`), never run.

6. **The repository's own audit says the same.**
   `splitfusion_hybrid_sac_ue_state_audit_v1/UE_STATE_EVIDENCE_AUDIT.md`, section "Required
   additional instrumentation", lists the explicit UE-runtime pre-enqueue read as
   "the part that does not exist today".

Training did not need an online feed. The Run-4 collector drew MCS from a fitted Markov
provider and backlog from the v2 transport head (`collector_v1.advance_once`). The 12-cell
production capture joined its T-tracer traces post-run (`parse.py`). So the gap is not
visible in any offline artifact.

## Why this cannot be closed inside the Phase-2 boundary

A live provider would need at least three new pieces, and each is outside the scope limits:

- a live T-protocol consumer. `record` holds the UE's T connection, so this means either a
  second client through `multi`, or an incremental `.raw` parser. Neither exists or has been
  proven.
- an online, causal wall→monotonic clock bridge with its own acceptance criterion. That is
  new temporal-validation and calibration work, which the prompt forbids.
- a qualification that the event reaches the UE application strictly before the 100-ms
  freshness bound. That is new radio evidence, which the prompt also forbids.

Anything short of that would either fabricate the source or silently turn the post-run join
into a live claim. Both are forbidden.

## Secondary observation (not the blocker)

The post-run parser keeps only `ndi == 1` grants (`parse.py:load_ue_traces`), while
`state_adapter.select_prior_new_data_ul_mcs` accepts NDI 0 or 1 ("new data is an NDI
*toggle*"). A future online provider must pick one rule, pin it and state it.

## What is still valid

- Phase 0 (`PHASE0_RECONCILIATION.md`) and Phase 1 (`PHASE1_REPORT.md`, the frozen, exactly
  restored seed-43 actor) do not depend on this seam and stay valid.
- Phases 3 (continuous execution) and 4 (hold and 170-ms ticket) are logically independent of
  the telemetry source. Under the stop rule they were **not** started.

## Decision required (Abiodun–Codex)

Choose one before the live package can proceed:

1. **Authorize a bounded telemetry seam.** A live consumer of UE T port 2023 or of the growing
   `ue.raw`, plus a causal clock bridge with a pre-registered acceptance test, qualified on a
   short radio run. This is new instrumentation.
2. **Authorize an explicitly labelled external fallback.** The qualification runs the actor
   only when a causal feed exists; otherwise it records `ExternalFallbackRequired`. Without
   (1) this degenerates to 100% fallback, so it is not a policy qualification.
3. **Authorize Phases 3–5 without Phase 2.** They can be built and tested offline, with the
   readiness verdict held at `BLOCKED` until (1) is resolved.
