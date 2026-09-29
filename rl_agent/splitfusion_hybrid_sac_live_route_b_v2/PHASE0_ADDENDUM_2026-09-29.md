# Addendum to `PHASE0_RECONCILIATION.md` (2026-09-29)

`PHASE0_RECONCILIATION.md` is kept unchanged. This addendum supersedes its §5 ("Phase-2 risk").

§5 said: "No retained module reads the T port live … or builds the wall→monotonic bridge
causally." The first half is **incorrect**. OAI `multi` already relays UE port 2023 to 2123 while
`record` stays attached (`ue_mcs_backlog_calibration_v1/runner.py:330-348`). Live `csv -f`
readers are already attached for DCI, RLC and PDCP
(`ue_mcs_backlog_near_capacity_v1/runner.py:399-418`). The second half is correct: no causal
bridge existed. v2 adds one (`ue_telemetry_provider_v2.CausalClockBridgeV2`).

Two clock facts were not recorded in Phase 0 and are pinned here:

- The T `time` column is `CLOCK_REALTIME`, printed by the `csv` tool as local time-of-day in
  microseconds (`common/utils/T/T.h:176`, `tracer/csv.c:32-34`). It wraps at midnight.
- The `NR_PDCP_TX_SDU` fields `mono_sec`/`mono_nsec` are `CLOCK_MONOTONIC`
  (`nr_pdcp_oai_api.c:942`), **not** `CLOCK_MONOTONIC_RAW`. The Run-4 contract clock is RAW.

The corrected per-phase allow-lists are in the respective phase reports. See
`PHASE2_BLOCKER_CORRECTION.md`.
