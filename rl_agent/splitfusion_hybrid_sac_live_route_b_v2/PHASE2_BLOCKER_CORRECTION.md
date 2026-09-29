# Correction to `PHASE2_BLOCKER.md` (2026-09-29)

**Status:** `PHASE2_BLOCKER_WITHDRAWN__LIVE_TTRACER_TRANSPORT_EXISTS`.
`PHASE2_BLOCKER.md` (commit `199e549`) is kept unchanged as historical evidence. This file
supersedes its verdict.

## The error

`PHASE2_BLOCKER.md` §Evidence 5 states that nothing in the repository reads UE T port 2023
live or runs a second T client. **That is false.** A committed, causal live transport already
exists:

| Evidence | What it does |
|---|---|
| `rl_agent/ue_mcs_backlog_calibration_v1/runner.py:330-348` (`start_telemetry`) | Launches OAI `multi -p 2023 -lp 2123` (and 2021→2121 for the gNB), then attaches `record` to the relay. |
| `rl_agent/ue_n2_oai_ul_calibration_smoke.py:259-307` (`LiveCsv`) | Live `csv -f` reader with host receipt timestamps. |
| `rl_agent/ue_mcs_backlog_near_capacity_v1/runner.py:399-418` | Attaches live `NRUE_MAC_DCI_GRANT`, `NRUE_MAC_RLC_BUFFER_STATUS` and `NR_PDCP_TX_SDU` readers to relay 2123 beside `record`. |
| `rl_agent/ue_mcs_backlog_near_capacity_v1/capacity_runner.py:451-486` (`aggregate_rlc_ticks`) | Aggregates per-LCID RLC rows into scheduler ticks. |
| `OAI/.../common/utils/T/tracer/multi.c` | Serves several tracer clients at once. |

Real live output exists too, for example
`rl_agent/experiments/ue_production_queue_capture_v1/20260928_214832_live/cells/*/ttracer/ue/primer_{dci,rlc,pdcp}_live.csv`.

Root cause: my search used the literal patterns `tracer/multi` and `/multi `. The code builds
the path as `troot / "multi"`, so neither pattern matched. Evidence 1–4 still describe the
post-run parser correctly. The conclusion drawn from them, that no online feed exists, did not
follow.

## What this changes

- The missing piece is an **actor-facing bounded telemetry provider**, not a live event
  transport. It is implemented in `ue_telemetry_provider_v2.py`. It reuses `multi` and
  `csv -f` unchanged, keeps `record`, and never tails `ue.raw`.
- The existing `LiveCsv` is **not** used on the actor path. It keeps an unbounded `rows` list
  and writes plus flushes an audit file per line. The v2 readers drain into fixed-capacity
  caches and publish an immutable snapshot. Audit output is written by a separate thread.
- The whole-cell median `ClockBridge` in `ue_production_queue_capture_v1/parse.py` stays
  post-hoc and is not used. v2 builds its own past-only REALTIME→MONOTONIC_RAW bridge.
- The NDI observation in the blocker's "Secondary observation" stands, now resolved: the live
  provider accepts NDI 0 and 1 (NDI is a toggle). The post-run parser's `ndi == 1` filter is
  not copied.

## Still valid from the blocker

- Phase 0 and Phase 1 are unaffected.
- The post-run parsing facts in Evidence 1–4 are correct.
