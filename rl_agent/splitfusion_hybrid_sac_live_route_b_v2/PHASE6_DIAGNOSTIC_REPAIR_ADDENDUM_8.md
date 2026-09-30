# Phase 6 addendum 8: keep route-failure evidence and make pre-warm timing interpretable

**Status:** `REGISTERED_BEFORE_ANY_NEW_PHASE6_EVIDENCE`, 2026-09-30. Base commit `f560fcd`.
The machine-readable record is `phase6_diagnostic_repair_addendum_8.json`, which binds all seven prior
attempts by hash.

This is diagnostic only. Nothing else changes: the actor, reward, 21-D state, action semantics,
timeout, scheduler, GT queue, transport and policy gates.

1. **Route failure is kept.**
   - `record_route_detail` saves the complete `run_route_b` detail, create-only, to
     `phase6_artifacts/route_detail.json`. It runs immediately after the route returns and before the
     collector assertion, so `collector=None` no longer hides the error.
   - Population-event files created during the route are copied into the attempt.
2. **The CARLA log is kept.** `preserve_service_log` copies `carla_server.log`, create-only, into the
   attempt directory after CARLA stops and before the service directory is deleted. This happens on
   both success and failure.
3. **Hot repeat.**
   - After the first pass over all 36 warm paths, the same paths run once more, immediately, with
     identical tensor content and shapes and new isolated identities.
   - First-pass and hot-repeat times are recorded separately for every path, with P50/P95/max overall
     and per mode, and mode 11 at its highest q.
   - No acceptance threshold is introduced. READY requires both passes.
4. **One live attempt.** The v7-style handshake with a 40-frame limit, stop after one decision and a
   60-s safety timeout; judged by the unchanged `handshake_verdict_v2`; no retry and no 300-frame run.
