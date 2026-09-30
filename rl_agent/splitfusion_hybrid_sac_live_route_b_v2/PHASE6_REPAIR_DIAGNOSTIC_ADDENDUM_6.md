# Phase 6 addendum 6: conservative repair and GT-handoff diagnosis

**Status:** `REGISTERED_BEFORE_ANY_NEW_PHASE6_EVIDENCE`, 2026-09-30. Base commit `09ba6e5`.
The machine-readable record is `phase6_repair_diagnostic_addendum_6.json`, which also binds by file hash
all five prior attempts.

Nothing in the scientific contract changes:
- the actor, the 21-D state and its scaling;
- telemetry and the 100-ms freshness limits;
- the fallback, k_min = 2, the reward and Q_perc;
- the inclusive 170-ms deadline;
- continuous mode/q execution and hold semantics;
- P0–P8.

The claim scope stays `SYSTEMS_INTEGRATION_QUALIFICATION_ONLY`.

1. **Latency boundary restored** (`RUN4_CONTRACT.md`; supersedes addendum-5 repair 1).
   - Before action-open: RGB conversion, radar rasterization, SI and P40 complete, and the causal state
     is committed.
   - Inside the unchanged 170-ms clock: the actor, 7-channel construction, front, compression, uplink,
     tail, evaluation and feedback.
   - Every frame records durable stage timestamps, and the offline tests check their order.
2. **E5 is non-vacuous.** It needs at least one real, identity-matched GT bundle and every evaluator
   start within 250 ms of GT becoming ready. It also needs zero reader exceptions, duplicate emissions
   and queue overflows. If no GT ever becomes ready, the result is `INCONCLUSIVE_NO_GT_READY`, never
   PASS.
3. **Stop only at a closed decision cycle.**
   - A new opportunity is admitted only if a complete k_min group fits; a hold is never refused.
   - A refusal assigns nothing and ends the run as `DECISION_CYCLE_BOUNDARY`.
   - The active ticket is drained before accounting.
4. **Exact GT handoff diagnosis.** For each of the three components, the evidence records:
   - the UE host write: path, identity, time, size, SHA-256;
   - the edge container view: first observed, SHA-256 at read, exact errors;
   - the actual bind mounts.

   The complete GT scratch directory is preserved create-only, with a verified manifest, before teardown
   deletes it.
5. **One-decision handshake.** Arguments: `--transmitted-budget 40 --stop-after-decisions 1
   --safety-timeout-s 60 --child-timeout-s 600`. There is no extension and no retry. The PASS criteria
   are listed in the JSON.
6. **Conditional check of at most 30 frames**, only if the handshake passes: `--transmitted-budget 30
   --safety-timeout-s 120`. Then stop. There is no 120- or 300-frame run.
