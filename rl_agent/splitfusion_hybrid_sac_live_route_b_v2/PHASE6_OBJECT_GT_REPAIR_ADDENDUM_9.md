# Phase-6 addendum 9: simulator object-GT repair (prospective)

Base commit `43b82b6`. Claim scope: SYSTEMS_INTEGRATION_QUALIFICATION_ONLY.
This file was registered before any new Phase-6 evidence was collected. The
machine-readable version is `phase6_object_gt_repair_addendum_9.json`.

## v8 result (unchanged)

This addendum does not change how the v8 handshake
(`20260930T043946Z_phase6_gt_handshake_v8`) is reported:

- policy frame 1255 was transported and map-installed about 123.5 ms after action-open;
- a valid Q_perc of 0.4598 was eventually produced;
- the reward ACK missed 170 ms because simulator object-GT generation was late;
- the qualification is FAIL, which is not a conclusion about policy performance.

All 51 v8 files are bound by SHA-256 in the JSON addendum.

## Diagnosis

The v8 reward frame spent 372.4 ms building object rows, after waiting
21.5 ms behind a running LOW ticket. The pinned builder runs at 140 m and
counts radar support against the whole radar window for every projected
actor. The pinned `_ground_truth` then keeps only rows at or under 40 m. In
v8 it kept 2 objects.

## Repair

Everything goes through Phase-6 seams. No pinned file is edited.

1. **Instrumentation.** For each ticket the log records:
   - builder wall time and thread-CPU time;
   - context switches;
   - actor counts at each stage;
   - time per stage;
   - eligibility-filter time, file-write time and HIGH queue wait, kept separate;
   - overlap with front/codec/send, all on `CLOCK_MONOTONIC_RAW`.
2. **40-m limit.** `build_object_rows_v2` runs the same actor-origin distance
   check before generating bbox corners. Beyond 40 m it skips only radar
   support and row construction. Projection and the stationary tracker still
   run up to 140 m, so tracker state and every retained row stay bit-identical.
3. **LOW never runs during a reward ticket.** The reward gate opens at
   decision open and closes when the HIGH objects file is written, or when the
   frame is not sent. While the gate is open:
   - an arriving or queued LOW ticket is skipped with an explicit status;
   - a running LOW ticket yields at the next actor;
   - HIGH is never preempted.
4. **Early start.** At decision open, a single prefetch thread computes the
   pure per-actor geometry and radar support from the immutable same-frame
   frozen snapshot. The evaluation worker still does distance, tracker update
   and row assembly, using the ticket's exact camera location. The cache is
   bound to the snapshot object and to the SHA-256 of each input array. A miss
   or a mismatch is recomputed inline.
5. The reward-only builder is **not implemented** unless Phase C shows items 1-4
   are insufficient.

## Gates

- **Offline parity**, in both interpreters and with real `carla` value types.
  On identical frozen snapshots, the old and new paths must give:
  - the same eligible targets, class and world x/y;
  - the same tracker state;
  - bit-identical Q_perc from identical predictions.

  The semantic path and identities must be untouched.
- **Phase C (CARLA only, about 30 frames)**, run only on a verified-idle host.
  The pass criteria are:
  - objects ready no later than action-open + 150 ms;
  - predicted feedback strictly under 170 ms;
  - HIGH queue wait under 5 ms;
  - any LOW ticket inside a reward window was preempted, with under 5 ms residual;
  - the post-run shadow parity check passes.

  If any criterion fails, stop.
- **Phase D.** One handshake, judged by the unchanged `handshake_verdict_v2`.
  - An evaluator miss is never counted as policy success.
  - An evaluator-fault terminal must arrive before 170 ms, is excluded from
    reward, and leaves the qualification non-pass.
  - No retry and no 300-frame run.
