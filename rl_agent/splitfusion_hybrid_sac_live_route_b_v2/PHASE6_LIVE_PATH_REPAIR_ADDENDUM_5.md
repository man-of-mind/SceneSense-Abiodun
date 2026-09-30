# Phase 6 addendum 5: three live-path repairs

**Status:** `REGISTERED_BEFORE_ANY_NEW_PHASE6_EVIDENCE`, 2026-09-30. Base commit `8cfb4ed`.
The machine-readable record is `phase6_live_path_repair_addendum_5.json`, which also binds by file hash
every prior attempt, including the failed 300-frame run `20260930T010207Z`.

These stay exactly as they were:
- the seed-43/update-10,000 actor and its weights;
- the 21-D state, its scaling and its freshness bounds, including the 100-ms SI/P40 and radio rule;
- the reward, the 170-ms deadline and k_min = 2;
- the fallback, mode 11 / `q_e4` 9800;
- continuous-q execution and the latest-only edge/map scheduler;
- the quality formula;
- Option-C semantics;
- P0–P8 and the verdict.

There is no retraining. The claim scope stays `SYSTEMS_INTEGRATION_QUALIFICATION_ONLY`.

## The three repairs

1. **Plan the action before expensive preparation.** A Phase-6 collector hook does the work in this
   order:

   | Step | Work |
   |---|---|
   | 1 | complete 4-sweep window plus synchronized RGB |
   | 2 | existing SI/P40 recipes |
   | 3 | telemetry snapshot and guard |
   | 4 | immutable `PlannedRun4FrameV2` |
   | 5 | radar tensor |
   | 6 | 7-channel input, front, codec, send |

   - The plan binds frame, capture and CARLA timestamps, the radar-window digest, the tensor sequence,
     the session/decision/ticket identity, `reward_requested`, mode, `q_e4`, the bundle and the anchor.
   - Materialization refuses any drift. A post-plan failure goes through the existing
     `transport_failed` path, so no ticket is ever left unresolved.
   - The RGB receipt instant stays the scene source time; nothing is re-stamped.
   - The actor is called at most once per decision.
   - The pinned adapter is unchanged; the hook is an instance-local proxy around `build_radar_sample`.

2. **No evaluator head-of-line blocking.**
   - Submitted tickets go into a bounded pending set, and GT is checked with short probes. A ticket is
     evaluated as soon as its GT exists.
   - A ticket whose GT is still absent after the local edge wait of `gt_timeout_s` is emitted as
     `GROUND_TRUTH_UNAVAILABLE`, which is an excluded evaluator fault.
   - Each ticket is emitted at most once, and shutdown drains or expires every pending ticket.
   - The quality calculation is byte-identical to before.

3. **Registered edge terminals reach the reward controller.**
   - `on_registered_terminal` resolves the active reward-requested policy frame as the existing
     `REGISTERED_SERVICE_FAILURE`: reward −1, no `q_perc`. It requires a full identity match.
   - This applies to `SUPERSEDED_PENDING` and to the two other terminals the Phase-6 edge emits only for
     frames it never processed: `STALE_BEFORE_EDGE` and pre-decode `STALE_BEFORE_MAP`.
   - `CREDIT_SUPERSEDED_BY_FRESHER` is recorded separately and is not counted as a loss.
   - Hold and fallback terminals, and map feedback, stay ledger-only.
   - Identical duplicates are ignored, conflicting ones fail closed, and a terminal arriving after the
     timeout is a late orphan.

## Interpretation notes

- **Radar preparation now counts against the deadline.** Action-open precedes radar-tensor preparation,
  so rasterization and RGB conversion fall inside the 170-ms window. `RUN4_CONTRACT.md` had placed
  sensor preparation before the reward clock. The new timing is stricter, and the clock is not
  restarted.
- **E5 threshold.** The head-of-line gate is measured as: every evaluated ticket starts evaluation
  within 250 ms of its GT becoming ready. Before the repair, the stall was about 2 s.

## Sequence

1. Offline tests and the real preflight.
2. One create-only 120-frame engineering qualification, scored on gates E1–E9 (see the JSON). No retry.
3. Only if every gate passes: one create-only 300-frame qualification. No retry.
