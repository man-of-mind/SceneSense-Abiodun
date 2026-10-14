> **Superseded in part by [CORRECTION_01.md](CORRECTION_01.md) and [HANDOFF_02.md](HANDOFF_02.md)** (independent review v1: the prep stack lacked V16's storage accounting V3.v2, and contained refusals skipped plan validation; both fixed and requalified in the `_02` runs). The text below is preserved as written.

# GT geometry-refusal containment v1 — offline implementation and qualification

**Status: READY_FOR_INDEPENDENT_OFFLINE_REVIEW.** Nothing is registered, deployed or live-qualified. S1 is unchanged:
it remains INCOMPLETE_ROUTE and is not replaced or reinterpreted. No campaign was prepared.

## 1. Result in one paragraph

The tilt-aware AVO evaluator still refuses a pedestrian outside the unchanged 0.03 m ground-rule envelope. With this
candidate, that refusal no longer terminates the GT service. A versioned, default-off layer in the GT prep worker turns
exactly the two recognised envelope refusals into an explicit per-frame **GT-unavailable** outcome:
`NO_GT_DEFERRAL_REFUSED_GT_GEOMETRY_ENVELOPE_TARGET` or `..._POSSIBLE_COMPETITOR`.

The refusal travels through the service's existing BIND-refusal path, and the edge sends the retained
`GROUND_TRUTH_UNAVAILABLE` fault feedback.

- The worker stays alive and later frames score exactly.
- Every other failure is fatal exactly as before, and the original diagnostic and V7 FREEZE snapshot are preserved.
- No score, quality, target, ACK or prediction is produced for the refused frame.
- No tolerance changed, no pedestrian is silently dropped, and no history policy is new.

## 2. The S1 failure, reproduced (`replay_s1_01`, PASS 8/8)

- **Failing request:** REQUEST.json gives frame **6146**, freeze sequence **1176**. The UE's connection reset at frame
  6160 was a consequence, not the failing request.
- **Reproduction:** this is an exact **AVO-step replay**, not a full producer replay. The saved `SOURCE.bin` is
  re-admitted through a fresh FreezeGate (all digests re-verified), and S1's own AVO step
  (`splitfusion_avo_tilt_v1.quality.numeric_avo`) re-raises the live exception **verbatim**.
- **Offender:** actor 106, a TARGET at pitch 3.719° / roll 7.646°, footprint 0.0370927 m.
- **Classifier:** the diagnostics pass every content check. The full classifier returns `RAISE_SITE` here only because
  an AVO-step replay has no `composition.assemble` frame.
- **Not available:** S1's producer, conflict and owner history were not retained, so no historical state is claimed to
  be restored. The full composition path is exercised below on retained-corpus targets with the same S1 pitch/roll.
  Their footprint is identical (0.0370927 m; same pedestrian extents).

## 3. What was implemented

Package: `rl_agent/splitfusion_gt_geometry_containment_v1/`. No existing source was edited. All dependencies are
imported unchanged, and git shows them unmodified.

**`classify.recognise`** accepts an error only if all of these hold:

| Requirement | Detail |
|---|---|
| Type | Exact `LiveGtContractError`, with no cause or context |
| Raise path | The innermost frames are exactly `composition.assemble → quality.numeric_avo → score_live_avo_people → _envelope` (code-object identity) |
| Diagnostics shape | `geometry_diagnostics` has the exact schema, finite floats, and footprint **> 0.03 m** |
| Role consistency | TARGET requires role TARGET with no structural reasons; POSSIBLE_COMPETITOR requires a structural actor |
| Message integrity | The message and args equal the text `_envelope` formats |

Anything else is not recognised and stays fatal: other contract errors, malformed or missing diagnostics, forged
errors, chained errors, subclasses.

**`engine.ContainedEngine`** adopts the exact constructed `DeferredProducerEngine` in place, so the capacity-substituted
`__init__` is kept. On a recognised refusal:

1. The inherited `prepare_unbound` has already cancelled the owner transaction through the pinned producer. That is an
   exact history rollback, which the producer verifies itself.
2. The engine then checks the rollback against a pre-call snapshot: history digest (including tracker and conflict
   state), owner frames, reservation, pending transaction, payload bytes, owner/engine failure, and unbound slot. Any
   mismatch re-raises, which is fatal.
3. It clears the error's finished frames and clears only this error's sticky failure.
4. It occupies the single unbound slot with a token-less refused handle.
5. `bind` raises the existing `DeferralRefused` with reason `GT_GEOMETRY_ENVELOPE_*`. `cancel_unbound` and `close`
   release the slot; `close` without abandon fails exactly like the inherited unbound close.

**`core.PrepCoreContained`** subclasses the unchanged `PrepCoreLifetime`. It writes bounded diagnostics while the
sealed fd is still open:
- `REFUSALS.jsonl`: at most 128 lines;
- full ERROR/REQUEST/SOURCE/READY copies for the first **2** refusals only;
- `COUNTERS.json`.

A diagnostics write error is counted and never changes the outcome.

**Trampolines:**
- `worker_entry` composes the layers explicitly: AVO switch, then lifetime repair, then containment, then V4 → V3.
  This is the same order as avo_tilt → V7, and every layer is restored with a binding record.
- `service_entry` and `gc_seams` are byte-identical to avo_tilt's apart from docstrings; they read this contract.

## 4. Offline qualification (all final runs PASS)

| Evidence | Result |
|---|---|
| `tests_01.log` | 14/14: real TARGET and COMPETITOR errors recognised; 22 mutations and a forged raise site rejected with the expected reason; bbox-rotation error rejected; install/adopt guards; bounded diagnostics; write-error path; metric taxonomy; actor-visible controller trace (§5) |
| `dependency_tests_01.log` | 27/27: unchanged avo_tilt and FREEZE-lifetime test suites |
| `verify_engine_01` (in-process, recorded clocks) | 11/11. **U:** 49/49 upright frames identical to the reference on all six `verify_deferred` fields. **C:** 7 refusals (6 TARGET incl. S1 tilt twice, 1 POSSIBLE_COMPETITOR; 6 BIND-refused, 1 held), history restored, capacity back to 0; the 42 accepted frames are identical to a reference that never saw the refused frames. **S:** 43/43 target frames refused in succession, history never advanced, owner retained/reserved 0 after each, heap growth +31 kB over 33 refusals. **F:** bbox rotation stays fatal (declined `RAISE_SITE`, engine failed); an inside-envelope tilt (0.0261 m) is scored. **K:** close behaviour as specified |
| `service_proof_01` (real service, workers, memfd FreezeGate, edge evaluator, R4FB) | 22/22, detailed below |

**`service_proof_01` detail:**
- **Upright parity:** containment path vs unchanged avo_tilt path on the V3 normal scenario (send-failure cancel, edge
  terminal, horizon timeout). Outcomes, q, measurement and AVO are identical, 11/11 exact. Containment was installed and
  never triggered.
- **Contained arm:** 14 refusals (13 TARGET, 1 POSSIBLE_COMPETITOR) interleaved with 20 valid frames. The orderings
  covered GT-first, prediction-first, send failure after admission, prediction after the horizon, and back-to-back
  refusals.
  - Each refusal received exactly one `NO_GT_DEFERRAL_REFUSED_GT_GEOMETRY_ENVELOPE_*` outcome. No refused frame was
    committed, had a target admitted or carried a score.
  - Every refused frame with a prediction received `EVALUATOR_FAULT` / `GROUND_TRUTH_UNAVAILABLE` with no q.
  - All 20 valid frames scored exactly, with archives rescored.
  - Service rc 0, no errors, engine valid, owner capacity 0.
  - Prep worker fd count was constant at 4 across all 14 refusals.
  - 14 diagnostic records and 2 source copies, all hash-verified; no `BufferError` and no fatal snapshot.
- **Unchanged-path control:** each refused frame was replaced by a send failure. Valid-frame q, measurement and AVO
  are identical to the contained arm. In this control the cancelled frames were voided before admission, so they were
  never computed.
- **Fail closed:** after a contained POSSIBLE_COMPETITOR refusal, a non-zero bbox rotation still killed the worker with
  the original message. The V7 FREEZE snapshot of that frame (1535) is hash-verified, and the earlier refusal record is
  retained.

## 5. Exact outcome and actor-history semantics

Nothing new is invented: the containment reuses the existing BIND-refusal and GT-unavailable paths end to end.

| Stage | Behaviour for a contained refusal |
|---|---|
| Prep worker | FREEZE replies OK (job handle); BIND replies `REFUSED DEFERRAL_REFUSED_GT_GEOMETRY_ENVELOPE_*`; CANCEL_UNBOUND/close release it |
| Service | `bind_refused` → exactly one `NO_GT_…` outcome; any waiting prediction is released from the scorer; the edge gets `UNSCORED` whether the prediction arrives before or after; frees `prep_open`, so the next FREEZE proceeds; no archive, as for any NO_GT outcome |
| Edge | Retained `GROUND_TRUTH_UNAVAILABLE` fault: kind `EVALUATOR_FAULT`, `q_perc=None` (`phase6_edge_runtime_v2.py:350`, `edge.py:201`) |
| UE controller, receipt ≤170 ms | Accepted as `EVALUATOR_FAULT`: excluded from learning, session break, so the next decision starts from genesis (the `CLEARED` history). This is the V15 `NO_GT_FREEZE_QUEUE_FULL` precedent (frame 7304 → next decision CLEARED) |
| UE controller, receipt >170 ms | The ticket was already resolved as `TIMEOUT` at the deadline; the fault is a `LATE_ORPHAN`, so the next history is `TIMEOUT` |

The controller rows are verified by the real wire codec and `RewardHoldControllerV2` in `tests_01`. Prediction
production, edge map publication and the 170-ms deadline are untouched. The containment is entirely inside the GT prep
worker, and the service never gates map publication. V15's live NO_GT decision was `RESULT_INSTALLED` on the map.

## 6. Scientific accounting (specification; historical results unchanged)

`metrics.py` implements the taxonomy below; every policy decision stays in every operational denominator:

- **any feedback ≤170 ms / all decisions:** the unchanged 80% screen, reported first;
- **scored feedback ≤170 ms / all decisions;**
- **scored coverage:** scored / all;
- **scored with ≥1 eligible object;**
- **scored with zero eligible objects:** a legitimate result, not missing GT;
- **GT unavailable, geometry:** this candidate's refusals;
- **GT unavailable, other:** e.g. queue-full;
- **late;**
- **missing.**

Applied to the retained timelines:
- **V15:** any 1,174 vs scored 1,173 (its one queue-full refusal was timely); 85 zero-eligible; 1 other GT-unavailable.
- **V16:** 1,112 by both definitions.

**A refusal can be fast.** In the rig, prediction-first refusals reached the UE in 66–98 ms vs a median of 101 ms for
scored frames. These are rig timings, not live timings. A timely refusal counts toward the unchanged screen but is not
useful feedback. It must be shown separately and never presented as improved performance. Missing GT is **unknown**:
refused frames are in neither the quality numerator nor its denominator, so every quality figure must state its scored
coverage.

**Proposed rule for review (not introduced):** keep the per-session screen unchanged, and register prospectively:
- report geometry refusals and scored coverage per session;
- label a session `QUALITY_COVERAGE_LIMITED` if geometry GT-unavailable decisions exceed 1% of its policy decisions,
  or any 10-route-index decile loses more than 5% to them;
- quality comparisons between sessions use only scored decisions, with their coverage stated.

## 7. Remaining limitations (plain)

- **Rollback semantics:** a refused frame is never committed to GT history, exactly like a held or cancelled frame.
  Later frames' GT is what it would be had that frame never been a decision; its tracker and conflict observations are
  not accumulated. This was the only state-consistent option; the alternative under the old behaviour was session death.
- **Map delivery:** this is argued from code and one live precedent (V15 frame 7304). The offline rig has no map server,
  so this is not re-verified here.
- **Fixtures:** tilts and the competitor relocation are labelled synthetic edits on retained frames, with depth not
  re-rendered. The corpus has no natural target/structural pair within reach, so the POSSIBLE_COMPETITOR case required
  moving pedestrian 102 to 1.0 m behind target 101 (40.97 m).
- **Live cost:** each of the first two refusals writes about 11.6 MB with fsync in the FREEZE reply path, delaying the
  next FREEZE once or twice per session. Later refusals write one small fsynced line. This is unmeasured live.
- **Outcome label:** the label carries the existing transport prefix `DEFERRAL_REFUSED_`. Analyses must classify by the
  `GT_GEOMETRY_ENVELOPE_` reason (`metrics.GEOMETRY_PREFIX`), not by the prefix alone.
- **Hold/cancel case:** if a refused frame is cancelled before BIND (hold, fallback, send failure before the plan), its
  outcome is the ordinary `CANCELLED_*`/VOID. The refusal is then visible only in the engine counters and diagnostics.
- **Source binding:** recognition is bound to the code identity of the qualified tilt-aware AVO sources. Any change to
  them requires requalification; they are already hash-bound in `exact_sources`.
- **Live frequency:** one trigger in the M3 captures so far. A future session may see zero or several.

## 8. Defects found during development (fixed before the final runs)

- **Traceback cycle:** the refused error and its `_envelope` frame form a cycle that keeps the borrowed decoded views
  exported. Unbroken, the FREEZE mapping release would raise `BufferError`, turning a contained refusal into a fatal one
  (the V6 failure class). The S1 replay exposed it. The engine and replay now clear the finished frames, and the
  real-service proof confirms no `BufferError`.
- **Harness corrections:**
  - Correct targets per frame: actor 146 is structural before frame 1475, and 122 leaves the targets after frame 1615.
  - Never bind an unrecognised frame.
  - A send failure before admission is a VOID that never computes AVO, so the service scenario sends the failure after
    admission.
- These development runs went to the scratchpad only and are not evidence.

Artifacts: [HANDOFF.md](HANDOFF.md), [COMMANDS.txt](COMMANDS.txt), [MANIFEST.json](MANIFEST.json) (package, unchanged-dependency, S1-input and output SHA-256; evidence 162.7 MB; 37.39 GB free).
