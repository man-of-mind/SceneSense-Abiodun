# Phase 6: what happens when a frame has no eligible ground truth — decision memo

**Status:** memo only. The frozen Phase-6 behaviour is unchanged, and no option is selected here.
The choice belongs to Abiodun–Codex. Written 2026-09-29 at HEAD `f32dbf4`.

## Correction to the Phase-6 report

`PHASE6_RUNNER_REPORT.md`, risk 2, says that "about 40% of grid frames have undefined Q_perc". **That
figure is wrong.** It was counted from the first 3,000 rows in storage order, which is a biased sample.
The full grid gives the numbers below.

The database was opened with `mode=ro&immutable=1`. Its SHA-256 was checked before and after and was
unchanged.

| Population | Scenes | Undefined Q_perc | Rate |
|---|---:|---:|---:|
| FIT, `canonical_v3_05_val_30_30_s601_tm1601` | 512 | 36 | 7.03% |
| held-out, `canonical_v3_06_val_50_50_s602_tm1602` | 256 | 1 | 0.39% |
| whole grid | 768 | 37 | 4.82% |

Supporting details:

- Of 101,376 rows, 4,884 are undefined, and every one has status `UNDEFINED_NO_LOCALIZATION_ELIGIBLE_GT`.
- Undefinedness depends only on the scene: within each scene it is identical for every `(mode, q)`, with
  0 disagreeing rows.
- Undefined scenes come in runs: 11 runs with a mean length of 3.4 frames and a maximum of 10.

The report is committed runner material, so this memo supersedes that sentence rather than editing it.

## 1. Why "no eligible CARLA GT" becomes an excluded evaluator outcome

1. `offline_quality_grid.quality.evaluate_exact_quality` returns `None` when neither vehicle nor person
   has an eligible GT instance (`quality.py:134`). Q_perc is a weighted geometric mean of per-class
   localisation terms, modulated by segmentation IoU. With no eligible instance, it has no value.
   Assigning zero would score an empty scene as a total perception failure.
2. `run4_live_wire_v2.quality_feedback` maps `None` to `EVALUATOR_FAULT` with reason
   `QUALITY_UNDEFINED_NO_ELIGIBLE_GT`. Three other cases map to the same excluded kind:
   - `GROUND_TRUTH_UNAVAILABLE` (the GT read failed);
   - `EVALUATOR_EXCEPTION`;
   - evaluator-queue overflow, which is flagged as infrastructure.
3. `run4_contract.resolve_reward` marks `EVALUATOR_FAULT` as `learning_included = False`. The Run-4
   reward schema has no value for a defined-but-empty scene.

## 2. Why the frozen previous-outcome contract forces a new genesis session

- `PreviousOutcomeV1.from_resolution` refuses excluded resolutions.
- `PolicyStateV2` requires, for every `decision_seq > 0`, the exact outcome of the immediately
  preceding decision in the same session and UE.
- So after an excluded outcome, the next decision cannot be built in that session:
  1. `RewardHoldControllerV2._resolve` sets `_session_broken`.
  2. `previous_for_next_decision` then raises `SessionBreakRequired`.
  3. `Run4DecisionEngineV2.plan_frame` opens a new controller session at genesis and counts
     `session_rollovers`.

Only a genesis state (decision 0, all previous-outcome features zero) is legal without a previous
outcome. A stale carry-forward of an older outcome would break the "immediately preceding" rule, so it is
not offered as an option.

Timeouts are **not** rollovers. A reward frame that never reaches the edge, or whose feedback misses
170 ms, resolves as `TIMEOUT`. That outcome is learning-included and is a legal previous outcome.

## 3. Expected session-rollover rate

**Per decision.** A rollover happens exactly when a decision's single reward-requested frame produces an
excluded outcome. The rate is therefore

p ≈ P(no eligible live GT) + P(GT unavailable) + P(evaluator exception or overflow).

**Live Route-B measurement.** The live quality-feedback probe
`experiments/splitfusion_quality_feedback_probe_v1/20260916_action50_favorable_adverse_retry4` ran the
same Route-B collector and live GT path. It also used the same eligibility rule as Phase-6
`live_measurement`: every live GT object, with GT count = tp + fn.

| Cell | Evaluated frames | No eligible GT | GT unavailable | `perception_metrics.csv` SHA-256 |
|---|---:|---:|---:|---|
| favorable | 271 | 4 (1.48%) | 0/300 | `34c6ecf4…a5eb` |
| adverse | 273 | 6 (2.20%) | 0/300 | `ea5d3de5…d04` |

The no-GT frames sit in one stretch of the route in both cells: frames 712–720 (favorable) and 705–715
(adverse). This is a route property that repeats between runs, not random noise.

**Expected count for a 300-frame run.**
- A run has at most about 150 decisions (k_min = 2), fewer after fallback.
- The live Route-B rate gives **about 2–3 rollovers**, grouped at that route stretch. A decision opened
  at genesis inside the stretch can itself roll over again.
- As bounds from other scene distributions: the FIT-grid rate of 7.0% would give about 10.5, and the
  held-out rate about 0.6.
- Evaluator exceptions and queue overflow have not been measured on the Phase-6 code. The run counts and
  reports them.

## 4. Why genesis resets undermine a clean policy-performance test, and how much at this rate

1. **The actor never saw a mid-route genesis state in training.**
   - The training scene source drops every scene with any undefined `q_perc`
     (`ue_production_transport_model_v2/scene_source.py:147-152`: 476 of 512 FIT scenes kept).
   - So training never produced an excluded evaluator outcome.
   - Each seed ran one persistent session, and the orchestrator allows a genesis state only at ordinal 0
     (`modeled_smoke_orchestrator.py:1374-1395`). That is 1 genesis state in 40,288 decisions.
   - A live rollover therefore feeds the actor a state it has essentially never seen: a genesis encoding
     paired with a mid-route scene and radio state.
2. **The reward statistics are censored, and not at random.** Excluded outcomes drop out of the reward.
   The ones dropped are exactly the object-free scenes, so reported reward and success rate are
   *conditional on eligible GT*, not a policy-level expectation.
3. **The causal chain is cut.** Each rollover ends the previous-outcome dependence that defines the
   semi-Markov process the policy was trained on. The decision after a rollover shows how the policy
   behaves at genesis, not how it behaves in the process.

At the measured Route-B rate, the affected decisions are about 1.5–2% of the total, bunched in one route
stretch. That makes this a real validity limitation, not a dominant one. Even so, (1) and (2) are
structural: no rate makes the conditional reward an unconditional policy measure. The frozen plan
already lists reward mean and success rate as reported-not-gated (`live_qualification_300_v2.json`).

## 5. The three bounded choices (none is selected)

| | (a) Define a no-object / no-eligible-GT reward in Run 5, retrain | (b) Add an explicit censored/unavailable outcome state in Run 5, retrain | (c) Keep current behaviour for Phase 6 as systems-integration qualification only |
|---|---|---|---|
| What changes | A new reward-schema version gives the empty scene a principled value, for example from false-positive counts or pixel false-positive rate, which the grid rows already carry (`loc_*_fp`, `seg_*_pred_pixels`). The 37 undefined grid scenes are re-derived. | The previous-outcome encoding gains an explicit "outcome unavailable" class, which changes the feature schema. The censored decision gives no reward sample, but the next state is defined, so the session continues. | Nothing. Rollovers are counted and reported. |
| Scientific validity | Valid if the value is derived from a first-principles definition of empty-scene perception quality, pre-registered before any data. An arbitrary constant would silently change the objective. | Valid. This is the standard treatment of censored rewards, and it covers GT-unavailable and evaluator exceptions the same way. The policy never learns an action's value in an empty scene. | Valid for the P0–P8 gates (identity, telemetry, hold/ticket, accounting, downlink feedback, no infrastructure fault). **Not** valid as a policy-performance claim. |
| Removes rollovers | Only those from no-object scenes. GT-unavailable and exceptions still roll over unless (b) is also adopted. | Yes, for every excluded outcome. | No. About 2–3 per run on Route-B. |
| Cost | Reward-schema change, grid re-derivation, Run-5 retraining and pre-registration. | Feature-schema change (21-D to a new version), training environment generating censored outcomes at the measured rate, Run-5 retraining and pre-registration. | None. The frozen Run-4 actor and Phase-6 runner are used as committed. |
| Compatible with the others | (a) and (b) can be combined. | Can be combined with (a). | Can sit alongside (a) and/or (b) for Run 5. |

In every option, the new sidecar (`sidecar.py`) should be registered in the Run-5 checkpoint callback.
Then any retrained actor is directly cold-loadable at every checkpoint.

## Evidence files (read-only)

- `experiments/splitfusion_hybrid_sac_quality_grid_v1/20260918_exact_continuous_q_grid_a1b_full/quality_rows.sqlite3`
  (byte-identical before and after)
- `experiments/splitfusion_quality_feedback_probe_v1/20260916_action50_favorable_adverse_retry4/cells/*/perception_metrics.csv`
- Code: `run4_live_wire_v2.py:334-360`, `reward_hold_controller_v2.py:209-277`,
  `phase6_decision_engine_v2.py:224-226`, `quality.py:122-135`, `scene_source.py:147-152`,
  `modeled_smoke_orchestrator.py:1374-1395`
