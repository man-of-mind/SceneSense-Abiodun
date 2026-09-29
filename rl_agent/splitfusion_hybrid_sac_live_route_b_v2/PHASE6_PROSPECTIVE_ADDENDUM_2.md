# Phase 6: prospective addendum 2, option (c)

**Status:** `REGISTERED_BEFORE_ANY_PHASE6_DATA`, 2026-09-29. Starting commit `42bca44`.
The machine-readable record is `phase6_prospective_addendum_2.json`.

## Decision

Abiodun–Codex selected **option (c)** of
`rl_agent/splitfusion_hybrid_sac_checkpoint_sidecar_v1/PHASE6_UNDEFINED_QUALITY_DECISION_MEMO.md`
(SHA-256 `65d17836…6e41`).

**The 300-frame Phase-6 run is a systems-integration qualification. It is not a policy-performance
validation.**

This addendum is additive:
- The plan `live_qualification_300_v2.json` (SHA-256 `a36f87be…2d80`) is unchanged.
- `PHASE6_RUNNER_REPORT.md` is unchanged. Its "about 40%" figure stays superseded by the memo.
- No evidence was rewritten.

## Unchanged

- The frozen seed-43/update-10,000 actor, with boundary `b61f27a9…cebd3`.
- The 21-D state.
- The Run-4 reward, the inclusive 170-ms deadline and the registered timeout reward.
- k_min = 2, the fixed fallback (mode 11 / `q_e4` 9800 / anchor 71), and every runtime and transport
  semantic.
- Gates P0–P8, their thresholds and the verdict rule.

## Undefined quality: preserved exactly

- No eligible GT gives an excluded `EVALUATOR_FAULT` (`QUALITY_UNDEFINED_NO_ELIGIBLE_GT`).
- It has no reward value. There is no zero or neutral reward and no timeout substitution.
- No stale previous outcome is carried forward. The next policy decision opens a genesis session, which
  is counted in `session_rollovers`.

## Reporting added

`phase6_result_reporting_v2.py` is called from `evaluate_phase6`. It adds these fields to `CELL_RESULT.json`
and writes a create-only `PHASE6_RESULT_SUMMARY.md` alongside it:

- total reward-requested decisions, resolved decisions, and decisions open at stop;
- **eligible-quality denominator**, defined as the learning-included resolutions (SUCCESS, TIMEOUT,
  registered delivery/service failures);
- **excluded count and rate**, as excluded / resolved;
- **exclusion-reason histogram**, keyed `terminal:evaluator reason`;
- **session-rollover count**, and a check that sessions = rollovers + 1;
- **conditional reward mean and success rate**. Both are computed only over non-excluded resolutions, are
  labelled `CONDITIONAL_ON_NON_EXCLUDED_OUTCOMES…REPORTED_NOT_GATED`, and carry `gated: false`;
- `claim_scope = SYSTEMS_INTEGRATION_QUALIFICATION_ONLY` and `policy_performance_claim = false`;
- two integrity counters, `extra_reward_frames` and `excluded_with_reward_value`, both expected to be 0.
  Reporting never changes a verdict.

Every summary begins with this statement:

> P0–P8 PASS is a systems-integration PASS, not a policy-performance PASS. Reward mean and success rate
> are conditional on eligible GT (non-excluded outcomes) and are not gated.

## Claims not made

- Policy performance.
- Reward mean or success rate as a policy-level expectation.
- Convergence or optimality of the selected actor.

## Offline readiness (2026-09-29)

**Tests**, all run with `env -u PYTHONPATH CUDA_VISIBLE_DEVICES=`:
- package suite: 123 OK, including the 8 new `test_phase6_result_reporting_v2` tests;
- inherited suites: 191 OK;
- sidecar synthetic suite: 17 OK.

The readiness manifest was re-sealed to pin `phase6_result_reporting_v2.py` and
`phase6_prospective_addendum_2.json`: `c379782859b9c4db8de6961258a81e4927cc1a0c120fbb935e6354a53ce667c1`.

`phase6_live_runner_v2 --preflight` returned `SPLITFUSION_RUN4_PHASE6_PREFLIGHT_PASS`:
- carrier cell `a71__favorable_stable`, 300-frame budget;
- 0 external processes started, CUDA not initialized;
- radio binding and emitter pins verified;
- actor boundary `b61f27a9…cebd3`;
- this addendum verified, SHA-256 `259f7bc5…d737`.

The live run needs its own authorization. Command, **not executed**:

```bash
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=0 python3 -u -m \
  rl_agent.splitfusion_hybrid_sac_live_route_b_v2.phase6_live_runner_v2 \
  --output-root rl_agent/experiments/splitfusion_hybrid_sac_live_route_b_v2/<UTC>_phase6_live_qualification_option_c \
  --transmitted-budget 300 \
  --execute SPLITFUSION_RUN4_PHASE6_LIVE_QUALIFICATION_V2_EXECUTE
```
