# Phase-2 map sharing — parked status

**Status:** PARKED / NO ACTIVE EXECUTION AUTHORITY (2026-08-19)

**Owners:** Abiodun and Codex.

This package is a later-stage helper-to-recipient map-sharing workstream. It is
not the current UE-agent milestone and it does not block the UE-side inference
controller in ../rl_agent/UE_AGENT_EXECUTION_CHECKLIST.md.

No Phase-2 CARLA collection, OAI run, factor rerun, warning repair, exact-16
launch, baseline ladder, or RL training is authorized while this status is in
force. Existing files and artifacts are preserved; parking is not deletion and
does not convert formative results into failures.

## Why it is parked

The current milestone is one UE choosing SPLIT, LOCAL, or permitted SKIP so its
own edge-map contribution remains acceptably fresh. Phase 2 instead asks when
and what a helper should publish to a recipient and whether recipient-available
information improves cooperative warning/actionability. Those are different
state, action, endpoint, corpus, and evaluation problems.

The paper reframe temporarily made recipient-specific C2 the implementation
critical path. That reordered the engineering stages and displaced the
unfinished UE controller. Abiodun and Codex reversed that ordering on
2026-08-19.

## Banked work

- Recipient-specific contribution schemas, causal timestamp checks, transport,
  source tracking, map association/fusion, and replay plumbing.
- Paired positive/benign scenario designs and reviewed Town10HD_Opt geometry.
- Static-environment truth capture and evaluation-only hazard adjudication.
- Warning-repair and decision-opportunity pilot evidence.
- Suite-A/Suite-B corpus specifications and hashed design manifests.
- Factor-realization runtime, validation, postflight, and detached-runner work.
- The proposed phase-based successor in
  FACTOR_REALIZATION_CONTROL_ARCHITECTURE_V2.md.

These assets may be reused after the UE milestone; none is evidence that the
current UE placement controller works.

### Working-tree package inventory

The unfinished factor tranche is kept together by these entry points:

- `FACTOR_REALIZATION_SMOKE_V1.md`
- `FACTOR_REALIZATION_LAUNCH_RUNBOOK_V1.md`
- `FACTOR_REALIZATION_CONTROL_ARCHITECTURE_V2.md`
- `factor_smoke_runtime_contract.py`
- `factor_smoke_postflight.py`
- `../data_collection/phase2_factor_realization_runtime.py`
- `../data_collection/launch_phase2_factor_realization_smoke.py`
- `../data_collection/validate_phase2_factor_realization_smoke.py`
- `../data_collection/configs/phase2_factor_realization_smoke_v1.yaml`
- `../data_collection/configs/phase2_factor_realization_detached_v1.yaml`
- the corresponding focused tests under both packages.

Key completed/formative evidence is retained at:

- `../data_collection/experiments/phase2_calibration_audit_v1/20260818_230028_audit`
- `../data_collection/experiments/phase2_calibration_replay_v1/20260818_235054_replay`
- `../data_collection/experiments/phase2_warning_repair_screen_v3/20260819_010500_screen`
- `../data_collection/experiments/phase2_decision_opportunity_pilot_v1/20260819_013333_pilot`
- `../data_collection/experiments/phase2_decision_opportunity_analysis_v1/20260819_020611_decision`

The former Phase-2 architecture is preserved as a parked raw Mermaid copy in
`STATE_DIAGRAM_PARKED_2026-08-19.md`.

## Last scientific state

- The helper-first decision-opportunity pilot established a formative local
  sensing opportunity, but its cooperative warning surface failed nuisance
  gates and was not C2 performance evidence.
- The eight factor-corner reviews were visually plausible with clean motion,
  but only one of eight passed the quantitative factor gate.
- The quantitative rejection was correct. The workflow conflated visual
  acceptance with physical-factor realization, and the factor design sampled
  acceleration/spawn settling instead of independently controlling closing
  speed and horizon.
- The exact-16 stage therefore remains unauthorized.
- The failed eight-corner fixture is diagnostic only; no gate will be widened,
  bypassed, or post-hoc relabelled.

## Unresolved Phase-2 successor

If Phase 2 resumes, design must precede code:

RESET/SETTLE -> PRE-ROLL -> ARMED -> spatial or ETA trigger -> deliberate
hazard entry -> factor measurement -> bounded rolling capture.

An offline kinematic feasibility table must first show an interior solution for
every proposed cell. If the Cartesian grid is physically empty, Phase 2 must
use a revised transparent design or continuous realized covariates.

## Resume conditions

Phase 2 can resume only when all are true:

1. The UE checklist has produced a Stage-5 controller decision and bounded
   Stage-7 single-UE validation.
2. Abiodun explicitly authorizes helper/recipient map-sharing work.
3. A short design note names the Phase-2 contribution and explains why the
   next task cannot be answered by the UE controller.
4. The factor architecture is proven offline before another CARLA corner run.
5. A new Phase-2 checklist is created; this parked work must not silently enter
   the UE checklist.

## Preservation notes

- Do not delete the Phase-2 source, configs, accepted geometry records, or
  experiment manifests merely because the work is parked.
- The raw eight-corner review root
  /tmp/phase2_factor_corner_final_20260819_050403 is ephemeral and must not be
  cited as a durable archive. The quantitative table and forensic conclusion
  are recorded in collab/REVIEW_NOTES.md and
  FACTOR_REALIZATION_CONTROL_ARCHITECTURE_V2.md.
- Historical documents may describe Phase 2 as the critical path; this parking
  status supersedes that execution ordering without rewriting the history.
