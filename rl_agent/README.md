# UE-Side SceneSense Agent Workspace

Status: **action/model characterization complete; controller harness not yet
implemented**. Last reconciled 2026-07-16.

See `../RL_AGENT_PLAN.md`, `../SCENESENSE_RL_SCHEMA.md`, and
`../SCENESENSE_MONTHLY_CHECKLIST.md` for the research plan and exit criteria.

## What exists

- `sweep_runner.py`, `sweep_analyze.py`, and `accuracy_aggregate.py` — static
  experiment fan-out and deterministic aggregation.
- `m_prime/`, `run_pipeline_m_prime.sh`, and `gate_a_check.py` — drop-aware
  M-prime training and model-level acceptance gating.
- `feature_ae/` — earlier standalone feature-AE experiments. Their historical
  object-head collapse is superseded by the integrated models.
- `ae_integrated/` — integrated AE-128/64/32 and no-AE model training/evaluation.
- `PERMODEL_KNOB_MATRIX.md` and `PERMODEL_KNOB_MATRIX_GROUPED.md` — current
  authoritative 42-profile AE/quantization/ROI action tables.
- `OAI_AB_RESULTS.md` — live single-UE OAI compression A/B.
- `OAI_CONFIG_ANALYSIS.md` — pre-sweep configuration hypothesis; final outcome
  is in `../oai_config_sweep/OAI_CONFIG_FINDINGS.md`.
- `REQUIREMENTS_AND_RL_DESIGN.md` — current dynamics/staleness-aware controller
  requirements.

`COMPLETE_KNOB_MATRIX.md`, `MONTH2_LOG.md`, and `AE_ACTION_OPTIONS.md` are useful
historical snapshots, but their pre-integrated-AE conclusions are not the
current action-model truth.

## What does not exist yet

- No `controller/` package for trace joining, action catalogs, reward scoring,
  policy replay, or LinUCB/DQN.
- No controller `guardrails.yaml` or accept/clamp/reject implementation.
- No learned-vs-best-fixed-vs-heuristic result.
- No online action execution. Keep it disabled until offline replay passes.

## Next implementation order

1. Build the replay trace loader and join action/task/OAI/staleness evidence.
2. Implement the route-masked action catalog and reward scorer.
3. Run simple static and heuristic baselines.
4. Add controller-level guardrails with accepted/clamped/rejected reporting.
5. Train/evaluate LinUCB; use DQN only if temporal behavior requires it.
6. Add controlled channel impairment and multi-UE contention after replay
   sanity checks.

## Sweep runner usage
```bash
python3 sweep_runner.py configs/static_sweep_quant_entropy.json --dry-run   # print commands
python3 sweep_runner.py configs/static_sweep_quant_entropy.json             # run them
```
Writes each variant to its own run dir + a `sweep_manifest.json` (variant → command → dir → status).
