# UE-Side RL Agent — stage workspace

Network-aware split-inference control policy (see `../RL_AGENT_PLAN.md`, `../SCENESENSE_RL_SCHEMA.md`,
`../SCENESENSE_MONTHLY_CHECKLIST.md` Month 2). Everything for this stage lives here.

## Layout
- `sweep_runner.py` — generic config-fan-out job runner (workstream **H**): runs many split-inference /
  training configs, one run-folder each, concurrency-capped. Powers A/D/E/F.
- `configs/` — sweep configs (JSON). `static_sweep_quant_entropy.json` = first A sweep.
- `controller/` — (B) offline controller harness: trace-join → action catalog → reward → baselines → LinUCB.
- `feature_ae/` — (D/E) task-aware autoencoders {128,64,32} + importance head (build on
  `../checkpoints/rd_ae_b128.pt`) + per-tensor packet tagging.
- `guardrails.yaml` — (C) task floors + vulnerable-object + network-fallback rules.

## Order of work (last week of Month 2)
H (sweep runner) → A (static sweeps → Pareto) → B (offline controller + baselines + LinUCB) → C (guardrails)
→ Month-2 report. Start D+E in parallel. Sionna (G) is a separate parallel track.

## Sweep runner usage
```bash
python3 sweep_runner.py configs/static_sweep_quant_entropy.json --dry-run   # print commands
python3 sweep_runner.py configs/static_sweep_quant_entropy.json             # run them
```
Writes each variant to its own run dir + a `sweep_manifest.json` (variant → command → dir → status).
