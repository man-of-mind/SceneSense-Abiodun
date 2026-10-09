# splitfusion_quality_pilot_design_v1: quality/reward review and seven-action pilot design (analysis only)

Offline, read-only analysis behind
`experiments/splitfusion_quality_pilot_design_v1/20261005_v1/DECISION_MEMO.md`.
Nothing here trains, launches or approves anything.

| File | What it holds |
|---|---|
| `analysis.py` | All numbers in the memo: exact Q_perc recomputation, per-class tolerance/tail/FP tables, Q_perc concealment, reward incentive, route-tick paired comparisons, deadline-shift sensitivity, grid frontier and reference, static headroom, cross-fitted SI/P40 rule, acceptance snapshot |
| `test_analysis.py` | Definitions tied to the approved spec and the Run-4 contract; exact Q_perc reproduction (312/312); 253/512 FIT and 103 held-out coverage |

Run from `abiodun/`, CPU only:

```bash
env -u PYTHONPATH CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  python3 -m rl_agent.splitfusion_quality_pilot_design_v1.analysis \
  experiments/splitfusion_quality_pilot_design_v1/20261005_v1/ANALYSIS.json
env -u PYTHONPATH CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  rl_agent.splitfusion_quality_pilot_design_v1.test_analysis
```

OTR (on-time within-tolerance localisation) = 1[receipt <= 170 ms] × class-weighted (person 0.6, vehicle 0.4)
share of eligible objects matched within 1.0 m (vehicle) or 1.2 m (person). It is a proposal for the pilot,
not an approved reward.
