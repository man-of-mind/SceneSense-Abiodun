# Map-freshness budget and latency presentation tables

## Definition used

An action **meets a physical map-freshness budget** here when its
counterfactual map age is no greater than the budget for at least 50%
of the complete Route-B observation time. This is majority-time
compliance, not a hard per-frame or worst-case guarantee. The simulator
uses measured 288-cell transport/reassembly behavior, final optimized
edge calibration, and strict latest-only/no-expiry scheduling.

## Number of eligible actions

Each entry is `actions / 72`; the parenthesis gives the best fraction
of route time achieved by any action at that budget.

| Budget | Favorable Stable | Mid Variable | Adverse Stable | Fade Recovery |
|---:|---:|---:|---:|---:|
| 150 ms | 0/72 (8.7%) | 0/72 (6.8%) | 0/72 (1.3%) | 0/72 (6.7%) |
| 200 ms | 0/72 (36.7%) | 0/72 (32.6%) | 0/72 (9.9%) | 0/72 (33.4%) |
| 250 ms | 19/72 (68.2%) | 12/72 (65.3%) | 0/72 (29.9%) | 12/72 (65.8%) |
| 300 ms | 41/72 (86.6%) | 33/72 (85.9%) | 10/72 (56.8%) | 33/72 (84.9%) |

## Sensitivity to the required route-time fraction

| Budget | Network profile | ≥25% of time | ≥50% of time (primary) | ≥75% of time | Best action | Best fresh time |
|---:|---|---:|---:|---:|---:|---:|
| 150 ms | Favorable Stable | 0 | 0 | 0 | 71 | 8.7% |
| 150 ms | Mid Variable | 0 | 0 | 0 | 65 | 6.8% |
| 150 ms | Adverse Stable | 0 | 0 | 0 | 65 | 1.3% |
| 150 ms | Fade Recovery | 0 | 0 | 0 | 65 | 6.7% |
| 200 ms | Favorable Stable | 11 | 0 | 0 | 71 | 36.7% |
| 200 ms | Mid Variable | 8 | 0 | 0 | 65 | 32.6% |
| 200 ms | Adverse Stable | 0 | 0 | 0 | 65 | 9.9% |
| 200 ms | Fade Recovery | 6 | 0 | 0 | 65 | 33.4% |
| 250 ms | Favorable Stable | 42 | 19 | 0 | 71 | 68.2% |
| 250 ms | Mid Variable | 30 | 12 | 0 | 65 | 65.3% |
| 250 ms | Adverse Stable | 8 | 0 | 0 | 65 | 29.9% |
| 250 ms | Fade Recovery | 30 | 12 | 0 | 65 | 65.8% |
| 300 ms | Favorable Stable | 52 | 41 | 21 | 53 | 86.6% |
| 300 ms | Mid Variable | 46 | 33 | 13 | 59 | 85.9% |
| 300 ms | Adverse Stable | 25 | 10 | 0 | 65 | 56.8% |
| 300 ms | Fade Recovery | 47 | 33 | 12 | 71 | 84.9% |

## Model performance of majority-time-eligible actions

The exact action-by-action values are in
`eligible_action_model_performance.csv`. These rows summarize the
quality range across the eligible action set.

| Budget | Network profile | Actions | Action IDs | Vehicle F1 | Person F1 | Segmentation mIoU | Vehicle XY MAE (m) | Person XY MAE (m) |
|---:|---|---:|---|---:|---:|---:|---:|---:|
| 150 ms | Favorable Stable | 0 | — | — | — | — | — | — |
| 150 ms | Mid Variable | 0 | — | — | — | — | — | — |
| 150 ms | Adverse Stable | 0 | — | — | — | — | — | — |
| 150 ms | Fade Recovery | 0 | — | — | — | — | — | — |
| 200 ms | Favorable Stable | 0 | — | — | — | — | — | — |
| 200 ms | Mid Variable | 0 | — | — | — | — | — | — |
| 200 ms | Adverse Stable | 0 | — | — | — | — | — | — |
| 200 ms | Fade Recovery | 0 | — | — | — | — | — | — |
| 250 ms | Favorable Stable | 19 | 29, 35, 41, 46, 47, 52, 53, 57, 58, 59, 63, 64, 65, 66, 67, 68, 69, 70, 71 | 0.477–0.894 | 0.331–0.626 | 0.130–0.682 | 0.510–1.048 | 0.843–0.966 |
| 250 ms | Mid Variable | 12 | 35, 41, 47, 52, 53, 58, 59, 64, 65, 69, 70, 71 | 0.477–0.874 | 0.331–0.600 | 0.130–0.593 | 0.553–1.048 | 0.869–0.966 |
| 250 ms | Adverse Stable | 0 | — | — | — | — | — | — |
| 250 ms | Fade Recovery | 12 | 41, 46, 47, 52, 53, 58, 59, 64, 65, 69, 70, 71 | 0.477–0.874 | 0.331–0.600 | 0.130–0.593 | 0.553–1.048 | 0.869–0.966 |
| 300 ms | Favorable Stable | 41 | 5, 11, 16, 22, 23, 28, 29, 33, 34, 35, 39, 40, 41, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71 | 0.477–0.896 | 0.331–0.658 | 0.129–0.708 | 0.492–1.048 | 0.826–0.983 |
| 300 ms | Mid Variable | 33 | 5, 11, 17, 22, 23, 28, 29, 34, 35, 40, 41, 45, 46, 47, 50, 51, 52, 53, 56, 57, 58, 59, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71 | 0.477–0.894 | 0.331–0.654 | 0.129–0.682 | 0.498–1.048 | 0.831–0.987 |
| 300 ms | Adverse Stable | 10 | 35, 41, 47, 53, 59, 64, 65, 69, 70, 71 | 0.477–0.874 | 0.331–0.600 | 0.130–0.593 | 0.553–1.048 | 0.869–0.966 |
| 300 ms | Fade Recovery | 33 | 5, 11, 23, 28, 29, 34, 35, 40, 41, 45, 46, 47, 50, 51, 52, 53, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71 | 0.477–0.896 | 0.331–0.654 | 0.129–0.685 | 0.498–1.048 | 0.831–0.983 |

## Best balanced quality–freshness candidate

This is a diagnostic ranking by
`0.5 × (vehicle F1 + person F1) × fresh-map fraction`; it is not
yet the final RL reward. A `no` in the final column means the best
trade-off candidate still does not satisfy the primary majority-time
rule.

| Budget | Network profile | Action | Fresh time | Vehicle F1 | Person F1 | Quality × freshness | Meets majority-time rule |
|---:|---|---:|---:|---:|---:|---:|---|
| 150 ms | Favorable Stable | 58 | 6.0% | 0.789 | 0.530 | 0.040 | no |
| 150 ms | Mid Variable | 70 | 5.4% | 0.788 | 0.524 | 0.036 | no |
| 150 ms | Adverse Stable | 70 | 1.2% | 0.788 | 0.524 | 0.008 | no |
| 150 ms | Fade Recovery | 70 | 5.9% | 0.788 | 0.524 | 0.039 | no |
| 200 ms | Favorable Stable | 69 | 27.7% | 0.874 | 0.600 | 0.205 | no |
| 200 ms | Mid Variable | 70 | 29.4% | 0.788 | 0.524 | 0.193 | no |
| 200 ms | Adverse Stable | 70 | 8.8% | 0.788 | 0.524 | 0.058 | no |
| 200 ms | Fade Recovery | 70 | 29.9% | 0.788 | 0.524 | 0.196 | no |
| 250 ms | Favorable Stable | 69 | 62.2% | 0.874 | 0.600 | 0.459 | yes |
| 250 ms | Mid Variable | 70 | 62.8% | 0.788 | 0.524 | 0.412 | yes |
| 250 ms | Adverse Stable | 70 | 27.7% | 0.788 | 0.524 | 0.181 | no |
| 250 ms | Fade Recovery | 70 | 61.8% | 0.788 | 0.524 | 0.405 | yes |
| 300 ms | Favorable Stable | 69 | 84.7% | 0.874 | 0.600 | 0.625 | yes |
| 300 ms | Mid Variable | 69 | 79.7% | 0.874 | 0.600 | 0.588 | yes |
| 300 ms | Adverse Stable | 69 | 50.1% | 0.874 | 0.600 | 0.369 | yes |
| 300 ms | Fade Recovery | 69 | 77.3% | 0.874 | 0.600 | 0.570 | yes |

## End-to-end latency by network profile

Values are medians across the per-action cell medians, so each
action has equal weight. Only actions with at least one useful map
installation contribute. The direct end-to-end median is computed
independently; therefore it need not equal the sum of marginal
component medians exactly.

| Network profile | Actions | Sensor/pre-action | UE action path | Feature transfer + reassembly | Edge queue | Edge compute + publication | Result return + map install | Direct capture → install |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Favorable Stable | 71 | 37.2 ms | 24.9 ms | 54.9 ms | 0.0 ms | 57.0 ms | 15.4 ms | 218.0 ms |
| Mid Variable | 62 | 36.5 ms | 24.5 ms | 66.1 ms | 0.0 ms | 57.1 ms | 15.4 ms | 231.6 ms |
| Adverse Stable | 46 | 36.7 ms | 23.7 ms | 62.6 ms | 0.0 ms | 55.2 ms | 66.3 ms | 272.8 ms |
| Fade Recovery | 63 | 38.2 ms | 24.8 ms | 60.3 ms | 0.0 ms | 57.2 ms | 15.8 ms | 230.2 ms |

### What the six stages mean

- **Sensor/pre-action:** RGB capture to entry into seven-channel
  tensor assembly. It includes radar-window extraction and other
  work before the selected split action can influence the frame.
- **UE action path:** tensor-assembly entry to transmission start.
  This includes final input assembly, the selected front/ranker/AE
  path, feature serialization, and dispatch preparation; it is not
  the pure backbone CUDA time alone.
- **Feature transfer + reassembly:** UE transmission start to a
  complete feature at the edge. This includes the sender loop, OAI/
  RFsim uplink, and application reassembly; it is not PHY-only time.
- **Edge queue:** wait after complete feature reassembly before edge
  work starts. It is zero for the installed frames under strict
  latest-only scheduling; superseded pending frames are not installs.
- **Edge compute + publication:** reconstruction, optimized tail
  service/post-processing, and compact-result serialization. This is
  the edge-processing column; it excludes return transport and map
  installation.
- **Result return + map install:** compact result delivery after edge
  publication plus receiver/map installation. The retained timestamps
  do not split those two contributions further.

## Discussion sequence for the supervisor

1. Start with the budget-count table: tighter freshness budgets
   sharply restrict the feasible action set, especially in adverse
   conditions.
2. Use the eligible-quality table to show the policy problem: among
   actions that are fresh often enough, choose the one providing the
   best person/vehicle utility—not simply the smallest payload.
3. Show the latency breakdown. Network-sensitive transfer changes by
   profile; sensor/pre-action and much of edge compute are outside the
   action's immediate control but still determine physical map age.
4. Explain that the agent must observe input age at decision time, so
   it is not blamed for already-consumed preparation time. Reward the
   resulting physical map utility, while using action-clock timing for
   attribution and diagnosis.
5. Ask whether the control objective should require 50% route-time
   compliance, a stricter fraction, or a soft freshness reward. The
   tables deliberately expose that choice rather than hiding it.

## Limits

This is an offline counterfactual, not a new 288-cell live run.
Observed and deterministically imputed edge arrivals are both used as
in the bound source simulator. Model validation quality is action-
specific and therefore repeats across network profiles; network
conditions alter freshness and feasibility, not the frozen validation
score itself.
