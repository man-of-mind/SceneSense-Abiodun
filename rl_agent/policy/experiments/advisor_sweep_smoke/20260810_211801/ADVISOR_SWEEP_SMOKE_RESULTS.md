# Track A advisor sweep — smoke

**Status:** advisor-facing characterization complete; no epsilon, preferred-core, or range value is selected by this report.

Scope is a two-corner smoke test at fixed `ucb_k=0` and C1=0.70, using the same three held-out vehicle replays, paired channel seeds, and per-tick latency common random numbers. Both the deployable shielded oracle and non-deployable clairvoyant upper bound are shown. No CARLA, OAI, LOCAL, RL, or model training was run.

## Headline: per-epsilon feasibility

|   epsilon_m | controller   |   frames |   over_budget_pct |   feasible_pct |   split_pct |   capture_attempt_pct |   false_reject_conditional_pct |   matched_false_reject_conditional_pct |   matched_false_admit_count |   admitted_send_count |   matched_false_admit_conditional_pct |   matched_false_admit_ci95_high_pct |
|------------:|:-------------|---------:|------------------:|---------------:|------------:|----------------------:|-------------------------------:|---------------------------------------:|----------------------------:|----------------------:|--------------------------------------:|------------------------------------:|
|         1.5 | clairvoyant  |     1699 |           44.4379 |        55.5621 |     50.5003 |               20.3649 |                         0      |                                19.4539 |                           0 |                   534 |                                0      |                              0.7142 |
|         1.5 | shielded     |     1699 |           58.7993 |        41.2007 |      5.7681 |                4.7675 |                        42.3784 |                                38.2171 |                           0 |                     8 |                                0      |                             32.4408 |
|         2.5 | clairvoyant  |     1699 |           46.2036 |        53.7964 |     62.0365 |               31.96   |                         0      |                                28.6495 |                           0 |                   669 |                                0      |                              0.5709 |
|         2.5 | shielded     |     1699 |           66.3331 |        33.6669 |      7.5338 |                4.8264 |                        48.7805 |                                43.0754 |                          17 |                    52 |                               32.6923 |                             46.2438 |

`over_budget_pct` is the direct achievability signal: no action in that controller's raw-safe set met the frame target. It is reported before interpreting reward or mode mix.

## Range boundary diagnostic (shielded controller, pooled across executed epsilon/core cells)

|   range_m |   matched_false_admit_count |   admitted_send_count |   matched_false_admit_conditional_pct |   over_budget_pct |
|----------:|----------------------------:|----------------------:|--------------------------------------:|------------------:|
|        25 |                           0 |                     8 |                                0      |           58.7993 |
|        40 |                          17 |                    52 |                               32.6923 |           66.3331 |

The 40 m result is not merely less feasible: its matched/tracked false-admit rate is materially larger. This supports retaining 25 m as the headline operating region and treating 40 m only as a diagnostic until the observation/risk residual is understood and live-validated.

## All cells

| cell_id                  | controller   |   over_budget_pct |   feasible_pct |   split_pct |   capture_attempt_pct |   degraded_tier_pct |   mean_prb_cost |   matched_false_admit_count |   admitted_send_count |   matched_false_admit_conditional_pct |   matched_false_admit_ci95_high_pct |   false_reject_count |   true_feasible_frame_count |   false_reject_conditional_pct |   matched_false_reject_conditional_pct |   mean_matched_true_scored_reward_finite |   observation_coverage_pct |
|:-------------------------|:-------------|------------------:|---------------:|------------:|----------------------:|--------------------:|----------------:|----------------------------:|----------------------:|--------------------------------------:|------------------------------------:|---------------------:|----------------------------:|-------------------------------:|---------------------------------------:|-----------------------------------------:|---------------------------:|
| eps1.5__core90__range25  | clairvoyant  |           44.4379 |        55.5621 |     50.5003 |               20.3649 |             12.5368 |          0.1442 |                           0 |                   534 |                                0      |                              0.7142 |                    0 |                         944 |                         0      |                                19.4539 |                                   0.3086 |                    45.1805 |
| eps1.5__core90__range25  | shielded     |           58.7993 |        41.2007 |      5.7681 |                4.7675 |              4.2966 |          0.0278 |                           0 |                     8 |                                0      |                             32.4408 |                  392 |                         925 |                        42.3784 |                                38.2171 |                                   0.4177 |                    45.1805 |
| eps2.5__core129__range40 | clairvoyant  |           46.2036 |        53.7964 |     62.0365 |               31.96   |             26.8393 |          0.2386 |                           0 |                   669 |                                0      |                              0.5709 |                    0 |                         914 |                         0      |                                28.6495 |                                   0.3333 |                    37.3374 |
| eps2.5__core129__range40 | shielded     |           66.3331 |        33.6669 |      7.5338 |                4.8264 |              5.7681 |          0.0351 |                          17 |                    52 |                               32.6923 |                             46.2438 |                  380 |                         779 |                        48.7805 |                                43.0754 |                                   0.4349 |                    37.3374 |

## Interpretation

- `aggregation=max` is a worst-object, per-frame bottleneck: one in-scope object's risk can make every whole-frame action infeasible. Object-selective transmission/scheduling is the explicit phase-2 relief, not an unmodeled fix applied here.
- Range<=25 m is the headline measured-validity operating region. The 40 m cells are labelled extrapolative sensitivity and must not silently replace the 25 m result.
- The preferred-core value is a quality preference tier, not a hard safety floor; degraded profiles remain available under graceful degradation.
- Shielded-versus-clairvoyant gaps measure observability/estimation cost inside the surrogate; the clairvoyant controller is not deployable.
- Vehicle-only replay, thin admitted-SPLIT denominators, and 90-KiB-anchored payload/FPS projections remain. Zero point estimates require their counts and descriptive Wilson intervals.

## Decision boundary

This run supplies evidence for the advisor discussion only. It does not rank or lock epsilon, the preferred segmentation core, or range, and it does not authorize LOCAL or RL training.

## Artifacts

Run directory: `rl_agent/policy/experiments/advisor_sweep_smoke/20260810_211801`

See `per_frame_metrics.csv`, `summary.csv`, `per_epsilon_summary.csv`, `per_replay_summary.csv`, `replay_registry.csv`, `resolved_config.yaml`, `manifest.json`, and `figures/advisor_achievability_frontier.{png,pdf}`.
