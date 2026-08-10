# Track A estimator-quality sensitivity — smoke

**Scope:** fixed epsilon=2.0 m, preferred core=90 KiB, range<=25 m, `ucb_k=0`, and `c1_pessimism_factor=0.70`; shielded oracle over the same three held-out vehicle replays. Only telemetry lag and estimate noise vary. No CARLA, OAI, LOCAL, RL, or model training was run.

## Paired finding

The baseline lag=2/noise=0.05 cell has 41.99% full-GT conditional false rejection. The idealized lag=0/noise=0 cell has 41.99%, a paired recovery of 0.00 percentage points in this surrogate.

The tested estimator settings do not explain the headline false-reject gap: the idealized estimator recovers 0.00 percentage points and the entire tested grid spans only 0.02 points. This falsifies the prior hypothesis that lag/noise drives the approximately 42% rate in this fixed surrogate.

Estimator settings still change as many as 324/1699 raw-safe sets, but only 13/1699 selected actions. Reward/preference narrowing and map-state dynamics absorb most availability changes at this operating point.

The residual at lag=0/noise=0 is not labelled irreducible: speed uncertainty, observation mismatch, worst-object aggregation, and map-state trajectory remain mixed in this three-episode vehicle-only corpus. Those mechanisms need a separate attribution diagnostic before changing the shield or reward.

## Grid

| cell_id         |   false_reject_count |   true_feasible_frame_count |   false_reject_conditional_pct |   false_reject_recovered_pp |   matched_false_reject_conditional_pct |   matched_false_reject_recovered_pp |   matched_false_admit_count |   admitted_send_count |   matched_false_admit_conditional_pct |   split_pct |   capture_attempt_pct |   over_budget_pct |   mean_matched_true_scored_reward_finite |   finite_matched_reward_delta |   selected_action_changes_vs_baseline |   raw_safe_set_changes_vs_baseline |
|:----------------|---------------------:|----------------------------:|-------------------------------:|----------------------------:|---------------------------------------:|------------------------------------:|----------------------------:|----------------------:|--------------------------------------:|------------:|----------------------:|------------------:|-----------------------------------------:|------------------------------:|--------------------------------------:|-----------------------------------:|
| lag0__noise0.00 |                  414 |                         986 |                        41.9878 |                      0      |                                37.4363 |                             -0.1381 |                           0 |                    17 |                                     0 |      5.8858 |                4.7087 |           56.6215 |                                   0.4385 |                       -0.0011 |                                    11 |                                324 |
| lag2__noise0.05 |                  414 |                         986 |                        41.9878 |                      0      |                                37.2982 |                              0      |                           0 |                    15 |                                     0 |      5.827  |                4.7087 |           56.5627 |                                   0.4396 |                        0      |                                     0 |                                  0 |
| lag4__noise0.10 |                  413 |                         984 |                        41.9715 |                      0.0163 |                                37.2449 |                              0.0533 |                           0 |                    15 |                                     0 |      6.0035 |                4.8264 |           56.5627 |                                   0.4358 |                       -0.0038 |                                    13 |                                269 |

## Guardrails

- False-reject percentages are conditional on a nonempty clairvoyant raw-safe set; counts are shown.
- Matched/tracked and strict full-GT metrics remain separate; perception misses are not shield errors.
- Any zero false-admit estimate is denominator-limited and must be read with its counts/Wilson interval.
- This sensitivity diagnoses a deterministic table-composed surrogate; it does not calibrate a live residual/conformal uncertainty model.

## Artifacts

Run directory: `rl_agent/policy/experiments/estimator_sensitivity_smoke/20260810_211707`

See `per_frame_metrics.csv`, `summary.csv`, `per_replay_summary.csv`, `replay_registry.csv`, `resolved_config.yaml`, `manifest.json`, and `figures/estimator_quality_surface.{png,pdf}`.
