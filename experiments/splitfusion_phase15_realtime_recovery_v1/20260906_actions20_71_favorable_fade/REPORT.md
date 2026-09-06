# Phase-15 real-time recovery: four-cell post-repair validation

- Terminal: `SPLITFUSION_PHASE15_REALTIME_RECOVERY_NOT_VALIDATED`
- Classification: `POST_REPAIR_REALTIME_QUALIFICATION`
- Bound audit: `experiments/splitfusion_phase15_retry4_latency_audit_v1/20260906_root_cause_audit` (`ROOT_CAUSE_LOCALIZED`)
- Structural integration qualification (16 cells, not rerun): `experiments/splitfusion_16_cell_live_carla_oai_pilot_v1/20260905_live_carla_oai_pilot_retry4`
- 288-cell campaign: not authorized and not launched.

| Cell | Action | Profile | Coverage | Sent | Timely installs | Median AoI ms | Monotone | Teardown |
|---|---:|---|---:|---:|---:|---:|---:|---|
| a20__favorable_stable | 20 | FAVORABLE_STABLE | n/a | None | 0 | n/a | n/a | NO |
| a71__favorable_stable | 71 | FAVORABLE_STABLE | n/a | None | 0 | n/a | n/a | NO |
| a20__fade_recovery | 20 | FADE_RECOVERY | n/a | None | 0 | n/a | n/a | NO |
| a71__fade_recovery | 71 | FADE_RECOVERY | n/a | None | 0 | n/a | n/a | NO |

## Registered gates

- `every_cell_reached_a_terminal_and_passed_structural_checks`: FAIL
- `no_dense_label_map_on_the_radio_return_path`: FAIL
- `all_counter_identities_reconcile`: FAIL
- `cold_teardown_verified_for_every_cell`: FAIL
- `nonzero_on_time_installations_in_every_cell`: FAIL
- `required_evaluation_masks_are_hash_valid`: FAIL
- `downstream_queue_depth_remains_bounded`: PASS
- `installed_frame_aoi_does_not_increase_monotonically`: PASS
- `no_seconds_long_median_install_aoi`: FAIL
- `expired_frames_are_dropped_before_later_expensive_stages`: PASS
- `a71_median_aoi_not_above_a20_beyond_preregistered_tolerance`: FAIL
- `fade_recovery_returns_to_steady_state_without_retained_backlog`: PASS
- `preparation_coverage_against_unchanged_target`: FAIL

## Preparation coverage

Coverage remains below the unchanged 0.95 target. The target was not weakened; the sustainable measured preparation FPS and the remaining bottleneck are reported per cell in `PROSPECTIVE_EVALUATION.json` and `cell_summary.csv`.
