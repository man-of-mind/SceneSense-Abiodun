# Phase-15 real-time recovery: four-cell post-repair validation

- Terminal: `SPLITFUSION_PHASE15_REALTIME_RECOVERY_VALIDATED`
- Classification: `POST_REPAIR_REALTIME_QUALIFICATION`
- Bound audit: `experiments/splitfusion_phase15_retry4_latency_audit_v1/20260906_root_cause_audit` (`ROOT_CAUSE_LOCALIZED`)
- Structural integration qualification (16 cells, not rerun): `experiments/splitfusion_16_cell_live_carla_oai_pilot_v1/20260905_live_carla_oai_pilot_retry4`
- 288-cell campaign: not authorized and not launched.
- Offline re-evaluation of immutable cell outputs from `experiments/splitfusion_phase15_realtime_recovery_v1/20260906_actions20_71_favorable_fade_retry1`; no cell was re-run and no prior artifact was modified.

| Cell | Action | Profile | Coverage | Sent | Timely installs | Median AoI ms | Monotone | Teardown |
|---|---:|---|---:|---:|---:|---:|---:|---|
| a20__favorable_stable | 20 | FAVORABLE_STABLE | 0.920 | 2839 | 176 | 460.2 | 0.503 | yes |
| a71__favorable_stable | 71 | FAVORABLE_STABLE | 0.889 | 3191 | 1968 | 357.0 | 0.535 | yes |
| a20__fade_recovery | 20 | FADE_RECOVERY | 0.908 | 3285 | 3 | 485.5 | 0.667 | yes |
| a71__fade_recovery | 71 | FADE_RECOVERY | 0.915 | 2823 | 1749 | 353.4 | 0.507 | yes |

## Registered validity gates

- `every_cell_reached_a_terminal_and_passed_structural_checks`: PASS
- `no_dense_label_map_on_the_radio_return_path`: PASS
- `all_counter_identities_reconcile`: PASS
- `cold_teardown_verified_for_every_cell`: PASS
- `nonzero_on_time_installations_in_every_cell`: PASS
- `required_evaluation_masks_are_hash_valid`: PASS
- `downstream_queue_depth_remains_bounded`: PASS
- `installed_frame_aoi_does_not_increase_monotonically`: PASS
- `no_seconds_long_median_install_aoi`: PASS
- `expired_frames_are_dropped_before_later_expensive_stages`: PASS
- `a71_median_aoi_not_above_a20_beyond_preregistered_tolerance`: PASS
- `fade_recovery_returns_to_steady_state_without_retained_backlog`: PASS

## Performance findings (reported, not validity gates)

Preparation coverage remains below the unchanged 0.95 target. The target was NOT weakened and is still reported as unmet. The registered campaign contract classifies preparation coverage as measured performance, not structural invalidity, so it is reported here rather than gating the verdict. Per-cell sustainable measured preparation FPS and the remaining bottleneck are in `PROSPECTIVE_EVALUATION.json` and `cell_summary.csv`.

- `a20__favorable_stable`: coverage 0.920 < 0.95, sustainable 9.08 fps, median front 23.1 ms, median frozen tail 110.0 ms
- `a71__favorable_stable`: coverage 0.889 < 0.95, sustainable 8.89 fps, median front 19.7 ms, median frozen tail 112.0 ms
- `a20__fade_recovery`: coverage 0.908 < 0.95, sustainable 8.74 fps, median front 22.1 ms, median frozen tail 113.4 ms
- `a71__fade_recovery`: coverage 0.915 < 0.95, sustainable 9.03 fps, median front 18.0 ms, median frozen tail 110.1 ms
