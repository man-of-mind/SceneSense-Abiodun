# Phase-15 sensor-preparation optimization qualification

- Terminal: `SPLITFUSION_PHASE15_PREPARATION_PATH_READY`
- Scope: actions 20 and 71 under `FAVORABLE_STABLE` only.
- 288-cell campaign: not launched.
- Service target: 100 ms; feedback timeout: 500 ms (reported separately).

| Cell | Coverage | Sustainable FPS | Radar prepare med/p95 ms | Pre-front med/p95 ms | <=100 ms | <=500 ms | Median AoI ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| a20__favorable_stable | 0.9808327825512227 | 9.806130521306 | 22.632592357695103 / 44.380489736795425 | 32.34053496271372 / 60.24715583771467 | 0 | 183 | 448.51112365722656 |
| a71__favorable_stable | 0.9812520924004018 | 9.817353139929764 | 22.387824952602386 / 45.24881672114134 | 32.730042934417725 / 62.71798722445965 | 0 | 1702 | 305.94921112060547 |

## Readiness checks

- `both_cells_pass_structural_and_teardown_checks`: PASS
- `preparation_timing_is_complete`: PASS
- `avoidable_pre_front_compute_keeps_up_with_100ms_arrival_period`: PASS
- `bounded_latest_frame_queues_remain_valid`: PASS
- `service_and_feedback_boundaries_are_distinct`: PASS

The 0.95 preparation target and 100 ms service target remain measured performance outcomes. They were not weakened or replaced by the 500 ms feedback timeout.
