# Run-4B pre-training latency-boundary gate: STOP

Verdict: `STOP_MODELED_LATENCY_INCLUDES_GT_EVALUATOR_DELAY`. Training was not
started. Reproduce with `env -u PYTHONPATH python3 latency_boundary_audit.py`
(exit 2 = stop); the full numbers and source hashes are in
`LATENCY_BOUNDARY_AUDIT.json`.

## Required property

Run-4B `L_ms` = action-open -> UE receipt of the immediate `TAIL_OUTPUT_READY`
operational ACK, excluding GT evaluation and map installation. Training may
reuse the existing modeled latency only if it already represents that path.

## What the existing model composes

`ue_production_transport_model_v2/collector_v1.py` builds every successful
latency as

```text
retained_residual + send_span + modeled_transport + actor_reserve
retained_residual = prepare_start->ue_rx - (first_send->ue_rx - eval_enqueued->ue_rx)
                  = [prepare_start -> first_send] + [eval_enqueued -> ue_rx]
```

The residual is drawn from the 555 retained action-50 probe rows
(`splitfusion_quality_feedback_probe_v1/20260916_action50_favorable_adverse_retry4`).
Its second term is the whole post-GT quality-ACK path.

## Evidence (pooled, 555 rows, ms)

| Stage inside the residual | p50 | p95 | p99 | max |
|---|---|---|---|---|
| GT wait (enqueued -> started) | 0.08 | 257.93 | 459.48 | 569.51 |
| GT scoring (started -> completed) | 2.51 | 117.47 | 185.74 | 250.96 |
| GT wait + scoring | 2.61 | 336.27 | 548.71 | 690.15 |
| Residual as used by the collector | 33.05 | 367.73 | 570.91 | 714.77 |
| Residual minus GT wait and scoring | 24.97 | 56.24 | 78.08 | 249.30 |

- GT stages are 61.6% of the summed residual.
- 66/555 residual draws alone exceed 170 ms, and therefore force a modeled
  timeout whatever action is chosen. Without the GT stages, 1/555 would.
- The downlink stage is the post-GT quality ACK. No `TAIL_OUTPUT_READY` ACK
  exists anywhere in the repository source or retained evidence.

## Secondary observation (unquantified)

The residual removes `first_send -> eval_enqueued`, but its replacement
(`send_span + transport`) ends at complete edge reassembly
(`ue_production_queue_capture_v1/contract.py`). Edge pre-model, model tail
and prediction-ready -> enqueue therefore appear in no composed term. The
probe's edge reassembly timestamps are unpopulated, so this gap is not
measured here.

## Not done

No alternative latency model was built. Two candidate repairs exist:
- drop the GT stages from the residual;
- add an edge-compute term.

Each is a new model, so it needs an explicit Abiodun decision. The 20-feature
state, operational prior, tests and training were not implemented.
