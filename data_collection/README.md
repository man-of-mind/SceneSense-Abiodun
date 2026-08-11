# Policy corpus collection

This directory owns the L10319 collection loop described in
`rl_agent/policy/DATA_COLLECTION_PLAN.md`.

- `carla_fusion_policy_corpus_collector.py` delegates to the validated shared
  fusion collector and adds pedestrian ground-truth rows. It does not modify or
  fork the real-time perception implementation.
- `run_policy_corpus.py` performs static/live preflight, runs the registered
  smoke trials or the locked 24-run batch, and stops at the first basic
  pipeline/timing or actor-cleanup failure.
- `verify_policy_corpus.py` applies the full §5 gates and writes a timestamped
  `CORPUS_VERIFICATION.md`, detailed CSVs, an explicit replay split manifest,
  and a hashed verification manifest.
- `rescore_policy_corpus_freshness.py` performs the phase-1, controller-independent
  freshness re-score over an existing immutable batch. It emits GT-seeded and
  detection-seeded views, speed/dwell distributions, right-censored breach times,
  liveness bands, detection coverage, and per-run/split concentration tables.
- `reconcile_detection_coverage.py` applies the identical live direct-coverage
  score to old and new traces, reports range/denominator sensitivity, and audits
  detector configuration plus timeout/empty-result accounting. It is table-only.
- `configs/detection_ab_gate_v1.yaml` defines the six-run matched three-arm gate;
  `analyze_detection_ab_gate.py` computes the hard vehicle/pedestrian gates,
  paired moving-block confidence intervals, fast-in-view dwell, and decoder
  top-80 saturation diagnostics.
- `configs/policy_corpus_v1.yaml` is the single pre-registered experiment
  definition: scenario arguments, 24 seeds, splits, provenance, and gates.
- `configs/freshness_rescore_v1.yaml` locks the table-driven re-score constants
  (epsilon, range, clock, reference localization profile, QC, and regime bands).

Use the CARLA 0.10.0 virtual environment; do not export `PYTHONPATH` and do not
start OAI. Start the packaged server with `Town10HD_Opt` already loaded, then:

```bash
/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  data_collection/run_policy_corpus.py --mode smoke
```

Inspect both smoke records in `batch_manifest.json`. Lock the healthy device
placement in the YAML before the full batch; do not simply keep the faster
placement if it violates `camera_frame_wait`.

```bash
/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  data_collection/run_policy_corpus.py --mode full

/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  data_collection/verify_policy_corpus.py \
  data_collection/experiments/policy_corpus_v1/<batch_timestamp>_full

/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  data_collection/rescore_policy_corpus_freshness.py \
  data_collection/experiments/policy_corpus_v1/<batch_timestamp>_full

/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  data_collection/reconcile_detection_coverage.py
```

Never rewrite a `FAIL_QUARANTINED` verification. A later, explicitly versioned
analysis may supersede only its use/disposition under a corrected goal; keep the
batch and both reports immutable. Do not add the batch to replay roots until the
documented human salvage/top-up decision is made.

The current v1 corpus is additionally held by
`rl_agent/policy/DETECTION_RECONCILIATION.md`: do not make the pedestrian-scope
call, collect a fast-car supplement, or start the controller ladder until the
detector recipe is reconciled.

The current authorized reconciliation command is:

```bash
/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  data_collection/run_policy_corpus.py \
  --config data_collection/configs/detection_ab_gate_v1.yaml --mode smoke

/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  -m data_collection.analyze_detection_ab_gate \
  data_collection/experiments/detection_ab_gate_v1/<batch_timestamp>_smoke
```

The analyzer exits non-zero at any failed gate. Do not start a corrected full
collection unless its status is `PASS_GATE_1_2`.
