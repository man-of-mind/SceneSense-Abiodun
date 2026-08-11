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
```

Never rewrite a `FAIL_QUARANTINED` verification. A later, explicitly versioned
analysis may supersede only its use/disposition under a corrected goal; keep the
batch and both reports immutable. Do not add the batch to replay roots until the
documented human salvage/top-up decision is made.
