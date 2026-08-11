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
- `configs/policy_corpus_v1.yaml` is the single pre-registered experiment
  definition: scenario arguments, 24 seeds, splits, provenance, and gates.

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
```

A `FAIL_QUARANTINED` batch must not be added to the surrogate replay roots.
Keep it immutable and collect a documented sibling replacement batch.
