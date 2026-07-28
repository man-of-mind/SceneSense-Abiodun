# Bursty traffic experiments

This folder contains real bursty traffic traces and replay helpers for comparing
CARLA split-inference traffic against other application traffic patterns.

Downloaded external data:

- `TRACTOR/` — sparse checkout of `https://github.com/genesys-neu/TRACTOR`
- The external checkout is ignored by git via `../.gitignore`.

Local files to keep:

- `BURSTY_TRAFFIC_EXPERIMENT_PLAN.md`
- `analyze_tractor_raw.py`
- `udp_trace_replay.py`
- `udp_sink.py`
- `analysis/tractor_trace_summary.csv`

Quick offline summary:

```bash
cd abiodun/bursty_traffic
python3 analyze_tractor_raw.py --raw-glob 'TRACTOR/raw/*.csv' --out analysis/tractor_trace_summary.csv
```

Quick dry-run replay check:

```bash
python3 udp_trace_replay.py TRACTOR/raw/urllc_03_03.csv --dst 127.0.0.1 --direction uplink --max-duration-s 5 --dry-run
```
