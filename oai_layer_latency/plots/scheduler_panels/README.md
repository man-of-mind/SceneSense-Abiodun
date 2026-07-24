# Scheduler panel plots

These are the corrected scheduler-panel plots generated after rerunning the
default 106PRB CARLA condition with the same no-AE ~1 MB payload used by the
UL-heavy 106PRB and 273PRB comparison rows.

The earlier mixed-payload plots were archived under:

`abiodun/oai_layer_latency/plots/scheduler_panels_invalid_mixed_payload_20260723/`

`abiodun/oai_layer_latency/plot_scheduler_timeseries_panels.py` now refuses to
generate plots unless all CARLA comparison rows have `feature_kb_p50 >= 900 KB`.
