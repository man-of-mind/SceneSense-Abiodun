THREE-ARM PRESENTATION PLOTS — PNG ONLY

Compare Agent, mode03 AE128/UINT8/q0 and mode09 AE32/UINT8/q0 from the same 13-arm OAI campaign.
These are separate 1,000-frame sessions, not full-route or matched-scene trials. No new experiment was run.
Suggested core plots: 01 completion curves; 03 payload; 06 deadline-aware perception; 08 map freshness.
Context/detail: 02 quantiles and missing outcomes; 04 channel; 05 latency traces; 07 scene and q; 09 observed quality traces; 10 map quantiles; 13 payload timelines.
Plot 03 is bars only; plot 13 shows absolute payload time series on one shared linear kB scale. Original filenames retained.
Plot 07 is agent-only: five aligned panels for SI, P40, q, decision-matched application payload and measured PUSCH SNR.
All PNGs are 3840 x 2160. No PDF or slide deck is produced. Full captions are in CAPTIONS.json.

Definitions:
- Completion/feedback time: CLOCK_MONOTONIC_RAW from action-open, independently for each milestone.
- Quality deadline: prediction ready <=170 ms, following the sealed static campaign, not UE receipt.
- Timely precision conditions on timely predictions; unlike recall it cannot express missing-detection coverage.
- Scene gaps and undefined quality are never filled with zero or interpolated.
- PUSCH radio time: host telemetry ingest; no frame-to-radio causal mapping is asserted.
- Map AoI CDF: exact time weighting of linear age intervals after the first successful install,
  stopping at the last sent capture. Initial no-map seconds are separate. Successful install ages include drain results.
- P50/P99: NumPy linear percentiles; latency quantiles condition on recorded events, payload uses all sent frames.
- Bars with <= / hatching are upper bounds caused by missing GT, not confidence intervals.

Reproduce from workspace root:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 nice -n 10 python3 experiments/splitfusion_supervisor_followup_20261013/prepare_plots.py
Only this output directory is written. Historical figures, runtime, UI and M4 evidence are untouched.
