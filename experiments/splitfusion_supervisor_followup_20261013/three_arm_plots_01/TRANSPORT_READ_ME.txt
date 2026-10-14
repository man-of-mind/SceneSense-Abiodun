TRANSPORT ADDITION — existing traces, CPU-only

11: offered application payload, PDCP ingress and RLC dequeue rates and one-second P50/P99.
12: actual UE RLC occupancy and separately labelled gNB estimated uplink backlog.
Direct BSR events were not enabled in the retained UE recording; BSR report values/rates are unavailable.
Do not call the gNB estimate a measured BSR. A rising/falling queue is not a throughput measurement.
T-tracer binary IDs and integer layouts are decoded from the embedded configuration of each raw file.
UE event timestamps use recorded CLOCK_REALTIME; gNB timestamps are collector ingest.
Only complete one-second bins are used for rate quantiles. No binning/interpolation of missing telemetry to zero.
RLC tick grouping closes only when the next (RNTI, UE, frame, slot) group begins, summing latest distinct LCIDs.
Application rate uses SFD4 bytes at capture time; it measures offered volume, not a packet send timestamp.
Reproduce after prepare_plots.py:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 nice -n 10 python3 experiments/splitfusion_supervisor_followup_20261013/transport_plots.py
