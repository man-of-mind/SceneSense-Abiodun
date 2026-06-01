# SceneSense Logging Plan

This file defines what we log before adding network-aware control.

## Run Directory

Each stream writes a metrics folder under:

```text
metrics_logs/scenesense_runs/<run_id>/
```

Streams from the same experiment may live in separate folders. Pair them during
plotting by `run_group` and `stream_id`. By default, `run_group` is an automatic
10-minute timestamp bucket plus the transport label, for example
`20260527_1950_multi_ue_oai`. If an exact label is needed, pass the same
`--run-group <label>` to each stream.

Use the automatic bucket for quick smoke tests. For official experiment runs or
anything that will feed plots/tables, always pass an explicit label such as
`--run-group exp01_oai_clear_multiue`; otherwise two experiments started inside
the same 10-minute window can share a group.

Recommended structure:

```text
run_manifest.json                  # optional shared run note
streams/
  <stream_id>_metrics.csv           # one CSV per pole/car stream
manifests/
  <stream_id>_manifest.json
  <stream_id>_resolved_config.json
network_snapshots/
oai_logs/
  core_container_logs/
t_tracer/
analysis/
```

The older `SCENESENSE_RUN_ID`/`--metrics-run-dir` path is still available when
you want multiple streams to write into one folder, but it is optional.

## Application Metrics

These are the source-of-truth metrics because they align directly to perception
frame IDs.

Per stream/frame:

- `frame_id`
- `run_group`
- `stream_id`
- `transport_label`
- `front_ms`
- `back_ms`
- `round_trip_ms`
- `feature_payload_bytes`
- `feature_payload_bytes_uncompressed`
- `feature_payload_chunks`
- `result_payload_bytes_estimate`
- `result_payload_chunks_estimate`
- `result_received`
- `mask_present`
- `segmentation_class_count`
- `object_count`
- `radar_projected_points`
- `spatial_map_dropped_packets`
- UE/front bind IP, back-half IP, and UDP ports

Use `scripts/analyze_scenesense_app_metrics.py` for the first-pass application
analysis. It scans `metrics_logs/scenesense_runs/`, groups streams by
`run_group`, and writes:

- `application_summary.csv`
- `application_combined_rows.csv`
- `application_summary.md`
- `application_timeseries.png`
- `application_summary_bars.png`

Example:

```bash
python3 scripts/analyze_scenesense_app_metrics.py --run-group exp01_oai_clear_multiue
```

If the application streams and network sampler accidentally used different
groups, keep the raw files unchanged and pass both labels:

```bash
python3 scripts/analyze_scenesense_app_metrics.py \
  --run-group 20260527_2120_oai \
  --network-run-group exp01_oai_clear_multiue
```

When matching network metrics exist, the same helper also writes:

- `network_summary.csv`
- `network_combined_rows.csv`
- `network_timeseries.png`

## Network Metrics

Collect these at the same run boundary. Some are available immediately from
Linux/OAI logs; some need T-tracer parsing.

Use `scripts/sample_oai_network_metrics.py` during OAI runs for the lightweight
time-series path:

```bash
python3 scripts/sample_oai_network_metrics.py \
  --run-group exp01_oai_clear_multiue \
  --ping-host 192.168.70.135
```

This writes under `metrics_logs/scenesense_network/<run_group>/`:

- `network_timeseries.csv`
- `network_summary.csv`
- `network_manifest.json`

For UE tunnel interfaces, `tx_*` is approximately UE uplink traffic and `rx_*`
is return/downlink traffic. This is not a replacement for RAN-layer telemetry,
but it gives us a stable transport time series before the T-tracer path is
fully parsed.

Immediate Linux/container metrics:

- UE tunnel TX/RX bytes and packet counts from `ip -s link`.
- UE tunnel drops/errors.
- UE-to-ext-DN ping latency/loss.
- Docker/core container logs.
- Back-half container logs.
- GPU state with `nvidia-smi`.

RAN/OAI metrics to extract from gNB/UE stdout or T-tracer:

- UL/DL bitrate or throughput.
- SNR/SINR.
- CQI.
- MCS index.
- Number of PRBs allocated.
- Transport block size.
- BLER.
- HARQ retransmissions and NACK rate.
- RLC/PDCP throughput and retransmission indicators, where exposed.
- Scheduling delay or slot timing indicators, where exposed.
- Buffer occupancy/backlog, where exposed.

Useful derived metrics:

- Application goodput: useful feature/result bytes per second.
- Network overhead estimate: tunnel bytes minus application payload bytes.
- Timeout rate: missed perception results / sent frames.
- Jitter: variation in per-frame round-trip time.
- Saturation indicator: rising RTT plus missed results plus stable/large payload.

## T-Tracer Positioning

OAI T-tracer is a deeper RAN debugging source. The local OAI docs describe it as
an event collector inside the softmodem plus tools for recording/displaying
events. Use it after the application CSV is stable.

Use the smoke-test runbook first:

```bash
TTRACER_SMOKE_RUNBOOK.md
```

The helper launchers add the typical softmodem options:

```bash
--T_stdout 2 --T_nowait --T_port <port>
```

Default ports:

```text
gNB softmodem -> 127.0.0.1:2021
UE softmodem  -> 127.0.0.1:2023
```

Smoke recorder:

```bash
scripts/ttracer_record_smoke.sh --run-group exp02_ttracer_smoke --source gnb --duration-s 60
scripts/ttracer_record_smoke.sh --run-group exp02_ttracer_smoke --source ue --duration-s 60
```

Smoke extractor:

```bash
scripts/ttracer_extract_csv_smoke.sh --run-group exp02_ttracer_smoke --source gnb
scripts/ttracer_extract_csv_smoke.sh --run-group exp02_ttracer_smoke --source ue
```

For the gNB stdout MAC summary blocks, use:

```bash
python3 scripts/parse_oai_gnb_mac_stats.py \
  --input <run_dir>/oai_logs/gnb_stdout.log \
  --output-dir <run_dir>/oai_logs/gnb_mac_parsed
```

Store raw tracer files and first CSVs under:

```text
metrics_logs/scenesense_ttracer/<run_group>/
```

The first event panel is intentionally small:

- gNB: `GNB_MAC_UL`, `GNB_MAC_DL`, `GNB_MAC_PUSCH_POWER_CONTROL`,
  `GNB_MAC_PUCCH_POWER_CONTROL`, `GNB_MAC_LCID_UL`, `GNB_MAC_LCID_DL`,
  `ENB_RLC_UL`, `ENB_RLC_DL`, `ENB_RLC_MAC_UL`, `ENB_RLC_MAC_DL`,
  `ENB_PDCP_UL`, `ENB_PDCP_DL`, `GNB_PHY_UL_PAYLOAD_RX_BITS`.
- UE clean profile: `NRUE_MAC_DCI_GRANT`.
- UE payload validation profile: `NRUE_MAC_DCI_GRANT`,
  `UE_PHY_UL_PAYLOAD_TX_BITS`.
- UE legacy/debug profile: `NRUE_MAC_DCI_GRANT`, `UE_PHY_MEAS`,
  `UE_PHY_ULSCH_UE_DCI`, `UE_PHY_DLSCH_UE_DCI`,
  `UE_PHY_UL_PAYLOAD_TX_BITS`.

These expose useful first-pass fields such as MCS, TBS, RB allocation size,
logical-channel bytes, RLC/PDCP SDU lengths, and UL payload bit counts. UE-side
`UE_PHY_MEAS` is useful for debugging but should not be part of the default
clean metric panel in rfsim until its fields are cross-checked against gNB MAC
stdout. Later analysis should join validated metrics with application metrics
by run group and time window, then produce `analysis/network_radio_summary.csv`.

For a UE-side RL agent, prefer `NRUE_MAC_DCI_GRANT` over UE PHY SNR/CQI fields
in the current rfsim setup. It exposes what the UE actually receives as usable
UL/DL scheduling grants: MCS, RB allocation, symbol allocation, TBS, HARQ
process, NDI, RV, and TPC. gNB metrics remain useful as a validation/oracle
source and for server-side orchestration.

Use `scripts/analyze_nrue_grant_metrics.py` to derive RL-ready per-RNTI/window
features from `NRUE_MAC_DCI_GRANT.csv`: scheduled UL/DL Mbps, grant rate, MCS,
RB allocation, symbol allocation, average TBS, modulation order, code rate, and
retransmission indicators.

Validation note from `exp07_full_logging_validation`: in this OAI tree the UE
UL grant `tb_size` is emitted in bytes, while the UE DL grant `tb_size` is
emitted in bits. The analysis helpers normalize both directions to bytes before
computing scheduled Mbps or comparing with gNB MAC TBS. UE UL `tbs * 8`
validated exactly against `UE_PHY_UL_PAYLOAD_TX_BITS.number_of_bits`.

The current OAI tree exposes SDAP mostly through legacy log categories rather
than clean per-packet T-tracer fields. Treat SDAP/QFI/5QI mapping as a later
instrumentation item if bearer-level QoS labels become necessary.

## Principle

Application CSV answers: "Did perception meet its latency/payload/task target?"

OAI logs answer: "What did the network and scheduler do during that interval?"

The paper needs both, but the frame-keyed application CSV should remain the
primary experiment clock.
