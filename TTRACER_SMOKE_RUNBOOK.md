# OAI T-Tracer Smoke Test Runbook

This is the first SceneSense path for OAI radio-side telemetry. It does not
replace the application CSV or the UE tunnel sampler. It adds a small T-tracer
recording that can expose RAN events such as MCS, TBS, RB allocation, symbol
allocation, HARQ state, and selected uplink payload scheduling fields.

## What This Smoke Test Proves

- The gNB and multi-UE softmodems can run with T-tracer enabled.
- We can record short raw traces from the gNB and UE tracer ports.
- We can replay those raw files and extract selected CSVs for later analysis.

This is intentionally not the final radio-metrics parser. BLER, HARQ, and
clean per-UE joins still need follow-up once the smoke path is stable.
The default UE profile is clean and NR-specific: it records the local
`NRUE_MAC_DCI_GRANT` event. Treat legacy UE-side `UE_PHY_MEAS` as diagnostic
for now in rfsim; gNB-side MAC/PDCP traces and `NRUE_MAC_DCI_GRANT` are the
more reliable sources for reported radio/network metrics.

## 1. Build T-Tracer Tools

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/scripts
./ttracer_build_tools.sh
```

The helper builds only the tools we need first: `record`, `replay`, `csv`,
`extract_config`, and `textlog`.

## 2. Start OAI With T-Tracer Enabled

Start the core normally:

```bash
./cn_start.sh
```

Start the gNB with the T-enabled launcher:

```bash
./gnb_start_ttracer.sh
```

Start the two-UE softmodem with the T-enabled launcher:

```bash
./ue_multi_start_ttracer.sh
```

Check the tunnels:

```bash
./ue_multi_check.sh
```

Default tracer ports:

```text
gNB softmodem -> 127.0.0.1:2021
UE softmodem  -> 127.0.0.1:2023
```

## 3. Start The Fusion Run

Use the normal multi-UE fusion runbook, but use a manual `run_group` so the app,
tunnel, and T-tracer files are easy to join later.

Example label:

```bash
RUN_GROUP=exp02_ttracer_smoke
```

Use that same value in:

- Both front-half commands: `--run-group exp02_ttracer_smoke`
- The tunnel sampler: `--run-group exp02_ttracer_smoke`
- The T-tracer recorder commands below

## 4. Record Short T-Tracer Raw Files

Run these while the fusion traffic is active. Use separate terminals so gNB and
UE traces are recorded during the same time window.

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/scripts

./ttracer_record_smoke.sh \
  --run-group exp02_ttracer_smoke \
  --source gnb \
  --duration-s 60
```

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/scripts

./ttracer_record_smoke.sh \
  --run-group exp02_ttracer_smoke \
  --source ue \
  --duration-s 60
```

Outputs:

```text
metrics_logs/scenesense_ttracer/exp02_ttracer_smoke/gnb/gnb.raw
metrics_logs/scenesense_ttracer/exp02_ttracer_smoke/ue/ue.raw
```

## 5. Extract Smoke CSVs

After recording, extract the default CSV panel:

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/scripts

./ttracer_extract_csv_smoke.sh \
  --run-group exp02_ttracer_smoke \
  --source gnb

./ttracer_extract_csv_smoke.sh \
  --run-group exp02_ttracer_smoke \
  --source ue
```

If a previous extraction left legacy UE CSVs in the same folder, add
`--clean-output` to remove old CSVs before writing the selected profile.

Default gNB CSVs:

- `GNB_MAC_UL.csv`: uplink scheduler MCS/TBS by RNTI.
- `GNB_MAC_DL.csv`: downlink scheduler MCS/TBS by RNTI.
- `GNB_MAC_LCID_UL.csv`: uplink logical-channel data by RNTI/LCID.
- `GNB_MAC_LCID_DL.csv`: downlink logical-channel data and queue occupancy by
  RNTI/LCID.
- `GNB_MAC_PUSCH_POWER_CONTROL.csv`: `snrx10`, `rbSize`, `mcs`, `tb_size`.
- `GNB_MAC_PUCCH_POWER_CONTROL.csv`: PUCCH SNR-like power-control trace.
- `ENB_RLC_UL.csv`, `ENB_RLC_DL.csv`: RLC SDU lengths by UE/radio bearer.
- `ENB_RLC_MAC_UL.csv`, `ENB_RLC_MAC_DL.csv`: RLC-to-MAC lengths by UE/radio
  bearer.
- `ENB_PDCP_UL.csv`: uplink PDCP SDU lengths by UE/radio bearer.
- `ENB_PDCP_DL.csv`: downlink PDCP SDU lengths by UE/radio bearer.
- `GNB_PHY_UL_PAYLOAD_RX_BITS.csv`: UL `rb_size`, `mcs_index`,
  `number_of_bits`.

Default UE CSVs:

- `NRUE_MAC_DCI_GRANT.csv`: NR UE decoded UL/DL grants for local network-state
  estimation. This is the preferred UE-side RL input because it exposes MCS,
  RB allocation, symbols, TBS, HARQ, NDI, and RV after the DCI has been decoded
  and validated by NR UE MAC.

Optional UE CSVs:

- `UE_PHY_UL_PAYLOAD_TX_BITS.csv`: UL `rb_size`, `mcs_index`,
  `number_of_bits`. Use `--profile payload` when you want to validate
  `NRUE_MAC_DCI_GRANT.tbs * 8` against the existing OAI payload trace.
- `UE_PHY_MEAS.csv`: RSRP, RSSI, SNR, wideband CQI.
- `UE_PHY_ULSCH_UE_DCI.csv`: UL grant MCS, RB range, TBS, HARQ round.
- `UE_PHY_DLSCH_UE_DCI.csv`: DL grant MCS and TBS.

Use `--profile legacy` only when you intentionally want the older UE PHY
measurement/DCI files for debugging. In current rfsim runs they can be empty or
carry sentinel-like measurement values, so they are not part of the clean
SceneSense UE metric panel.

To convert the clean UE grant CSV into per-RNTI/window features:

```bash
python3 scripts/analyze_nrue_grant_metrics.py \
  --run-group exp02_ttracer_smoke \
  --window-s 1.0
```

This writes:

```text
metrics_logs/scenesense_ttracer/<run_group>/ue/analysis/nrue_grant_windows.csv
metrics_logs/scenesense_ttracer/<run_group>/ue/analysis/nrue_grant_summary.csv
metrics_logs/scenesense_ttracer/<run_group>/ue/analysis/nrue_grant_summary.md
```

For a complete logging-validation run where application metrics, tunnel
metrics, UE grant traces, and gNB traces share one `run_group`, run the
post-processing bundle:

```bash
scripts/run_logging_validation_analysis.sh \
  --run-group exp02_ttracer_smoke \
  --window-s 1.0
```

This writes a small manifest under:

```text
metrics_logs/scenesense_ttracer/<run_group>/analysis/logging_validation_manifest.txt
```

## Notes

The multi-UE softmodem currently exposes one UE-side tracer port for both UEs.
The gNB-side events include `rnti`, which should let us separate UEs during the
next parser step. Treat this smoke test as proof that the telemetry source is
reachable and parseable, not as final per-UE radio analysis yet.

In OAI's current trace code, `GNB_MAC_LCID_UL.data_size` is emitted as bits
while `GNB_MAC_LCID_DL.data_size` is emitted as bytes. Keep that unit difference
in mind when comparing uplink and downlink logical-channel totals.

The current OAI tree exposes structured RLC and PDCP events. SDAP appears as
legacy log categories rather than a clean per-packet T-tracer event in this
tree, so keep SDAP/QFI/5QI mapping as a later instrumentation task if the
policy needs bearer-level QoS labels.

`NRUE_MAC_DCI_GRANT` is a local SceneSense/OAI instrumentation event. After
editing `common/utils/T/T_messages.txt`, rebuild the UE softmodem so the
generated `T_IDs.h` includes the new event before expecting this CSV to populate.
