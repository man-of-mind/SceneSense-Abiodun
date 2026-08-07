# OAI uplink latency bottleneck handoff

This note explains how we diagnosed the high uplink latency seen when sending
large/bursty split-inference payloads over OAI RFsim, what OAI files mattered,
and how to reproduce the same investigation on another experiment.

The short version:

> Our bottleneck was not edge inference or downlink. It was UE-side RLC queueing
> caused by low uplink MCS under bursty application traffic. On a clean RFsim
> channel, OAI's default BLER/OLLA path reduced or held UL MCS too low because
> the split-inference traffic produced sparse scheduling windows. Switching the
> UL new-data MCS selection to OAI's SINR lookup path fixed the clean-channel
> MCS behavior while keeping HARQ/retransmission enabled.

Use this guide if another OAI experiment shows high uplink latency, growing BSR
or RLC backlog, or unexpectedly low MCS despite a good RFsim channel.

## 1. Symptom we observed

The workload was CARLA split inference: about a 1 MB feature tensor per frame,
sent in bursts over the OAI uplink.

The vanilla OAI behavior on default 106 PRB / RFsim clean channel looked like:

| Metric | Vanilla OAI symptom |
|---|---:|
| gNB PUSCH SNR | about 50 dB |
| BLER / retransmissions | near zero in clean channel |
| UL MCS | low, around 7 in the final clean 106PRB baseline |
| Feature uplink handling p50 | about 142 ms |
| UE RLC / BSR behavior | backlog forms under bursty 1 MB payloads |
| Downlink result latency | small, only a few ms |

The key contradiction was:

> Same clean RFsim channel, but smooth iperf could receive high MCS while the
> bursty split-inference flow was assigned low MCS. This pointed away from RF
> channel quality and toward OAI's UL link-adaptation cadence.

Fixed-MCS and SINR-policy diagnostics showed that when UL MCS stays high, the
RLC queue drains quickly and uplink latency collapses toward tens of ms.

## 2. Mental model: what is actually happening

For a bursty application, the app offers traffic like:

```text
offered load = payload_per_frame × frame_rate
```

The OAI uplink can only drain at the scheduled service rate:

```text
served rate ≈ grants/sec × TBS/grant
```

If:

```text
offered load > served rate
```

then the excess bytes sit in the UE RLC queue. That produces:

- rising UE RLC occupancy;
- rising BSR / LCG backlog;
- higher capture-to-edge latency;
- delivery cliff if the queue saturates or the app times out.

In our case, the served rate was low because the UL MCS was low. The low MCS was
not caused by low SNR in clean RFsim; it was caused by the default BLER/OLLA
update behavior under sparse bursty traffic.

## 3. Diagnostic checklist

Run these checks before changing OAI logic.

### A. Confirm the latency is uplink, not edge/downlink

Break the application latency into at least:

- front/sensor/model-front processing;
- uplink transport / feature handling;
- edge tail inference;
- downlink/result return, if any;
- capture-to-result or capture-to-map total.

If uplink dominates while downlink is small, continue.

### B. Compare against iperf

Run both:

1. smooth uplink traffic, e.g. iperf UDP/TCP;
2. the actual application traffic.

The important comparison is not only throughput. Compare:

- MCS over time;
- PRB allocation;
- TBS/grant;
- grant rate;
- scheduled UL Mbps;
- UE RLC buffer;
- UE BSR / LCG backlog;
- BLER / retransmission rate;
- PUSCH SNR.

If iperf gets high MCS but the app gets low MCS on the same RFsim channel, the
traffic cadence is likely interacting badly with the UL link-adaptation loop.

### C. Look for this bottleneck signature

The same mechanism is likely present if you see:

| Evidence | Interpretation |
|---|---|
| high PUSCH SNR | channel is not the limiting factor |
| low BLER / low retx | MCS is not being lowered because of real errors |
| low UL MCS | low spectral efficiency caps drain rate |
| high RLC occupancy | bytes are waiting at UE RLC |
| high BSR / LCG backlog | UE is asking for uplink resources but queue is not draining fast enough |
| payload-sized RLC spikes | whole application bursts are stuck in the queue |
| fixed MCS28 improves latency | MCS/link adaptation is the active lever |

## 4. OAI files that mattered

All paths below are relative to the OAI tree:

```text
openairinterface5g/
```

### UL scheduler and MCS selection

```text
openair2/LAYER2/NR_MAC_gNB/gNB_scheduler_ulsch.c
```

Important roles:

- main UL scheduling path;
- chooses `selected_mcs` for new UL data;
- calls either BLER/OLLA-based MCS or SINR-based MCS;
- contains the env-gated `SCENESENSE_MCS_POLICY=sinr` branch in our tree;
- contains fixed-MCS diagnostic override `SCENESENSE_FORCE_UL_MCS`;
- emits `T_GNB_MAC_UL_MCS_DECISION`.

Important code idea:

```c
if (bo->harq_round_max == 1 || scenesense_use_sinr_mcs_policy()) {
  selected_mcs = get_mcs_from_SINRx10(
      current_BWP->mcs_table,
      sched_ctrl->pusch_pc.avg_snr * 10,
      nrOfLayers);
  ...
} else {
  selected_mcs = get_mcs_from_bler(...);
}
```

The key point is that the `SCENESENSE_MCS_POLICY=sinr` branch uses the SINR
lookup for new-data MCS while **not disabling HARQ**. Do not simply set
`harq_round_max = 1` as a shortcut, because that changes retransmission behavior
and ruins reliability measurements.

### BLER/OLLA MCS update and SINR table

```text
openair2/LAYER2/NR_MAC_gNB/gNB_scheduler_primitives.c
```

Important roles:

- `get_mcs_from_SINRx10(...)`;
- `get_mcs_from_bler(...)`;
- `BLER_UPDATE_FRAME`;
- `BLER_FILTER`;
- default sparse-window behavior;
- optional hold-few and AIMD diagnostic policy;
- emits `T_GNB_MAC_BLER_MCS_DECISION`.

Important code idea from the vanilla BLER path:

```c
if (num_dl_sched <= 3) {
  branch = 3;
  if (!hold_mcs_few_samples)
    new_mcs -= 1;
}
```

This was the suspicious behavior for bursty traffic: sparse windows could lead
to conservative MCS reduction even when the channel was clean and BLER was zero.

### Function prototypes

```text
openair2/LAYER2/NR_MAC_gNB/mac_proto.h
```

Important symbols:

- `get_mcs_from_SINRx10(...)`;
- `get_mcs_from_bler(...)`.

### T-tracer event database

```text
common/utils/T/T_messages.txt
```

Important events:

- `GNB_MAC_UL`;
- `GNB_MAC_UL_MCS_DECISION`;
- `GNB_MAC_BLER_MCS_DECISION`;
- `GNB_MAC_PUSCH_POWER_CONTROL`;
- `NRUE_MAC_DCI_GRANT`;
- `NRUE_MAC_RLC_BUFFER_STATUS`;
- `NRUE_MAC_BSR_STATUS`;
- `NR_RLC_TX_SDU`;
- `NR_RLC_TX_DEQUEUE`;
- `GNB_MAC_RX_SDU`;
- `GNB_PDCP_RX_DELIVER`.

Important warning:

> If you edit `T_messages.txt`, rebuild both `nr-softmodem` and
> `nr-uesoftmodem`. T-tracer verifies the runtime binary against the T database
> and can abort if the database and binary are mismatched.

### UE queue / BSR instrumentation

```text
openair2/LAYER2/NR_MAC_UE/nr_ue_scheduler.c
```

Important events in our tree:

- `NRUE_MAC_RLC_BUFFER_STATUS`;
- `NRUE_MAC_BSR_STATUS`.

These are essential for proving whether bytes are actually stuck at UE RLC.

## 5. Runtime knobs we added/used

These are read by the patched OAI gNB process at startup. Set them before
launching `nr-softmodem`.

| Env var | Meaning | Use |
|---|---|---|
| `SCENESENSE_MCS_POLICY=sinr` | Use OAI SINR-to-MCS helper from tracked PUSCH SNR | Main candidate fix for controlled RFsim/channel sweeps |
| `SCENESENSE_FORCE_UL_MCS=28` | Force UL MCS to a fixed value | Diagnostic upper bound only |
| `SCENESENSE_HOLD_MCS_FEW_SAMPLES=1` | Do not decrement MCS when BLER window has too few samples | Diagnostic for sparse-window artifact |
| `SCENESENSE_MCS_POLICY=aimd` | TCP-Reno-like BLER-aware policy | Secondary diagnostic |
| `SCENESENSE_AIMD_MAX_DROP=3` | Cap AIMD MCS decrease per bad window | Safer AIMD variant |

For a plain OAI launch:

```bash
sudo env SCENESENSE_MCS_POLICY=sinr ./nr-softmodem \
  -O ../../../targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.sa.band78.fr1.106PRB.usrpb210.conf \
  --gNBs.[0].min_rxtxtime 6 \
  --rfsim \
  --T_stdout 2 \
  --T_nowait \
  --T_port 2021
```

For our wrapper scripts, the short form was:

```bash
MCS_POLICY=sinr bash abiodun/downlink_latency_fps/run_oai_default106_ttracer_10fps.sh
```

Vanilla baseline:

```bash
unset SCENESENSE_MCS_POLICY
unset SCENESENSE_FORCE_UL_MCS
unset SCENESENSE_HOLD_MCS_FEW_SAMPLES
unset SCENESENSE_AIMD_MAX_DROP
```

If the colleague is using a fresh upstream OAI tree, these env vars will do
nothing until the SceneSense hooks are ported into the source files listed in
Section 4. The minimum useful port is:

1. add `SCENESENSE_MCS_POLICY=sinr` parsing in `gNB_scheduler_ulsch.c`;
2. branch the UL new-data `selected_mcs` path to
   `get_mcs_from_SINRx10(current_BWP->mcs_table, sched_ctrl->pusch_pc.avg_snr * 10, nrOfLayers)`;
3. keep HARQ enabled;
4. add or preserve T-tracer events that expose selected/final MCS, SNR, TBS,
   RBs, and queue state.

## 6. Rebuild after source edits

After editing any of these:

- `gNB_scheduler_ulsch.c`;
- `gNB_scheduler_primitives.c`;
- `T_messages.txt`;
- UE queue instrumentation files;

rebuild:

```bash
cd openairinterface5g/cmake_targets/ran_build/build
ninja nr-softmodem nr-uesoftmodem
```

If `ccache` causes permission issues:

```bash
env CCACHE_DISABLE=1 ninja nr-softmodem nr-uesoftmodem
```

Verify the binary contains the policy strings:

```bash
strings nr-softmodem | grep -E "SCENESENSE_MCS_POLICY|SCENESENSE_FORCE_UL_MCS|SCENESENSE_AIMD"
```

## 7. T-tracer recording/extraction quick path

Start gNB and UE with T-tracer enabled:

```bash
# gNB
sudo ./nr-softmodem ... --T_stdout 2 --T_nowait --T_port 2021

# UE
sudo ./nr-uesoftmodem ... --T_stdout 2 --T_nowait --T_port 2023
```

In this repository, the helper scripts are:

```bash
abiodun/scripts/gnb_start_ttracer.sh
abiodun/scripts/ue_multi_start_ttracer.sh
abiodun/scripts/ttracer_record_smoke.sh
abiodun/scripts/ttracer_extract_csv_smoke.sh
```

Example record/extract pattern:

```bash
# record during the run
bash abiodun/scripts/ttracer_record_smoke.sh \
  --source gnb --profile latency --run-group <run_group> --duration-s 180

bash abiodun/scripts/ttracer_record_smoke.sh \
  --source ue --profile queue --run-group <run_group> --duration-s 180

# extract CSVs after the run
bash abiodun/scripts/ttracer_extract_csv_smoke.sh \
  --source gnb --profile latency --run-group <run_group> --clean-output

bash abiodun/scripts/ttracer_extract_csv_smoke.sh \
  --source ue --profile queue --run-group <run_group> --clean-output
```

For the MCS/link-adaptation diagnosis, make sure the gNB trace includes:

- `GNB_MAC_UL_MCS_DECISION`;
- `GNB_MAC_BLER_MCS_DECISION`;
- `GNB_MAC_UL`;
- `GNB_MAC_PUSCH_POWER_CONTROL`.

Make sure the UE trace includes:

- `NRUE_MAC_DCI_GRANT`;
- `NRUE_MAC_RLC_BUFFER_STATUS`;
- `NRUE_MAC_BSR_STATUS`.

## 8. Recommended replication sequence

Use the same PRB/TDD/payload/traffic pattern for all rows. Do not compare a
106PRB baseline against a 273PRB policy run and call that a policy effect.

### Step 1 — vanilla app baseline

Run the application over OAI with no MCS override.

Collect:

- app latency breakdown;
- gNB T-tracer full profile;
- UE T-tracer queue profile;
- delivery rate;
- payload size;
- offered app rate.

Expected if this bottleneck exists:

- high SNR;
- low MCS;
- growing RLC/BSR backlog;
- high uplink latency;
- downlink/edge not dominant.

### Step 2 — iperf sanity check

Run iperf on the same OAI setup.

If iperf receives high MCS and the app does not, the issue is not simply the RF
channel or PRB configuration. It is likely traffic cadence / scheduler behavior.

### Step 3 — fixed-MCS upper-bound diagnostic

Run:

```bash
sudo env SCENESENSE_FORCE_UL_MCS=28 ./nr-softmodem ...
```

or through our wrapper:

```bash
FORCE_UL_MCS=28 bash abiodun/downlink_latency_fps/run_oai_default106_ttracer_10fps.sh
```

Interpretation:

- If latency drops sharply and RLC backlog collapses, the bottleneck is MCS /
  spectral efficiency / drain rate.
- Do not present fixed MCS as a real deployment fix. It is unsafe under bad
  channel conditions.

### Step 4 — SINR-policy run

Run:

```bash
sudo env SCENESENSE_MCS_POLICY=sinr ./nr-softmodem ...
```

Expected clean-channel behavior:

- PUSCH SNR near clean RFsim value;
- selected UL MCS near 28;
- retransmissions near zero;
- RLC queue much smaller;
- uplink latency near the fixed-MCS/high-MCS diagnostic range.

### Step 5 — AWGN/channel ladder guardrail

Do not stop at the clean-channel result. Test at multiple controlled channel
qualities.

In our RFsim ladder, rough expected SINR-table behavior was:

| Channel rung | Observed SNR | SINR-policy MCS expectation |
|---|---:|---:|
| clean | about 50 dB | 28 |
| mild AWGN | about 19.5 dB | about 24 |
| medium/mid | about 15 dB or 10 dB, depending profile | lower than mild |
| strong | about 8 dB | much lower |

The guardrail:

> `avg_snr_x10` must move monotonically with the channel profile, and MCS must
> follow it. If SNR telemetry is not trustworthy, the SINR policy is not valid.

## 9. What to plot

For a team discussion, the most useful plots are:

1. Latency breakdown:
   - front/sensor;
   - uplink;
   - edge;
   - downlink, if closed loop.
2. MCS over time:
   - vanilla vs SINR/fixed/AIMD;
   - same PRB/TDD/payload.
3. Scheduled uplink Mbps / TBS:
   - MAC scheduled rate;
   - first-transmission vs retransmission if available.
4. UE RLC and BSR backlog:
   - LCID 4 RLC occupancy;
   - LCG 1 BSR.
5. SNR and BLER:
   - prove whether MCS is channel-driven or artifact-driven.
6. App offered rate vs served rate:
   - if offered > served, queueing/congestion is inevitable.

## 10. How to interpret common outcomes

| Observation | Likely interpretation |
|---|---|
| high SNR, zero BLER, low MCS | BLER/OLLA sparse-window artifact or scheduler policy issue |
| low SNR, high BLER, low MCS | MCS reduction may be correct; do not force high MCS |
| high MCS but still high latency | check grant rate, TBS, offered-vs-served rate, retransmissions, and app pacing |
| app offered Mbps > scheduled Mbps | RLC/BSR backlog and latency growth are expected |
| fixed MCS helps but SINR policy does not | verify `SCENESENSE_MCS_POLICY=sinr` actually reached `nr-softmodem`; check binary rebuild and run logs |
| iperf behaves fine but app does not | app traffic burstiness/pacing is part of the problem |
| 273PRB looks different from 106PRB | not a policy conclusion unless all other variables are controlled |

## 11. Config files to keep consistent

RAN configs:

```text
targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.sa.band78.fr1.106PRB.usrpb210.conf
targets/PROJECTS/GENERIC-NR-5GC/CONF/ue.conf
```

AWGN/channel configs in our tree:

```text
targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.sa.band78.fr1.106PRB.scenesense_rfsim.awgn_mild.conf
targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.sa.band78.fr1.106PRB.scenesense_rfsim.awgn_mid15.conf
targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.sa.band78.fr1.106PRB.scenesense_rfsim.awgn_strong.conf
targets/PROJECTS/GENERIC-NR-5GC/CONF/ue.awgn_mild.conf
targets/PROJECTS/GENERIC-NR-5GC/CONF/ue.awgn_mid15.conf
targets/PROJECTS/GENERIC-NR-5GC/CONF/ue.awgn_strong.conf
targets/PROJECTS/GENERIC-NR-5GC/CONF/channelmod_rfsimu_awgn_mild.conf
targets/PROJECTS/GENERIC-NR-5GC/CONF/channelmod_rfsimu_awgn_mid15.conf
targets/PROJECTS/GENERIC-NR-5GC/CONF/channelmod_rfsimu_awgn_strong.conf
```

Core-network files are usually not the MCS problem, but keep the UE DNN/IP
consistent:

```text
oai-cn5g/conf/config.yaml
oai-cn5g/database/oai_db.sql
```

## 12. Important caveats

1. The OAI SINR table is an existing OAI helper, but its source comment says it
   was generated from `nr_dlsim`. We used it as an OAI-provided SINR-driven
   diagnostic/policy path for UL PUSCH SNR, not as a final UL-calibrated table.
   A stricter final study can regenerate a UL-specific table using `nr_ulschsim`.

2. The SINR policy is best for controlled RFsim/channel sweeps where SNR is a
   meaningful state variable. In a real fading channel, validate with BLER,
   HARQ, and throughput before claiming it as a general fix.

3. Do not disable HARQ just to reach the OAI SINR branch. Keep HARQ enabled and
   branch to `get_mcs_from_SINRx10()` explicitly, otherwise retransmission
   metrics become invalid.

4. Always compare policies under the same:
   - PRB/bandwidth;
   - TDD pattern;
   - payload;
   - compression;
   - application pacing;
   - trace profile;
   - run duration.

5. If `T_messages.txt` changed, stale binaries can crash T-tracer or silently
   make extracted CSVs wrong. Rebuild both softmodems.

## 13. One-slide explanation for the team

> We found that the OAI uplink latency was caused by queueing at the UE RLC
> buffer. The queue formed because our bursty split-inference feature frames
> were served at low UL MCS, even on a clean RFsim channel. The default OAI
> BLER/OLLA selector can reduce MCS when the BLER update window has too few
> scheduled samples, which is common for bursty closed-loop traffic. Fixed MCS
> proved that high spectral efficiency removes the queue, and the SINR-policy
> patch uses OAI's existing SINR-to-MCS helper with measured PUSCH SNR while
> keeping HARQ enabled. To replicate, collect RAN/app traces, confirm high SNR
> + low MCS + RLC/BSR backlog, then compare vanilla, fixed-MCS, and
> `SCENESENSE_MCS_POLICY=sinr` under the same OAI config.
