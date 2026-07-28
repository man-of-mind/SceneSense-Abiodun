# Bursty Traffic Replay Plan for OAI Link-Adaptation Study

Date: 2026-07-27

## Goal

Test whether the low-MCS / RLC-queue behavior we observed for CARLA split
inference is specific to our closed-loop split-tensor traffic, or whether OAI's
current BLER/OLLA link adaptation also struggles with other real bursty
application traffic patterns.

The core question:

> Does bursty uplink traffic with sparse scheduling windows cause OAI to spend
> too much time in the `num_sched <= 3` branch and hold/decrease MCS despite high
> SNR?

## Data

Downloaded source:

- `TRACTOR/raw/` from `https://github.com/genesys-neu/TRACTOR`

Raw trace schema:

```text
App name, No., Time, Source, Destination, Protocol, Length
```

Default UE/phone IP detected in the traces:

- `172.30.1.250`

For replay, packets where `Source == 172.30.1.250` are treated as uplink.

## Local helpers

- `analyze_tractor_raw.py`: offline burstiness summary for all raw traces.
- `udp_trace_replay.py`: replays packet timing and packet length as UDP traffic.
- `udp_sink.py`: simple UDP receiver/logger on the edge/DN side.

Generated summary:

- `analysis/tractor_trace_summary.csv`

## Initial trace candidates

These are selected to span heavy uplink, real-time video/voice, background
sparse traffic, and mixed traffic.

| Trace | Traffic type | Why include |
|---|---|---|
| `embb_03_03a.csv` | OneDrive large video download/upload trace; heavy uplink in raw data | strongest uplink volume and high 1s peak rate |
| `embb_03_03b.csv` | OneDrive heavy uplink | repeat of large-uplink behavior for robustness |
| `embb_2.csv` | YouTube / UDP-heavy eMBB | bursty high peak but likely less closed-loop than CARLA |
| `urllc_03_03.csv` | Google Meet video chat while walking | real-time bidirectional traffic |
| `urllc_05_18.csv` | Facebook Messenger video chat | second real-time video-chat pattern |
| `urllc_06_12.csv` | Microsoft Teams call | Teams voice/video pattern |
| `mmtc_05_18.csv` | background traffic | sparse low-volume burst baseline |
| `mixed.csv` | mixed application traffic | closer to real UE multiplexing |
| CARLA split tensor | our traffic | large uplink burst + edge-result wait loop |
| CARLA-shaped synthetic replay | controlled version of our traffic | isolates traffic timing/size from CARLA/model compute |

## OAI patch under test

Patch location:

- `../OAI/openairinterface5g/openair2/LAYER2/NR_MAC_gNB/gNB_scheduler_primitives.c`

New runtime flag:

```bash
SCENESENSE_HOLD_MCS_FEW_SAMPLES=1
```

Behavior:

- flag unset / `0`: default OAI behavior remains active;
- flag `1`: when `num_sched <= 3`, branch `3` holds MCS instead of decrementing.

This lets us compare default OAI versus the few-samples-hold behavior using the
same rebuilt binary.

## Experiment sequence

### Phase 0: offline trace characterization

Already started:

```bash
cd abiodun/bursty_traffic
python3 analyze_tractor_raw.py --raw-glob 'TRACTOR/raw/*.csv' --out analysis/tractor_trace_summary.csv
```

Outputs to use for trace selection:

- uplink MB;
- mean/p95/peak 1s uplink Mbps;
- peak-to-mean ratio;
- p50/p95/max burst size;
- p50/p95/max inter-uplink-packet gap.

### Phase 1: default OAI replay baseline

Purpose:

- establish how current OAI treats each bursty trace before the MCS-hold patch.

Run with:

```bash
SCENESENSE_HOLD_MCS_FEW_SAMPLES=0
```

For each selected trace:

1. Start OAI default 106PRB or 273PRB path with T-tracer enabled.
2. Start UDP sink on edge/DN side:

   ```bash
   python3 abiodun/bursty_traffic/udp_sink.py --bind 0.0.0.0 --port 55000 --out <run_dir>/udp_sink.csv
   ```

3. Replay selected uplink trace from UE side:

   ```bash
   python3 abiodun/bursty_traffic/udp_trace_replay.py \
     abiodun/bursty_traffic/TRACTOR/raw/<trace>.csv \
     --dst <edge_or_dn_ip> \
     --port 55000 \
     --direction uplink \
     --max-payload 1400
   ```

4. Collect T-tracer metrics:

   - `GNB_MAC_BLER_MCS_DECISION`
   - `GNB_MAC_UL_MCS_DECISION`
   - `NRUE_MAC_BSR_STATUS`
   - `NRUE_MAC_RLC_BUFFER_STATUS`
   - `NRUE_MAC_DCI_GRANT`
   - `GNB_MAC_LCID_UL`
   - `GNB_MAC_PUSCH_POWER_CONTROL`

### Phase 2: patched OAI replay

Repeat Phase 1 with:

```bash
SCENESENSE_HOLD_MCS_FEW_SAMPLES=1
```

Primary comparison:

- MCS p50/p95;
- branch-3 rate and whether final MCS stays higher;
- RLC queue occupancy;
- BSR backlog;
- delivered throughput and packet delivery at the UDP sink.

### Phase 3: CARLA control under patched OAI

Run the corrected CARLA drivable-route no-AE full-payload case with:

```bash
SCENESENSE_HOLD_MCS_FEW_SAMPLES=1
```

Compare against existing adaptive-MCS and fixed-MCS28 results:

- if latency moves toward fixed-MCS28, link adaptation was the bottleneck;
- if MCS stays low or delivery remains poor, the next candidate patch is
  stricter: decrement only on high BLER with enough scheduled samples.

### Phase 4: CARLA-shaped synthetic replay

Replay a synthetic trace matching CARLA's application pattern:

- ~1.05 MB uplink burst;
- edge-result wait gap based on observed RTT;
- repeat at closed-loop pace.

Purpose:

- isolate traffic shape from CARLA rendering/model execution.

## Metrics for final analysis

For each run:

| Metric | Why |
|---|---|
| MCS p50/p95 over active window | shows spectral-efficiency behavior |
| `num_sched` distribution | shows whether sparse scheduling triggers branch 3 |
| branch code distribution from `GNB_MAC_BLER_MCS_DECISION` | direct evidence for BLER/OLLA behavior |
| RLC LCID4 occupancy | actual UE-side data queue |
| BSR LCG1 backlog | scheduler-facing backlog |
| PRB allocation and TBS | confirms whether available bandwidth is being used |
| UDP sink delivery and throughput | application-level delivery check |
| SNR / PUSCH power-control trace | proves channel is not poor |

## Expected interpretations

If only CARLA / CARLA-shaped traffic suffers:

- split inference has a distinct traffic class: large uplink bursts plus
  application-level closed-loop gaps.

If TRACTOR heavy/bursty traces suffer too:

- OAI's BLER/OLLA handling of sparse bursty uplink is generally fragile.

If the hold patch improves CARLA but not all TRACTOR traces:

- the branch-3 few-samples decrement is central to split-inference latency, but
  other traffic may be constrained by different factors.

If the hold patch does not improve CARLA:

- next patch: decrement only on real high BLER with enough samples, or add
  high-SNR guard/fallback.

## First vanilla-OAI replay results (2026-07-27)

We suspended the AWGN/bad-channel thread and ran two TRACTOR traces over the
clean/default 273PRB RFsim OAI path with vanilla OAI adaptive MCS:

- UE T-tracer profile: `all`
- gNB T-tracer profile: `latency`
- channel model: default RFsim, no AWGN/channelmod
- sink: tcpdump capture on `oai-ext-dn`
- replay: uplink rows from the TRACTOR UE IP, large raw rows split into
  MTU-sized UDP datagrams instead of capped.

| Trace/window | Offered | Delivery | Median UL MCS | UL retx | Few-sample branch | RLC LCID4 p95 | Mean RLC wait | RAN UL p50/p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `embb_03_03a.csv`, offset 100 s, 60 s | 6.09 Mbps | 100% | 27.9 | 0.0% | 59.0% | 8.1 KB | 3.6 ms | 5.2 / 7.2 ms |
| `urllc_03_03.csv`, offset 240 s, 60 s | 1.12 Mbps | 100% | 27.9 | 0.0% | 58.2% | 2.2 KB | 4.0 ms | 5.6 / 7.5 ms |

Important early interpretation:

- These real TRACTOR bursty traces **do not reproduce** the CARLA split-tensor
  low-MCS/RLC-queue failure.
- The few-sample branch appears frequently in both TRACTOR runs, but MCS remains
  high (`~28`), retransmissions are zero, and RLC queueing stays tiny.
- This suggests the few-sample branch alone is not sufficient to explain the
  CARLA failure. The stronger differentiator is CARLA's closed-loop whole-frame
  split-feature pattern: a large ~1 MB feature burst must fully drain before the
  edge can infer and return a result.

Artifacts:

- OneDrive summary:
  `../metrics_logs/tractor_replay/tractor_replay_bw273_vanilla_embb0303a_off100_60s_tcpdump_20260727_210512/tractor_oai_summary.csv`
- Google Meet summary:
  `../metrics_logs/tractor_replay/tractor_replay_bw273_vanilla_urllc0303_off240_60s_tcpdump_20260727_210842/tractor_oai_summary.csv`
- Combined summary:
  `analysis/tractor_oai_vanilla_baseline_summary.csv`
- Summary plot:
  `plots/tractor_oai_vanilla_baseline_summary.png`
- Summary plot PDF:
  `plots/tractor_oai_vanilla_baseline_summary.pdf`
- Focused traffic-rate/MCS time-series plot:
  `plots/tractor_vs_carla_rate_mcs_timeseries.png`
- Focused traffic-rate/MCS time-series plot PDF:
  `plots/tractor_vs_carla_rate_mcs_timeseries.pdf`
- Full traffic/MCS/BSR/RLC time-series evidence:
  `plots/tractor_vs_carla_timeseries_rate_mcs_queue.png`
- Full traffic/MCS/BSR/RLC time-series evidence PDF:
  `plots/tractor_vs_carla_timeseries_rate_mcs_queue.pdf`
- Time-series summary table:
  `analysis/tractor_vs_carla_timeseries_summary.csv`
- Matched 70 s traffic/MCS plot:
  `plots/tractor_vs_carla_rate_mcs_timeseries_70s.png`
- Matched 70 s traffic/MCS plot PDF:
  `plots/tractor_vs_carla_rate_mcs_timeseries_70s.pdf`
- Fine-bin 100 ms traffic/MCS plot:
  `plots/tractor_vs_carla_traffic_100ms_mcs_70s.png`
- Fine-bin 100 ms traffic/MCS plot PDF:
  `plots/tractor_vs_carla_traffic_100ms_mcs_70s.pdf`
- Fine-bin activity summary:
  `analysis/tractor_vs_carla_100ms_activity_summary.csv`
- Preferred slide-clean aligned traffic/MCS plot:
  `plots/tractor_vs_carla_clean_aligned_traffic_mcs_70s.png`
- Preferred slide-clean aligned traffic/MCS plot PDF:
  `plots/tractor_vs_carla_clean_aligned_traffic_mcs_70s.pdf`
- Preferred slide-clean aligned summary:
  `analysis/tractor_vs_carla_clean_aligned_70s_summary.csv`
- Simple 1-second traffic-rate plot:
  `plots/tractor_vs_carla_traffic_rate_1s_70s.png`
- Simple 1-second traffic-rate plot PDF:
  `plots/tractor_vs_carla_traffic_rate_1s_70s.pdf`
- Simple 1-second traffic-rate summary:
  `analysis/tractor_vs_carla_traffic_1s_summary.csv`
- RLC drain/queue/MCS comparison plot:
  `plots/tractor_vs_carla_rlc_drain_queue_mcs_70s.png`
- RLC drain/queue/MCS comparison plot PDF:
  `plots/tractor_vs_carla_rlc_drain_queue_mcs_70s.pdf`
- RLC drain/queue/MCS comparison summary:
  `analysis/tractor_vs_carla_rlc_drain_summary.csv`

Time-series comparison with the CARLA 273PRB no-AE split-inference baseline:

- TRACTOR has real burstiness, but the scheduler generally reaches/holds high
  MCS (`~28`) and the UE queues stay small.
- CARLA repeatedly offers large whole-frame feature bursts (`~1 MB/frame`),
  while active UL MCS remains around `4`.
- In the CARLA run, BSR LCG1 and RLC LCID4 remain near the full feature-burst
  scale for long intervals (`~1 MB`), while TRACTOR BSR/RLC are mostly tens of
  KB or less.
- This strengthens the story: the issue is not generic bursty traffic alone.
  The problematic pattern is the CARLA split-inference burst shape combined
  with low OAI adaptive-MCS drain rate.

Fine-bin traffic check over the first 70 s using 100 ms bins with zeros
included:

| Traffic | Sustained rate over 70 s | Active 100 ms bins | Median rate during active bins |
|---|---:|---:|---:|
| TRACTOR OneDrive eMBB | 5.2 Mbps | 82% | 4.0 Mbps |
| TRACTOR Google Meet URLLC | 1.0 Mbps | 84% | 1.0 Mbps |
| CARLA split inference | 13.3 Mbps | 15% | 86.3 Mbps |

Simpler 1-second traffic-rate view over the first 70 s:

| Traffic | Mean rate | 1 s p50/p95 | Typical count per 1 s |
|---|---:|---:|---:|
| TRACTOR OneDrive eMBB | 5.2 Mbps | 3.8 / 17.2 Mbps | many UDP datagrams |
| TRACTOR Google Meet URLLC | 1.0 Mbps | 0.8 / 1.7 Mbps | many UDP datagrams |
| CARLA split inference | 13.3 Mbps | 17.1 / 25.8 Mbps | 2 frames p50, 3 frames p95 |

Use the 1-second plot as the first/intuitive traffic-rate plot. It shows CARLA
exactly as expected from the payload arithmetic: one `~1.08 MB` feature frame is
`~8.6 Mb`, so `1/2/3` frames in a second appear as roughly
`8.6/17.2/25.8 Mbps`. The 100 ms plot is only a second-level zoom that explains
why each frame appears as a tall instantaneous spike inside a shorter window.

The `86.3 Mbps` CARLA value is **not** a sustained 1-second average. It is the
equivalent rate inside active 100 ms bins. A typical CARLA feature payload is
about `1.08 MB`, so one feature burst in a 100 ms bin gives
`1.08 MB * 8 / 0.1 s ~= 86 Mbps`. If CARLA sustained 10 FPS with one 1 MB
feature frame every 100 ms, that would indeed imply about `80-90 Mbps`. In this
OAI closed-loop run, however, CARLA is active in only about `15%` of 100 ms
bins, so the sustained average is much lower while the instantaneous burst
rate remains high.

Measured CARLA no-AE 273PRB effective send rate:

- Frames logged: `1300`
- Run duration after first payload: `~849 s`
- Effective frame rate: `1300 / 849 ~= 1.53 FPS`
- Payload p50: `~1.08 MB ~= 8.64 Mb/frame`
- Sustained offered rate: `1.53 FPS * 8.64 Mb ~= 13.2 Mbps`
- First aligned 70 s: `108` payloads in `70 s` => `1.54 FPS`, `~13.3 Mbps`
- In 1-second bins, p50 is `2 frames/s` => `~17.1 Mbps`; p95 is
  `3 frames/s` => `~25.8 Mbps`.
- The 100 ms active-bin fraction is `108 / 700 = 15.4%` in the first 70 s:
  one frame burst occupies one 100 ms bin, followed by several empty bins.

So the clean wording is:

> CARLA is configured for a 10 FPS target, but this closed-loop OAI run
> effectively sends about 1.5 FPS because the front end waits on the result path
> and timeouts. Each actual send is still a large ~1 MB burst, so the
> instantaneous active-bin rate is ~86 Mbps, while the sustained offered rate is
> only ~13 Mbps.

Interpretation: TRACTOR is bursty, but still mostly continuously active at
100 ms resolution. CARLA is much more sparse: it injects large feature bursts,
then waits. This burst/wait structure is hidden by 1-second aggregation and is
likely important for why CARLA sees persistent low MCS plus large RLC/BSR
backlog.

RLC drain comparison over the first aligned 70 s. The drain p50/p95 values
include zero/idle seconds so they show the full timeline, not only active
transmission periods:

| Traffic | RLC drain p50/p95 | RLC LCID4 p95 | Active MCS p50 |
|---|---:|---:|---:|
| TRACTOR OneDrive eMBB | 3.9 / 17.7 Mbps | 36.7 KB | 27.9 |
| TRACTOR Google Meet URLLC | 0.9 / 1.8 Mbps | 4.4 KB | 27.9 |
| CARLA split inference | 8.5 / 18.1 Mbps | 1011.6 KB | 3.9 |

This follows the earlier adaptive-vs-fixed-MCS analysis: MCS controls the RLC
drain rate. TRACTOR's offered load is drained without persistent queue buildup.
CARLA's instantaneous feature bursts are much larger than the low-MCS drain can
clear quickly, so the RLC queue stays near one full feature frame.

How to read the RLC plot:

- `RLC dequeue Mbps` is the observed service/drain rate: bytes leaving UE RLC
  toward MAC per second.
- `RLC LCID4 KB` is the queue/backlog level in the UE data bearer.
- They are different units: one is a flow rate, the other is stored data.
- Queue evolution is approximately:
  `next backlog = current backlog + new feature bytes - dequeued bytes`.
- Example: if CARLA shows `~1000 KB` LCID4 backlog and `~15 Mbps` dequeue, then
  roughly one feature frame is waiting while RLC drains at `~15 Mbps`. If no new
  frame arrived, `1 MB * 8 / 15 Mbps ~= 0.5 s` would clear it. But because new
  ~1 MB feature bursts arrive before the queue fully drains, backlog remains
  near one frame. The plot is therefore evidence of burst/service mismatch, not
  a direct per-frame latency measurement.

For presentation, use the `clean_aligned` plot. It zeroes each series at the
first active traffic/scheduler sample, removing raw trace startup/idling time.
This is why CARLA begins at `t=0` in the clean plot instead of appearing shifted
to the right by the pre-traffic scheduler trace.

Next recommended run:

- Run a **CARLA-shaped closed-loop synthetic replay** beside these TRACTOR traces
  using the same tcpdump + `all`/`latency` tracing path. This isolates the
  application traffic shape without CARLA/model compute and should directly test
  whether whole-frame closed-loop pacing is the real trigger.
