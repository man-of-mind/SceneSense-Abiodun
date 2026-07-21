# OAI Config CARLA Frontend Results

Date: 2026-07-20

Purpose: compare simple OAI network-side config changes using the actual live CARLA frontend path, not synthetic replay.

This is the corrected Step-1-style comparison after the quick replay diagnostic. All runs use:

- live CARLA frontend: `staleness/carla_fusion_staleness_scenario.py --role front`
- moving ego on the fixed training route
- no-AE checkpoint: `experiments/ae_integrated_20260710/noae_baseline/checkpoints/mprime_joint_noae/best.pt`
- 200k radar PPS
- zlib + per-channel uint8 split features
- 10 FPS target, 1300 frontend frames
- remote OAI back-half at `192.168.70.140:51002`
- result return to the active UE tunnel IP

## Closed-loop CARLA frontend comparison

Important: this is the normal frontend mode. It waits for the result or timeout before advancing to the next frame. So it measures closed-loop application behavior, not an open-loop 10 FPS offered-load stress test.

| Condition | RAN config | Frames | Returned | Delivery | Ego speed mean | Moving frac | RTT p50 | RTT p95 | Front p50 | Back p50 | Downlink p50 | Feature/uplink handling p50 | Capture→result p50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Default OAI | 106 PRB, mu=1, default TDD 7 DL / 2 UL | 1300 | 944 | 72.6% | 0.907 m/s | 32.2% | 209.0 ms | 279.5 ms | 49.4 ms | 7.1 ms | 11.1 ms | 187.4 ms | 264.0 ms |
| UL-heavy TDD | 106 PRB, mu=1, TDD 4 DL / 5 UL | 1300 | 935 | 71.9% | 0.893 m/s | 32.3% | 200.1 ms | 249.5 ms | 48.6 ms | 10.3 ms | 9.8 ms | 177.0 ms | 252.9 ms |
| Wider bandwidth | 273 PRB, mu=1, default TDD 7 DL / 2 UL | 1300 | 974 | 74.9% | 0.836 m/s | 29.9% | 235.2 ms | 263.7 ms | 48.5 ms | 7.4 ms | 9.6 ms | 216.1 ms | 285.7 ms |

Important correction to the correction: the `Wider bandwidth / 273 PRB` row is
reportable for the **manual validated `bw273_mu1` run**. The matching RAN logs
show the gNB using `gnb.sa.band78.fr1.273PRB.scenesense_rfsim.conf`, the UE
launched with `-r 273 -C 3649260000 --ssb 516`, `N_RB_DL 273`, and the UE tunnel
IP `10.0.0.5`, which matches the CARLA frontend bind host. Do not confuse this
with the earlier automated `prb_273` sweep attempt, which failed because the UE
was launched with mismatched center-frequency/SSB settings.

## Interpretation

The live CARLA frontend result is different from the synthetic replay diagnostic.

In the normal closed-loop CARLA harness, the validated OAI config changes do
**not** solve delivery:

- default OAI: 72.6%
- UL-heavy TDD: 71.9%
- wider bandwidth 273PRB: 74.9%

UL-heavy TDD is still useful, but mainly as a latency improvement:

- RTT p50 improves from 209.0 ms to 200.1 ms
- RTT p95 improves from 279.5 ms to 249.5 ms
- feature/uplink handling p50 improves from 187.4 ms to 177.0 ms

The validated 273PRB run improves delivery slightly relative to default OAI,
but it **worsens** p50 latency:

- RTT p50 increases from 209.0 ms to 235.2 ms
- feature/uplink handling p50 increases from 187.4 ms to 216.1 ms
- capture-to-result p50 increases from 264.0 ms to 285.7 ms

So the clean Step-1 conclusion should be conservative:

> In the actual closed-loop CARLA deployment, validated OAI-side changes produce
> only modest/mixed movement. UL-heavy TDD improves latency slightly but not
> delivery; 273PRB improves delivery slightly but worsens p50 latency. Neither
> gets close to reliable low-latency 10 FPS delivery for the ~1.1 MB no-AE split
> payload.

This keeps compression/model-side payload reduction as the stronger lever for
the RL-agent action space. Network-side controls remain useful as
secondary/contextual actions, especially under future
Sionna/multi-UE/channel-stress experiments, but they should not be framed as
sufficient by themselves in the current single-UE RFsim closed-loop deployment.

## Replay diagnostic caveat

The replay results were still useful, but they answer a different question:

- replay = open-loop, fixed-rate transport stress with synthetic no-AE payloads
- CARLA frontend = closed-loop deployed application behavior with real CARLA frames, sensors, traffic, and result wait/timeout coupling

Replay showed stronger config sensitivity because it forced a clean ~92 Mbps offered load. The live CARLA frontend does not maintain that same open-loop offered load because it waits on result/timeout before advancing.

If we want a CARLA-based open-loop offered-load test, use the existing `--queue-probe-mode`: it still uses real CARLA frames but sends at fixed wall-clock FPS and logs send/result events separately.

## 273PRB validation status

There are two different 273PRB cases:

- **Validated manual `bw273_mu1` run, reportable.** This is the row in the table
  above. Evidence is in `../metrics_logs/oai_config_sweep/bw273_mu1_carla_live/`:
  the gNB log shows `DLBW 273` / `fp->N_RB_DL=273`; the UE log shows `-r 273`,
  `-C 3649260000`, `--ssb 516`, `N_RB_DL 273`, and tunnel IP `10.0.0.5`.
- **Automated `prb_273` sweep / later reproduction attempts, not reportable.**
  These failed because the UE launch did not match the working 273PRB center
  frequency/SSB setup, or never reached a usable tunnel.

Date: 2026-07-21

We attempted to rerun the 273PRB condition with UE-side T-tracer enabled so the
same PRB/MCS/scheduled-rate plot could be generated for the wider-bandwidth
case. Those attempts are **not** replacements for the validated manual run:
they used the wrong/incomplete 273PRB bring-up recipe and did not reach the same
usable `10.0.0.5` tunnel.

Attach diagnostics:

| Attempt | UE config | Softmodem T-tracer | gNB min_rxtxtime | Result |
|---|---|---:|---:|---|
| `downlink_oai_bw273_mu1_ttracer_fps10_20260721_161914` | `ue.multi2.conf` | on | 6 | Failed; UE-side config stayed at `N_RB_DL 106`, so this was not a valid 273PRB UE attach. |
| `downlink_oai_bw273_mu1_ttracer_fps10_20260721_162243` | `ue.conf` + `-r 273` but wrong/default center-frequency path | on | 6 | Failed; UE reached `N_RB_DL 273` and initial sync, then repeated SIB1/PBCH failures; no usable tunnel. |
| `downlink_oai_bw273_mu1_ttracer_fps10_attach_plain_20260721_162656` | `ue.conf` + `-r 273` but wrong/default center-frequency path | off | 6 | Failed the same way, so this was not caused by T-tracer recording. |
| `downlink_oai_bw273_mu1_ttracer_fps10_attach_plain_min12_20260721_163159` | `ue.conf` + `-r 273` but wrong/default center-frequency path | off | 12 | Failed the same way; the simple timing increase did not fix the mismatched bring-up. |

Current interpretation: keep the validated `bw273_mu1` live CARLA result in the
comparison, but do **not** generate or report a 273PRB PRB/MCS T-tracer plot
until we rerun T-tracer with the exact working recipe:
`-r 273 -C 3649260000 --ssb 516` plus the 273PRB gNB config.

## Artifacts

- Default OAI closed-loop run: `runs/oai_default/fps_10_oai_default_20260720_one_loop/`
- UL-heavy closed-loop run: `runs/oai_ulheavy_106/fps_10_oai_ulheavy106_carla_10fps_full_20260720/`
- Validated 273PRB closed-loop run: `runs/oai_bw273_mu1/fps_10_oai_bw273_mu1_carla_10fps_full_20260720/`
- UL-heavy summary CSV: `runs/downlink_fps_summary_oai_ulheavy106_carla_10fps_full_20260720.csv`
- Validated 273PRB summary CSV: `runs/downlink_fps_summary_oai_bw273_mu1_carla_10fps_full_20260720.csv`
- Validated 273PRB RAN proof: `../metrics_logs/oai_config_sweep/bw273_mu1_carla_live/`
- Failed 273PRB T-tracer attach-attempt logs: `../metrics_logs/carla_oai_ttracer/downlink_oai_bw273_mu1_ttracer_fps10_20260721_162243/`, `../metrics_logs/carla_oai_ttracer/downlink_oai_bw273_mu1_ttracer_fps10_attach_plain_20260721_162656/`, `../metrics_logs/carla_oai_ttracer/downlink_oai_bw273_mu1_ttracer_fps10_attach_plain_min12_20260721_163159/`
