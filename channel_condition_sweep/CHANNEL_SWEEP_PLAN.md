# Channel-condition sweep — PLAN

**Created:** 2026-07-31. **Home:** `abiodun/channel_condition_sweep/`. **For:** a fresh session (Abiodun
will run the OAI side separately). **Recommended model:** Opus 4.8 (OAI-stack orchestration + judgment on
channel realism + composing with offline accuracy; judgment-heavy, feeds the RL agent). Haiku-high only for
mechanical plotting afterward.

> Supplies the ONE missing factor in the locked RL design (`rl_agent/AGENT_CONSTRAINTS.md §9`): the
> transport function `transport_latency(payload, channel)` and `delivery(payload, channel)` over OAI. Merge
> with the pending **uplink-only-over-OAI** run — same stack, same T-tracer instrumentation.

## Question
For the uplink car→edge feature stream over OAI 5G, **how do end-to-end latency and delivery reliability
depend on payload size × channel quality (SNR)?** Combined with the (transport-invariant) offline
accuracy-vs-knob matrix, this answers: *at a given channel state and object speed, which knob meets the
freshness + accuracy budget at minimum airtime — and when is it worth trading segmentation (ROI drop) for
delivery?*

## Why this is the only thing left to measure (read `AGENT_CONSTRAINTS.md §9.4` first)
- **Transport-INVARIANT (reuse as-is, do NOT re-measure):** accuracy-vs-knob (recall/loc/mIoU) and
  payload-vs-knob. The codec is lossless, so the entire density/seg knob matrix
  (`rl_agent/PERMODEL_KNOB_MATRIX_ZSTD.md`, `density_knob/`) holds byte-for-byte over OAI.
- **Transport-DEPENDENT (this sweep):** payload → latency → **delivery**. Loopback is linear and ~free
  (0.009 ms/KB, delivery 1.00); OAI is ~14× steeper (≈0.13 ms/KB), nonlinear near capacity, with a delivery
  cliff (we already saw 75%→99% from a payload cut, `oai_compression_ab`). The bottleneck is the UE RLC
  queue-wait driven by the MCS cap (`oai_layer_latency`), which only appears under a real channel.

**Consequence — sweep PAYLOAD × SNR, not all 72 knobs.** Transport is a function of bytes-on-wire (+ burst
structure), so many knobs collapse to the same payload → same transport. Sweep a **payload ladder** and map
knobs → payload → transport afterward. This turns an intractable OAI grid into a few dozen runs.

## Channel mechanism — reuse SCAN-AI's approach WITHOUT SionnaRT
SCAN-AI (paper §5.1.3) precomputes a position→SNR heatmap with SionnaRT and injects it into OAI rfsim over
the **Telnet control interface**. The SionnaRT part is separable and — per supervisor — skipped. We reuse:

- **OAI native `channelmod` AWGN**, already wired: `RFSIM_CHANMOD=1` adds
  `--rfsimulator.[0].options chanmod --channelmod.modellist …` to both softmodems
  (`downlink_latency_fps/run_oai_bw273_ttracer_10fps.sh`), with configs
  `…273PRB.scenesense_rfsim.awgn.conf` + `ue.awgn.conf` → `channelmod_rfsimu.conf`.
- **The SNR knob** lives in `channelmod_rfsimu.conf`, UL model **`rfsimu_channel_ue0`** (AWGN, "modify on
  gNB side" — this is the uplink we care about): `noise_power_dB` (currently −10) and `ploss_dB`. Raising
  `noise_power_dB` (and/or `ploss_dB`) lowers uplink SNR. **Static per-run** is the right choice for a
  characterization sweep (controlled, repeatable) — one SNR value per run.
- **Optional later (closed-loop demo only, NOT this sweep):** OAI telnet `channelmod modify` at runtime for
  a position/time-varying SNR trace — exactly SCAN-AI §5.1.3, same parameter.

## Sweep design
1. **SNR ladder** (per-run, static): set `rfsimu_channel_ue0.noise_power_dB`/`ploss_dB` to a ladder giving
   roughly {clean/ideal, ~25, ~20, ~15, ~10, ~5 dB} effective uplink SNR — i.e. from "full MCS" down to
   "QPSK-capped, retransmitting." Calibrate the exact dB values to the observed gNB PUSCH SNR (guardrail 1).
2. **Payload ladder** (map from the knob matrix; u4 + zstd throughout): the seg-safe floor
   **`ae32/u4/ROI0` ≈ 90 KB**, plus a spread — e.g. ~15 KB (`ae32/u4/q0.9`), ~40 KB (`ae64/u4/q0.7`),
   ~130 KB (`ae128/u4/ROI0`), ~400 KB (`noae/u4/ROI0`), ~1 MB (`noae/u8/ROI0`, the uncompressed burst).
   ~6 payload points. Reuse the shaped-burst sender if a knob→bytes replay is cleaner than live inference
   (`oai_layer_latency/carla_shaped_udp_burst_sender.py`).
3. **Grid:** ~6 SNR × ~6 payload ≈ 36 runs, ~300 delivered frames (or fixed wall-time) each. Fewer if a
   region is clearly saturated. Fixed FPS (e.g. 10) unless FPS is added as a third axis later.

## Metrics per (payload × SNR) run — reuse the T-tracer + MAC-stats pipeline we already have
Log, per run and per-frame where possible:
- **delivery**: delivered-frame rate, and **fresh-delivered** rate (delivered within the freshness budget);
  drops / timeouts / no-result rate.
- **latency**: capture→result RTT distribution (p50/p95), and the layer split (front / RLC queue-wait /
  PHY / core) from the T-tracer (`oai_layer_latency/analyze_uplink_layer_latency.py`).
- **radio**: gNB PUSCH SNR, selected UL MCS, BLER, HARQ retransmissions, PRB/airtime occupancy
  (`scripts/parse_oai_gnb_mac_stats.py`, `analyze_mcs_decision_trace.py`, `analyze_bler_olla_trace.py`).
- **result age** at the map.

## Output — the transport surface + the composed policy
- `transport_latency(payload, SNR)` and `delivery(payload, SNR)` surfaces (tables + heatmap plots).
- **Compose with offline accuracy:** for each (SNR, object-speed) cell, the affordable payload = largest
  payload meeting the freshness budget at ≥target delivery; map that back to the best seg-safe knob
  (`ae*/u4/ROI0`); flag the cells where even the 90 KB floor is infeasible → the **ROI-escalation** region
  where the agent must trade seg (q 0.3/0.5) or drop FPS. This is the concrete evidence for
  `AGENT_CONSTRAINTS.md §9.2` lever 4 and the §9.1 `channel_state` variable.
- One-line note back into `AGENT_CONSTRAINTS.md §9` once measured.

## 🚦 Guardrails
1. **Confirm the SNR ladder actually moves MCS/BLER** before trusting any latency number. The config sweep
   (`oai_config_sweep`) found TDD/5QI/PRB levers were near-no-ops in clean single-UE channel; the AWGN runs
   (`oai_layer_latency/AWGN_HOLD_MCS_RESULTS`) did create real BLER/retx. Print gNB PUSCH SNR + median MCS
   per rung; if two rungs give the same MCS/BLER, the channel knob is not biting — fix the dB values, don't
   report a flat surface as if it were physics.
2. **MCS-cap / OLLA awareness.** The known uplink bottleneck is the QPSK MCS cap → RLC queue-wait
   (`oai_layer_latency`, root cause `gNB_scheduler_ulsch.c`). Run **vanilla adaptive MCS** for the
   characterization (not the `HOLD_MCS_FEW_SAMPLES` variant) so the surface reflects stock OAI; note it.
3. **Burst / packetization check** at one point: verify transport depends ~only on total bytes, not on how
   the payload is packetized (one dense AE tensor vs sparse ROI packets). If it does depend on burst
   structure, the payload ladder must hold packetization fixed or add it as a factor. Do not assume.
4. **Reuse the running OAI/CARLA; never kill another session's processes.** Check `/proc/loadavg`, GPU, and
   `docker ps` first. The T-tracer requires BOTH softmodems rebuilt after any `T_messages.txt` edit
   (`oai_ttracer_rebuild_constraint`) — do not edit it mid-sweep.
5. **Do NOT export `PYTHONPATH` for the CARLA client** (`dont_set_pythonpath_for_carla_client`): it shadows
   `abiodun/` with the stale `neu_collab/` copy → `UDPMessageSocket … remote_host`. Analysis scripts only.
6. **8 MB socket buffers** (`sudo sysctl net.core.rmem_max=8388608`) — the overnight sweep once silently
   reset to 212992 and Stage 2 died; fail-fast check it at the top of the runner.
7. **Validate + demote, don't rescue.** If a run's delivery is pathological (e.g. 1/100 frames, the
   bounded-buffer artifact), demote it and say so; do not average it into the surface.
8. **IP-pool gotcha** (`oai_config_sweep`): after an NF/UE restart the UE can come up as `.2` not `.3`;
   re-check the tunnel IP before each run block.

## Reuse / outputs / merge
- **Reuse:** `downlink_latency_fps/run_oai_bw273_ttracer_10fps.sh` (RFSIM_CHANMOD path), the `.awgn.conf`
  configs, `channelmod_rfsimu.conf`, `oai_layer_latency/` analyzers (layer latency, MCS, BLER, MAC stats),
  `carla_shaped_udp_burst_sender.py`, and the offline knob matrix for payload↔knob↔accuracy.
- **Outputs →** `channel_condition_sweep/`: `CHANNEL_SWEEP_RESULTS.md` (surfaces + composed policy + the
  ROI-escalation region), raw per-run CSVs, heatmap plots, and the one-line `channel_state` note for
  `AGENT_CONSTRAINTS.md §9`.
- **Merge with the pending uplink-only-over-OAI run** — same stack; do the uplink-only validation of the
  seg-aware ~90 KB knob (does it hold ≥ target delivery over OAI?) as the first cell of this grid.
