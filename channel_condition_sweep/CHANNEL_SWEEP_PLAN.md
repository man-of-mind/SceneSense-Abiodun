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

- **Config family: 106PRB ONLY** (locked 2026-08-03 — Track-2 keeps one OAI config). The official
  uplink-only sweep uses the 106PRB clear/AWGN ladder in
  `uplink_only_spatial_map_pipeline/run_track2_sinr_uplink_only_ladder.sh`:
  `clear_sinr`, `mild_sinr`, `mid15_sinr`, `strong_sinr`. Do **not** use the 273PRB runner here, and do
  **not** use the closed-loop `oai_mcs_policy_track2/run_awgn_106prb_ladder.sh` for the payload knee.
- **OAI native `channelmod` AWGN**, already wired: `RFSIM_CHANMOD=1` adds
  `--rfsimulator.[0].options chanmod --channelmod.modellist …` to both softmodems.
- **The SNR knob** lives in `channelmod_rfsimu.conf`, UL model **`rfsimu_channel_ue0`** (AWGN, "modify on
  gNB side" — the uplink we care about): `noise_power_dB` (mild −10 ⇒ ~19.5 dB, mid15 −8 ⇒ ~15.6 dB,
  strong −4 ⇒ ~8.2 dB observed) and `ploss_dB`. **Static per-run** (one SNR per run).
- **MCS policy: REQUIRED `SCENESENSE_MCS_POLICY=sinr`** (2026-08-03). This is not optional for the sweep: on
  the good/clean rung the reactive BLER-OLLA collapses UL MCS to ~7 via the sparse-window artifact
  (`oai_mcs_policy_track2`), which would make the high-SNR end of the transport surface an MCS artifact, not
  the channel. SINR-driven MCS follows the injected SNR (clean→28, mild→24, mid15→19, strong→9), keeps HARQ
  on, and measured ~0% retx / higher delivery / stable MCS vs vanilla (`results/awgn106_ladder_track2_sinr_awgn_ladder_20260803.md`).
- **Optional later (closed-loop demo only, NOT this sweep):** OAI telnet `channelmod modify` at runtime for
  a position/time-varying SNR trace — exactly SCAN-AI §5.1.3, same parameter.

## Sweep design
1. **SNR ladder** (per-run, static): the official grid uses clean(~50) / mild(~19.5) / mid15(~15.6) /
   strong(~8.2). Add intermediate/low rungs later (~35, ~25, ~10, ~5 dB) by tuning `noise_power_dB`/`ploss_dB`, so
   the ladder covers the range a realistic trace visits (e.g. 50-15-35-20-45-5). **Calibrate dB→observed gNB
   PUSCH SNR first** (guardrail 1) — the low end is nonlinear (a 1 dB noise_power step moves SNR a lot).
2. **Payload as an INDEPENDENT axis — decoupled from the compression knobs.** Vary payload with the
   **shaped-burst sender** (`oai_layer_latency/carla_shaped_udp_burst_sender.py`): it puts a fixed N-KB UDP
   burst on the wire per frame, no model in the loop, so payload is clean bytes — not confounded with any
   knob's accuracy cost. This is *why* we don't use the compression knobs to make payload: transport is a
   function of bytes (+ burst structure), and we want that function measured cleanly. **Ladder: ~10, 25, 50,
   90, 150, 300, 600, ~1046 KB** (start at ~10 KB, not 50–100 — the aggressive-compression regime is exactly
   where a bad channel forces the agent to live). The knob→payload mapping is applied *afterward* from the
   offline matrix (`ae32/u4/q0.9`≈16 KB … `ae32/u4/ROI0`≈90 KB … `noae/u8/ROI0`≈1046 KB).
3. **Accuracy is NOT re-measured per SNR — it is composed.** A delivered feature tensor decodes identically
   regardless of the channel (lossless codec), so per-frame model accuracy is transport-invariant and already
   known from the knob matrix. What the channel changes is **which frames arrive and how late** → delivery +
   staleness. End-to-end localization error = compose( per-frame floor+knob accuracy [knob matrix] , staleness
   from measured latency/delivery [this sweep] , object speed [AGENT_CONSTRAINTS §1] ). So the sweep measures
   latency+delivery; accuracy enters via the staleness model, not a re-run of the fusion model.
4. **Validation runs (few):** run a handful of REAL closed-loop CARLA runs (actual knob → real payload +
   real delivery) at 2–3 SNR rungs, confirm they land where the shaped-burst surface + composition predict.
5. **Grid:** ~8 SNR × ~8 payload shaped-burst runs + ~3 validation runs, ~300 delivered frames each,
   fixed FPS 10 (add FPS as a third axis only if the freshness budget needs it).

## Metrics per (payload × SNR) run — reuse the T-tracer + MAC-stats pipeline we already have
Log, per run and per-frame where possible:
- **delivery**: delivered-frame rate, and **fresh-delivered** rate (delivered within the freshness budget);
  drops / timeouts / no-result rate.
- **latency**: capture→result RTT distribution (p50/p95), and the layer split (front / RLC queue-wait /
  PHY / core) from the T-tracer (`oai_layer_latency/analyze_uplink_layer_latency.py`).
- **radio**: gNB PUSCH SNR, selected UL MCS, BLER, HARQ retransmissions, PRB/airtime occupancy
  (`scripts/parse_oai_gnb_mac_stats.py`, `analyze_mcs_decision_trace.py`, `analyze_bler_olla_trace.py`).
- **congestion indicator (REQUIRED — the uplink-only lesson):** offered app rate vs scheduled/drain rate,
  and the **UE RLC/BSR backlog trend over the run** (growing-and-pinned vs flat-and-bounded). A rung is only
  a valid latency point if the backlog is bounded; a growing/saturated backlog means offered > capacity and
  the latency is unbounded queueing, not the channel — flag and demote it (do NOT report it as "latency at
  X dB"). See `uplink_only_spatial_map_pipeline/results/presentation_sinr_uplink_only/` — 1 MB open-loop
  offered 50–67 Mbps, pinned the UE buffer at 47.7 MiB below clean, and gave 6–15 s latency at ~0% retx.
- **result age** at the map.

## Output — the transport surface + the composed policy
- `transport_latency(payload, SNR)` and `delivery(payload, SNR)` surfaces (tables + heatmap plots).
- **THE headline deliverable — the per-SNR congestion knee `payload_stable(SNR)`.** For each SNR rung, find
  the largest payload that stays on the *good* side of the knee: delivery ≥ ~99%, bounded (non-growing) RLC
  backlog, and latency at its flat floor (not the queueing blow-up). This is precisely the user's intuition
  — "at lower payload we hold ~100% delivery with low latency" — made quantitative. Expect a sharp knee at
  offered ≈ capacity (~7.2 Mbps at the 90 KB/10 fps floor sits *under* capacity even at Strong 8.2 dB
  ~12 Mbps, so the seg-safe floor should stay stable far down the ladder — the sweep confirms exactly how
  far).
- **This IS the agent's core control law.** `payload_stable(SNR)` × a safety margin = the agent's payload
  target for the current `channel_state`; the knob ladder then picks the max-accuracy config whose payload ≤
  target (u4 → AE bottleneck → … ), and only when even the 90 KB seg-safe floor exceeds the target does the
  agent escalate (drop FPS, or ROI-trade seg — `AGENT_CONSTRAINTS.md §9.2` lever 4). So the sweep directly
  populates the §9.1 `channel_state → payload_budget` map that §9 currently asserts qualitatively.
- **Compose with offline accuracy:** map each `payload_stable(SNR)` back to the best seg-safe knob
  (`ae*/u4/ROI0`) via the offline matrix, and flag the **ROI-escalation** cells where even 90 KB won't fit.
- One-line note back into `AGENT_CONSTRAINTS.md §9` once measured.

## 🚦 Guardrails
0a. **⚠️ MEASURE THE KNEE ON THE UPLINK-ONLY PIPELINE, NOT THE CLOSED-LOOP DOWNLINK RUNNER.** (Learned
   2026-08-03: a sweep used `downlink_latency_fps/run_oai_default106_ttracer_10fps.sh` — the closed-loop
   front that WAITS for a returned result (`--result-timeout`, `--camera-result-port`). That self-throttles
   to ~2 app fps → offers ~17 Mbps → 1 MB "delivers 99%" at 19.5 dB, directly contradicting the uplink-only
   reality (1 MB open-loop at 10 fps = ~84 Mbps ≫ 38 Mbps capacity → 47.7 MiB backlog, 6.4 s, 15% delivery,
   see `uplink_only_spatial_map_pipeline/results/presentation_sinr_uplink_only/`). Nominal 10 fps with a
   ~1 MB tensor would be ~84 Mbps; the live CARLA front actually offered ~50–68 Mbps after sensor-prep
   limits, and that was still above the served UL capacity below clear.) The RL agent controls the
   **open-loop uplink feature stream at target FPS** (car→features→edge, NO downlink return), so the
   `payload_stable(SNR)` knee MUST be measured there. Use
   **`uplink_only_spatial_map_pipeline/run_track2_sinr_uplink_only_ladder.sh`** swept across payload points
   (u8≈1 MB, u4≈400 KB, AE-32≈90 KB), `POLICIES="sinr"`. A knee measured on the closed-loop runner is invalid
   for the agent — its "capacity" is really the self-throttled offered rate, not channel capacity. (The
   closed-loop runner is fine for a *separate* end-to-end-RTT-with-result measurement — just not this knee.)
   Sanity check every rung: log achieved **app fps** and **offered Mbps**; if app fps ≪ target FPS, the run
   self-throttled and the delivery number is meaningless.
0. **⚠️ RUNS ARE NOT CONCURRENCY-SAFE — SERIALIZE, NO `&`, NO BACKGROUND POLLER.** (Learned the hard way on
   2026-08-03: a session background-chained runs and a "if results exist, start next" poller mis-fired,
   launching ~7 overlapping runs → 2 gNBs on different AWGN configs + 3 UE configs alive at once, all
   corrupt.) Each single-run inner runner (`run_track1_oai_default106_ttracer_10fps.sh`) does a **system-wide
   `sudo pkill -x nr-softmodem / nr-uesoftmodem` at startup**, so any new run **kills the softmodems of the
   run already in flight**; only one gNB can hold rfsim port 4043 / T-tracer 2021; and coexisting AWGN
   configs make the per-run SNR (the experiment's independent variable) meaningless. Rules:
   - Call `uplink_only_spatial_map_pipeline/run_track2_sinr_uplink_only_ladder.sh` **foreground, once per
     payload point, and WAIT for it to return** — it
     already loops the profiles sequentially and tears down between them. Do NOT wrap runs in `&`.
   - Between runs, assert the stack is clean: `pgrep -x nr-softmodem; pgrep -x nr-uesoftmodem` must print
     nothing before starting the next.
   - Discard and re-run any cell produced while an overlap was live. One serial 4-profile payload ladder is
     roughly 25–30 min; the current 3-payload × 4-profile grid is roughly 75–90 min serial — trustworthy
     beats fast.
1. **Confirm the SNR ladder actually moves MCS/BLER** before trusting any latency number. The config sweep
   (`oai_config_sweep`) found TDD/5QI/PRB levers were near-no-ops in clean single-UE channel; the AWGN runs
   (`oai_layer_latency/AWGN_HOLD_MCS_RESULTS`) did create real BLER/retx. Print gNB PUSCH SNR + median MCS
   per rung; if two rungs give the same MCS/BLER, the channel knob is not biting — fix the dB values, don't
   report a flat surface as if it were physics.
2. **MCS policy = `sinr` (SUPERSEDES the old "use vanilla" note).** The known uplink bottleneck was the
   QPSK MCS cap → RLC queue-wait (`oai_layer_latency`, `gNB_scheduler_ulsch.c`); the reactive BLER-OLLA also
   collapses UL MCS to ~7 on the clean rung via the sparse-window artifact (`oai_mcs_policy_track2`). Both
   are fixed by the SINR-driven policy, which is why the sweep pins `POLICIES="sinr"`. It follows injected
   SNR (clean→28 … strong→9), ~0 retx, higher/stable delivery
   (`results/awgn106_ladder_track2_sinr_awgn_ladder_20260803.md`). Do NOT run vanilla/hold for the surface —
   they would report an MCS artifact on the good-SNR rung, not the channel. (Keep one vanilla run per rung
   only if a policy-comparison appendix is wanted.)
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

## 🏁 Runbook — how a fresh session kicks this off cleanly
Execution chain that already exists for the official surface:
`uplink_only_spatial_map_pipeline/run_track2_sinr_uplink_only_ladder.sh` (clear/AWGN loop, fixed 106PRB +
`MCS_POLICY=sinr`, knobs overridable) → `uplink_only_spatial_map_pipeline/run_track1_oai_default106_ttracer_10fps.sh`
(brings up the full stack + runs the uplink-only CARLA front + T-tracer, one knob/payload per run).

**Step 0 — preflight (do every session start):** `docker ps` + `/proc/loadavg` + GPU (reuse a running
stack, never kill another user's); confirm `net.core.rmem_max=8388608` (guardrail 6); confirm the UE tunnel
IP (`.2` vs `.3`, guardrail 8); do NOT export `PYTHONPATH` (guardrail 5); do NOT edit `T_messages.txt`
mid-sweep (guardrail 4).

**Step 1 — calibrate the SNR ladder FIRST (guardrail 1, gating).** One `sinr` run per profile, no payload
sweep yet, just to read gNB PUSCH SNR + median MCS per rung and confirm each rung is distinct:
```
RUNS="clear_sinr mild_sinr mid15_sinr strong_sinr" FRONT_DURATION_S=30 \
  bash uplink_only_spatial_map_pipeline/run_track2_sinr_uplink_only_ladder.sh
python3 uplink_only_spatial_map_pipeline/summarize_track2_sinr_uplink_only_ladder.py \
  --base-batch <printed_id>
```
If two rungs collapse to the same SNR/MCS, tune `noise_power_dB`/`ploss_dB` in the `ue.awgn_<profile>.conf`
until the ladder spans ~50→~5 dB. Don't run the full grid until the ladder is confirmed to move.

**Step 2 — official knob-driven payload × SNR.** Run the knob sweep on the **uplink-only** ladder
`uplink_only_spatial_map_pipeline/run_track2_sinr_uplink_only_ladder.sh` (open-loop, no result return). The
old closed-loop downlink chain is invalid for `payload_stable(SNR)` because it self-throttles while waiting
for returned detections; keep it only for a separate end-to-end-RTT-with-result appendix. Here, payload comes
from the knobs (large end only, ~90 KB→1 MB), and each point yields a real deployed payload. Per payload
point, set the knob env and loop profiles, e.g.:
```
# ~1 MB (noae/u8):        QUANTIZATION_MODE=per_channel_uint8 ROI_THRESHOLD=0.0
# ~400 KB (noae/u4):      QUANTIZATION_MODE=per_channel_uint4 ROI_THRESHOLD=0.0
# ~90 KB (seg-safe floor):add the AE-32 bottleneck env used by the knob-matrix loopback runners
#   (grep `rl_agent`/loopback runners for the AE var; not in COMMON_ENV yet) + u4 + ROI0
QUANTIZATION_MODE=per_channel_uint4 ROI_THRESHOLD=0.0 \
  RUNS="clear_sinr mild_sinr mid15_sinr strong_sinr" \
  bash uplink_only_spatial_map_pipeline/run_track2_sinr_uplink_only_ladder.sh
```
(The knob env is now overridable — 2026-08-03 edit made `QUANTIZATION_MODE`/`ROI_THRESHOLD`/`ENTROPY_CODER`/
`ZSTD_LEVEL` `${VAR:-default}` in `run_awgn_106prb_policies.sh`, defaults unchanged.)

**Step 3 — MODE A (the clean transport surface; needs ONE new script): shaped-burst payload × SNR.** This
is the decoupled surface (full 10 KB→1 MB, no accuracy confound). It needs a small orchestrator that does
NOT exist yet — **build + smoke-test it interactively before the overnight run, do not blind-run it:**
- Create `channel_condition_sweep/run_oai_default106_shaped.sh` by adapting
  `downlink_latency_fps/run_oai_default106_ttracer_10fps.sh`: keep the stack bring-up + T-tracer capture +
  MAC-stats recording exactly; **replace the CARLA-front invocation** with a loop over the payload ladder
  running `oai_layer_latency/carla_shaped_udp_burst_sender.py --frame-bytes <N> --fps 10 --frames ~600
  --remote-host <UE tunnel IP> --remote-port 5001 --log-csv <per-point csv>` (chunk-bytes ≤ 65507; do the
  burst/packetization check, guardrail 3). Pin `MCS_POLICY=sinr`, `UE_PRB=106`, `RFSIM_CHANMOD=1`.
- Wrap it in a profile×payload loop (reuse the profile→config mapping from `run_awgn_106prb_policies.sh`).
- Ladder: `FRAME_BYTES ∈ {10,25,50,90,150,300,600,1046}·1024`, `PROFILES` = the calibrated rungs.

**Step 4 — analyze & compose.** Per run: `oai_layer_latency/analyze_uplink_layer_latency.py`,
`scripts/parse_oai_gnb_mac_stats.py`, `analyze_mcs_decision_trace.py`. Build the `latency(payload,SNR)` +
`delivery(payload,SNR)` surfaces, then compose with the offline knob-matrix accuracy + the staleness model
(§ Sweep design 3) into `CHANNEL_SWEEP_RESULTS.md`.

**Bottom line:** Steps 0–2 are runnable tonight with zero new code (knob-driven surface + accuracy). Step 3
(the clean shaped surface) needs one orchestrator built and smoke-tested first — that's the only gap.

## Reuse / outputs / merge
- **Reuse:** `uplink_only_spatial_map_pipeline/run_track2_sinr_uplink_only_ladder.sh` (106PRB + RFSIM_CHANMOD +
  `SCENESENSE_MCS_POLICY=sinr`), the 106PRB `.awgn*.conf` configs, `channelmod_rfsimu.conf`,
  `oai_layer_latency/` analyzers (layer latency,
  MCS, BLER, MAC stats) + `carla_shaped_udp_burst_sender.py`, and the offline knob matrix for
  payload↔knob↔accuracy. (The 273PRB runner is NOT used — 106PRB is the locked config.)
- **Outputs →** `channel_condition_sweep/`: `CHANNEL_SWEEP_RESULTS.md` (surfaces + composed policy + the
  ROI-escalation region), raw per-run CSVs, heatmap plots, and the one-line `channel_state` note for
  `AGENT_CONSTRAINTS.md §9`.
- **Merge with the pending uplink-only-over-OAI run** — same stack; do the uplink-only validation of the
  seg-aware ~90 KB knob (does it hold ≥ target delivery over OAI?) as the first cell of this grid.
