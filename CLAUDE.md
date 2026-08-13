# abiodun/ — project state for Claude (repo-tracked; survives ~/.claude cache wipes)

Abiodun Ganiyu (IDCC). Research: a **UE-side, network-aware split-inference RL controller** for RGB+radar
fusion cooperative perception (CARLA 0.10 / Town10HD_Opt → OAI 5G edge). This file is the durable index —
the authoritative detail lives in the docs below (read them before acting on their topic).

## Conventions (do not violate)
- Work only in `abiodun/`; never edit top-level shared scripts (copy into an `abiodun/` subfolder).
- GT convention = **actor origin** offline, and offline knob-matrix numbers are the accuracy anchor (~0.95 m
  loc floor), never a loose-matcher live number.
- **Do NOT export `PYTHONPATH` for a CARLA client** (front/back/loopback) — it shadows `abiodun/` with the
  stale `neu_collab/` copy → `UDPMessageSocket … remote_host`. Analysis/eval scripts that only `import
  carla` are fine.
- T-tracer byte-compares `T_messages.txt` vs the compiled copy → rebuild BOTH softmodems after any edit.
- Don't kill other users' CARLA/OAI; reuse a running server; check `/proc/loadavg` + `docker ps` first.
- Deployed entropy codec = **zstd** (lossless; accuracy codec-invariant).
- Be systematic; validate the pipeline before writing findings; never rescue broken data with "relative
  patterns still hold"; don't lower score gates to force a pass.

## Authoritative docs (source of truth)
- **RL design (LOCKED):** `rl_agent/AGENT_CONSTRAINTS.md` — §9 = state/action/reward synthesis; §8 = the
  density+segmentation finding; §1–§6 = the staleness/latency/FPS bounds.
- **Knob matrix (accuracy↔knob↔payload, transport-invariant):** `rl_agent/PERMODEL_KNOB_MATRIX_ZSTD.md`.
- **Density+seg study (DONE, 9/9 gates):** `rl_agent/density_knob/DENSITY_KNOB_RESULTS.md` — ROI drop
  destroys segmentation; seg-safe knob is `ae32/u4/ROI0 ≈ 90 KB`, density-invariant.
- **Channel sweep (DONE):** `channel_condition_sweep/CHANNEL_SWEEP_RESULTS.md` and `combined_surface.csv` —
  106PRB OAI channelmod AWGN surface used by the table-driven policy environment.
- **OAI MCS policy work:** `oai_mcs_policy_track2/` — SINR-driven UL MCS implemented at
  `OAI/…/gNB_scheduler_ulsch.c:2027`; fixes the clean-channel MCS-7 sparse-window artifact.
- **Uplink-only architecture + staleness redo:** `uplink_only_spatial_map_pipeline/`,
  `staleness/uplink_only_latency_budget/`.
- **Papers:** `SCAN_AI_03_13_26_2.pdf` (single-UE foundation), `V2X_for_AD.pdf` (CoDriving).

## Current status / next (2026-08-12 local / 2026-08-13 UTC)
- Density+seg study, RL design lock, measured loopback latency, and the **channel-condition sweep: DONE**
  (`channel_condition_sweep/CHANNEL_SWEEP_RESULTS.md` + `combined_surface.csv` + `plots/`). Knee: 1 MB→clear
  only, 400 KB→to 15.6 dB, 90 KB seg-safe floor→every rung. `AGENT_CONSTRAINTS §9.1` holds the measured
  channel_state + `payload_budget=capacity/fps` rule.
- The native-10-Hz advisor-rich v5 corpus is accepted. Collection completed 24/24 runs; verification
  `data_collection/experiments/policy_corpus_advisor_rich_v5/20260813_045142_full/verification/20260813_061952`
  is `PASS` on structural controller-corpus gates after excluding only impact run `pcarv5_mixed_va01` (23/24
  retained). Recall is report-only: held-out pedestrian <=12 m is 67.87% and vehicle <=25 m is 67.20%, with
  trajectory-grouped CIs. Accepted-run radar density is 19,404.5/frame; traffic and cleanup are clean.
- Freshness re-score `freshness_rescore/20260813_062203` has no QC exclusions and confirms both slow pedestrian
  and sustained >=10 m/s vehicle regimes, with 54.43% GT-seeded mapped freshness pressure.
- The authoritative reward-v5 pre-RL ladder is
  `rl_agent/policy/experiments/controller_ladder/20260813_063514`. Greedy reward is 0.19655 and MPC is 0.19834
  with identical 91.13% matched-safe rate; they disagree on only 2.54% of finite frames. **Current RL decision:
  NO-GO** for SAC/DQN/PPO under the present SPLIT+SKIP surrogate. LOCAL remains uncalibrated; if it is added,
  rerun the simple ladder before reconsidering RL. See `data_collection/EVALUATION_CONTRACT_DECISION_V5.md`.

> The `~/.claude` memory cache was wiped by a retention cleanup on 2026-08-03 (harness, not us). This
> repo-tracked file exists so project state is never lost that way again. Keep it updated as work advances.
