# Overnight decision + progress log (RL agent, Month-2 catch-up)
Read this first when back. Goal = hit the **Month-2 Definition of Done** (offline controller harness +
static sweeps scoring policies on logged metrics). Autonomy granted 2026-07-07 — decisions logged here.

## Objective for tonight
1. **[running]** Live static sweep (8 quant×entropy variants, PID 3910456) → payload + front-latency/profile.
2. **[building]** Offline accuracy-vs-compression eval (deterministic, no CARLA) → accuracy-degradation/profile.
3. Join → payload/latency/accuracy **Pareto** (+ best-fixed + lowest-byte-unsafe).
4. **B** offline controller harness → score send-everything / low-byte / best-fixed / heuristic / **LinUCB**.
5. **Month-2 status summary** (done/remaining, [ENG]/[RES]) for slides.

## Decision framework (what I will / won't do unattended)
- Prioritize the DoD core (1–5). Keep everything reversible; only edit under `abiodun/`; no risky installs.
- **Will NOT** launch a full unattended AE training run (D) — build + tiny smoke-test only; full train is
  with-you (early M3). Not required for the DoD.
- If CARLA dies or the sweep stalls → pivot to the offline pieces (no CARLA needed) using existing traces.
- If a sweep variant fails (actor-cleanup over 8 sequential CARLA connects, etc.) → note it, continue with
  the variants that succeeded; the Pareto degrades gracefully.

## Decisions & progress (appended as I go)
- 2026-07-07 21:5x — Sweep launched (PID 3910456, background). AE training deliberately excluded (not DoD;
  needs validated objective). Early data already differentiating profiles: per_tensor_uint8+none ≈ 1418
  KB/frame vs +zlib ≈ 805 KB (3.5× vs uncompressed 2835 KB).
- 2026-07-07 21:5x — **Decision:** the ONLY thing I automate unattended tonight is the deterministic sweep
  aggregation (safe). Built `sweep_analyze.py` (tested on partial data) + `overnight_analyze.sh` (setsid,
  PID 4055263) which waits for the sweep to finish, then writes `analysis/static_sweep_summary.md` +
  `static_sweep_payload.png` and appends here. **Guaranteed morning deliverable = the payload/latency table
  + Pareto over all 8 quant×entropy profiles.**
- **Deferred to when we reconnect (validation-heavy, NOT run blind):** offline accuracy-vs-compression eval,
  the controller harness B (reward/LinUCB), and AE build+train. Rationale: correctness needs a human check;
  running them unattended risks a wrong objective. The sweep data will be ready for them.
- **If the sweep partially fails:** the aggregation degrades gracefully (only completed variants shown).
[2026-07-07 22:28:56] sweep finished -> aggregating
[sweep_analyze] 6 variant(s) aggregated -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/rl_agent/analysis/static_sweep_summary.md
# Static sweep — payload + front-latency by compression profile

Deterministic aggregation of the quant×entropy sweep (loopback). Accuracy-vs-compression is a separate offline eval (deterministic, human-validated). `frames_with_result` shows how often the loopback result returned (low = the ~1MB payload fragments over UDP; why live accuracy is unreliable).

| variant | quant | entropy | payload KB (comp) | KB (uncomp) | compression× | front_ms | frames | w/result |
|---|---|---|---|---|---|---|---|---|
| q_pchan_u4_none | per_channel_uint4 | none | 717.2 | 2835.0 | 3.95 | 26.7 | 300 | 91 |
| q_pchan_u4_zlib | per_channel_uint4 | zlib | 381.4 | 2835.0 | 7.43 | 34.9 | 300 | 300 |
| q_pchan_u8_none | per_channel_uint8 | none | 1426.0 | 2835.0 | 1.99 | 27.6 | 300 | 32 |
| q_pchan_u8_zlib | per_channel_uint8 | zlib | 1025.3 | 2835.0 | 2.77 | 50.5 | 300 | 27 |
| q_ptensor_u8_none | per_tensor_uint8 | none | 1418.2 | 2835.0 | 2.0 | 28.7 | 300 | 24 |
| q_ptensor_u8_zlib | per_tensor_uint8 | zlib | 805.5 | 2835.0 | 3.52 | 46.2 | 300 | 38 |
[2026-07-07 22:28:56] analysis written to rl_agent/analysis/
