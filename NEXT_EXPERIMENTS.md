# Next experiments — pick-up plan (written 2026-07-16)

For a FRESH Claude Code session (keep context lean). Start by reading the memory index (MEMORY.md) —
especially [[staleness_fps_results]], [[model_live_validation]], [[coop_fusion_findings]], [[rl_agent_pivot]] —
plus `staleness/STALENESS_RESULTS.md`. These three experiments came out of the 2026-07-16 supervisor meeting.

## Validated setup to reuse (don't re-derive)
- **Sensor/model:** moving **car-height ego** (z=1.55, pitch −4°, fov 120° — matches training), RGB+radar fusion,
  no-AE u8 checkpoint `experiments/ae_integrated_20260710/noae_baseline/checkpoints/mprime_joint_noae/best.pt`,
  loopback (clean, full delivery). Scenario: `staleness/carla_fusion_staleness_scenario.py`.
- **GT convention (CRITICAL):** compare predictions against the actor ORIGIN — GT logs `origin_x/y` (= training's
  `actor.get_location()`), NOT bbox-center `world_x`. All staleness scripts already use `origin_*`.
- **Opportunity-window method:** ego drives among traffic; any vehicle in camera frustum AND ≤25 m matching a
  prediction (gate 2 m, score ≥0.2) is an observation, binned by MEASURED instantaneous speed.
- **High speeds (28–32 mph):** induced by over-speeding NPC traffic — `--npc-speed-difference-pct` negative
  (−45..−88) + `--npc-ignore-lights-pct 100`; still Town10 (in-domain); binned by measured speed.
- **Latency is swept ANALYTICALLY** on live captures: `error(Y) = ‖pred(t) − GT_origin(t+Y)‖` via `gt_at()`
  trajectory interp. Captures are live; the latency axis is post-hoc (no real transport delay injected).
- **Data on disk:** speed-sweep runs (walk→32 mph, 6 runs, run_group `speedsweep_*`) and FPS captures
  (`egofps_*`, `egofpsfast_*`) live under **`staleness/metrics_logs/scenesense_runs/`** (NOT the top-level one).
- **CARLA** on 127.0.0.1:2000 (Town10HD_Opt). venv: `carla_0_10_venv`. PYTHONPATH must include
  `pole_lraspp_multimodal_fusion:.:rl_agent/feature_ae`.

---
## Experiment 1 — Curve road-state plot (quick, existing data)
Context: `staleness/make_roadstate_speed_plots.py` splits the per-speed error(Y) into straight vs intersection
(plots `roadstate_{straight,intersection}_speed.pdf`). Town10 curves came up ~0 because the curve threshold
(`CURVE_DEG_PER_5M = 8.0`) is strict / curves sit inside junctions.
- **Do:** lower `CURVE_DEG_PER_5M` (try 3–5°/5 m) and re-run on existing speed-sweep data; if a "curve" bin gets
  enough samples (≥~15), add a third per-speed plot. If still too sparse → target a curved road segment
  (spawn ego on a known Town10 curve) in a short capture.
- **Output:** `roadstate_curve_speed.pdf` (if feasible) + a note in STALENESS_RESULTS.md Result 1b.

## Experiment 2 — FPS as spatial-map staleness (CORRECTED framing — read this)
> We previously did single-frame vs Kalman accumulation. **That is NOT what the supervisor wants — drop it.**
> The point: the **spatial map** queries the latest detection; between frames it holds a STALE position. At
> update rate FPS, worst case the held position is a full **1/FPS** old (queried just before the next frame).
> Static car → fine; fast car → error ≈ `v × (1/FPS)` **even at ZERO network latency**. FPS is a latency analog.

- **Metric:** at 0 network latency, map-staleness error = `‖pred(t) − GT_origin(t + 1/FPS)‖` (worst case),
  and/or `t + 1/(2·FPS)` (average query time). Bin by measured speed.
- **Sweep:** FPS ∈ {1, 5, 10, 15, 20, 25, 30} × speed 0–32 mph. **Computable POST-HOC from the existing
  speed-sweep captures** — no new runs (detection accuracy is FPS-independent; the FPS effect is pure staleness,
  and `gt_at()` extrapolates the trajectory for the 1/FPS lookahead, which is exactly the v·(1/FPS) displacement).
- **Plot:** loc error (y) vs object speed (x), **one line per FPS** — shows error flat for static, exploding at
  low FPS for fast cars, collapsing toward the model floor (~1.1 m) at high FPS. Worst-case line is the headline;
  optionally add the average-case.
- **Then:** repeat with added network latency Y → error at lag `Y + 1/FPS` (FPS+latency coupling). Overlay the
  real operating Y (loopback ~50, AE-128 ~105, no-AE ~267 ms).
- **Build from:** `staleness/make_speed_error_report.py` / `make_roadstate_speed_plots.py` (same collection loop;
  just set the lookahead lag = 1/FPS instead of a fixed Y, and make FPS the line variable).

## Experiment 3 — Radar/camera FOV-position diagnostic (the parked-car localization puzzle)
Why: loc error was ~2 m even for a nearby/parked car; segmentation looked OK but localization was off. Hypothesis:
accuracy degrades as the target moves from FOV center toward the edge / partial occlusion.
- **Pattern to adapt:** `carla_radar_pedestrian_distance_pps_diagnostic.py` + `radar_pedestrian_diagnostic_runs/`
  (deterministic actor placement, fire radar+camera, sweep a parameter). Reuse that structure.
- **Setup:** place a **fixed target car deterministically**; ego directly behind, **aligned (target centered)**,
  both **static**, at a fixed range (~15 m). Car-height ego, RGB+radar → model, measure loc error for the target.
- **Sweep:** shift the ego (or target) **laterally** — 0, ±2, ±4, ±6, ±8 m (target center → edge of FOV). At each
  offset, average loc error over N static frames; log the target's **pixel x from center** + radar support count.
- **Plot:** loc error vs FOV position (lateral offset / pixel-from-center). Expect error rising toward edges.
- **Downstream:** if center detections are more accurate, the agent can PRIORITIZE center-FOV tensors under
  bandwidth limits → feeds the RL controller. Also sanity-check: is the error a localization-head issue vs a
  radar-support/occlusion issue (log radar_support_count vs FOV position).

---
## Also carry forward
- **RL agent state:** add **object speed** as a state feature — it sets the compression/latency budget needed to
  land features within the accuracy threshold. Note in `rl_agent/REQUIREMENTS_AND_RL_DESIGN.md`.
- **OAI config sweep (paused):** findings in `oai_config_sweep/OAI_CONFIG_FINDINGS.md` — config barely moves
  transport in single-UE; compression is the lever. Gotcha: automated rfsim gNB↔UE restarts are flaky → use a
  full cold-restart per config; extreme UL TDD not achievable (gNB K2 + UE DCI limits). Revisit under multi-UE /
  channel impairment (SIONA-RT).
- **Meeting framing that landed:** cars tracked ≤25 m (median 13 m); high speeds via NPC over-speed; latency swept
  analytically on live captures; road state doesn't change the trend (only a ~0.3 m floor penalty for slow objects
  at intersections).
