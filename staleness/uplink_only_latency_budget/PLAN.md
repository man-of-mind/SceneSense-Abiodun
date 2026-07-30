# Uplink-only latency-budget staleness analysis — pick-up plan (auto mode)

**Created:** 2026-07-29. **Owner analysis:** redo the staleness / latency-budget → RL-agent-constraint analysis for the
**uplink-only Track-1 architecture**, using the current loopback findings.

> **This feeds the RL agent's constraints. A silent conceptual error here corrupts the agent. Read the GUARDRAILS
> section before touching numbers, and validate against them before writing any finding. Do NOT rescue broken data
> with "the relative trend still holds."**

---

## 0. What changed vs the original staleness analysis (why we redo it)

Original analysis: `../STALENESS_RESULTS.md` + `../../rl_agent/AGENT_CONSTRAINTS.md`. It modeled localization
staleness as **error(v) ≈ √(floor² + (v·L)²)**, floor ≈ 1.1 m, and produced latency-upper-bounds and FPS-lower-bounds
per object speed. Its lag `L` was essentially the **network transport latency** (it used OAI anchors 105 ms AE-128 /
267 ms no-AE), plus the map-staleness term 1/FPS. It also carried a downlink-return term `Y_down` in the extended
`L_total`.

Three corrections from the new `../../uplink_only_spatial_map_pipeline/` (Track-1) work:

1. **Uplink-only architecture → NO downlink detection return.** The edge publishes detections straight to the
   spatial map; the car receives only a tiny async warning. So **drop `Y_down` entirely.** The freshness budget is
   uplink + edge + map only.
2. **Include sensor-preparation time in the lag.** The object keeps moving during sensor prep (RGB wait + radar
   rasterization), front compute, uplink, tail, and map insert — the whole window from **scene capture → map update**.
   The original network-only `L` **undercounted** the true age by omitting sensor prep (a large term). The realistic
   staleness lag is `capture_to_map_update_done`, not just transport.
3. **Incorporate the fast-rasterizer optimization.** Radar rasterization was the dominant sensor-prep cost
   (~139 ms legacy → ~33 ms fast, validated behavior-preserving by a same-frame shadow test, tensor diff 5.96e-08).
   Report the budget with the **fast** rasterizer, and show the legacy→fast improvement as a staleness-error
   reduction for moving objects.

**New lag definition (loopback, uplink-only):**
```
L = Y_sensorprep + Y_front + Y_uplink + Y_tail + Y_mapinsert            (capture → map-update-done)
    (+ 1/FPS map-holds-last-detection staleness on top)
    NO Y_down.
```

---

## 1. The realistic loopback lag — anchor numbers (from Track-1, already measured)

Source: `../../uplink_only_spatial_map_pipeline/TRACK1_IDEAL_LOOPBACK_RESULTS.md`
(ideal 8 MB loopback, no-AE u8, zstd, 200k radar PPS, corrected drivable route, uplink-only no-return).

Fast-rasterizer live profile (p50 / p95), the **capture→map** decomposition:

| Component | p50 | p95 | maps to |
|---|--:|--:|---|
| radar tensor build (fast) | 32.6 ms | 52.8 ms | part of `Y_sensorprep` |
| `capture_to_backbone_input_ms` (sensor prep total) | 53.6 ms | 82.1 ms | `Y_sensorprep` |
| `front_to_edge_ms` (front compute + serialize + uplink, loopback) | 7.8 ms | 13.4 ms | `Y_front + Y_uplink` |
| `tail_ms` | 10.2 ms | 21.4 ms | `Y_tail` |
| `map_queue_ms` | ~8 ms | ~18 ms | `Y_mapinsert` |
| **`backbone_input_to_map_update_done_ms`** (core split→map) | 37.7 ms | 80.2 ms | non-sensor-prep part of L |
| **`capture_to_map_update_done_ms`** (FULL realistic lag) | **93.3 ms** | **136.1 ms** | **L (use this)** |

Legacy (pre-optimization) for the improvement delta: `capture_to_map` 180.7 / 247.5 ms; radar build 139.3 / 180.3 ms.

**Use `capture_to_map_update_done_ms` (fast) ≈ 93 ms p50 as the realistic loopback operating lag `L`.** The original
analysis's ~50 ms network-only anchor was optimistic; ~93 ms (with sensor prep) is the honest uplink-only-loopback age.

> Optional refinement (only if you re-run): run one fresh **uplink-only Track-1 loopback speed-sweep** (fast
> rasterizer) to get per-frame `capture_to_map_update_done_ms` alongside object motion in one dataset. Not required —
> the decomposition above is already measured; see step B for the cheaper post-hoc path.

---

## 2. Method (mirror the original, re-parameterized)

The staleness *physics* (how far an object moves during the lag) is unchanged — it is object kinematics, independent
of the pipeline. So reuse the original approach and just plug in the new `L`.

**Step A — establish L and its decomposition** (from §1; optionally confirm with one fresh loopback run). Produce a
clean "where the freshness age goes" breakdown (sensor prep vs split vs map), fast vs legacy.

**Step B — error(v) at the new L, post-hoc on existing GT-motion captures.**
- GT-motion source: the speed-sweep captures in `../metrics_logs/scenesense_runs/` (39 run dirs; moving-ego + tracked
  targets across walk→~32 mph). These log per-frame predictions + GT with object motion — the same data the original
  used.
- Compute, exactly like the original (`../make_speed_error_report.py`, `../analyze_staleness.py`):
  `error(v) = || pred(t) − GT(t + L + s/FPS) ||`, for the realistic `L` from §1, and cross-check against the closed
  form `√(floor² + (v·L)²)` with floor ≈ 1.1 m. Use average (`s=0.5`) and worst-case (`s=1`) map-hold terms.
- Report error vs object speed at: L=0 (floor), L=93 ms (fast loopback), and L=181 ms (legacy loopback) — so the
  fast-rasterizer staleness benefit is visible.

**Step C — recompute the budgets for uplink-only.**
- Latency upper bound per speed: max `L` to hold error ≤ ε, i.e. `L_max(v,ε) = √(ε² − 1.1²) / v`.
- FPS lower bound per speed (corrected framing: map holds last detection → staleness up to 1/FPS):
  `FPS_min(v,ε) = v / (√(ε²−1.1²) − v·L)`.
- **Master constraint (uplink-only): `v·(L + 1/FPS) ≤ √(ε² − 1.1²)`, with `L = Y_sensorprep + Y_front + Y_uplink +
  Y_tail + Y_mapinsert` and NO `Y_down`.**

**Step D — update the agent-constraint doc.** Write `results/UPLINK_ONLY_AGENT_CONSTRAINTS.md` (do NOT overwrite the
original `../../rl_agent/AGENT_CONSTRAINTS.md`; add a clearly-dated uplink-only section/pointer). State: the deployed
lag is now capture→map (uplink-only), sensor prep is a first-class term, the fast rasterizer cuts it ~106 ms, and
`Y_down` is gone.

---

## 🚦 GUARDRAILS (the traps — most caused real errors before)

> Scene-density analysis is **out of scope here** — it moved to the control-knob-matrix work
> (`../../rl_agent/DENSITY_ADAPTIVE_KNOB_PLAN.md`). Density affects payload only via ROI/entropy compression, so its
> home is knob selection, not staleness. Keep this analysis to latency/FPS staleness only.



1. **`L` = full `capture_to_map_update_done` (fast rasterizer), NOT `backbone_input_to_map`.** The object moves during
   sensor prep too. Using the core 38 ms instead of the full 93 ms would understate staleness. Report both, but the
   **staleness lag is the full capture→map age.**
2. **NO downlink term.** Uplink-only. Do not add `Y_down`. If you copy the old `L_total` formula, delete `Y_down`.
3. **GT convention = actor origin (`origin_x/origin_y`), NOT bbox-center (`world_x/world_y`).** Using bbox-center
   re-introduces the fixed ~1 m offset bug that once inflated live loc to 2–3 m (see `../MODEL_VALIDATION.md` /
   memory `model_live_validation`). Verify `USING_ORIGIN = True` before trusting any loc number.
4. **Loopback only for this analysis.** `L ≈ 93 ms` is the ideal-loopback uplink-only age. Do NOT mix in OAI transport
   here (OAI is higher and belongs to a separate radio study). Label everything "ideal loopback, uplink-only."
5. **Corrected FPS framing:** map holds the last detection between updates → staleness `≈ v·(1/FPS)` at L=0. This is
   NOT single-frame-vs-accumulation (that framing was wrong; per-frame accuracy is FPS-independent).
6. **Floor ≈ 1.1 m is model-limited** — no L/FPS choice beats it. ε < ~1.1 m is infeasible; flag lane-level as a
   model problem, not a latency one. Anchor the model floor to the offline knob-matrix no-AE u8 ≈ 0.95 m, not to any
   loose-matcher live number (~3 m loose-matcher figures are NOT the floor).
7. **Live FPS is CARLA/testbed-bound (~7–10 FPS after the fast rasterizer).** When quoting achievable FPS operating
   points, note the ceiling is CARLA sim/render + sensor prep, not the split path. Use the offered-load replay
   (`../../uplink_only_spatial_map_pipeline/`) evidence that the map path itself sustains 30 FPS.
8. **Validate before findings:** confirm a sane speed-sweep sample (non-empty predictions, origin GT, floor ~1 m at
   v≈0) before computing budgets. Cross-check the direct `GT(t+L)` method against `√(floor²+(v·L)²)` — they should
   agree within noise; if not, stop and diagnose.

---

## Reuse pointers (real paths)
- GT-motion captures: `../metrics_logs/scenesense_runs/*` (speed-sweep).
- Original error/FPS scripts to adapt: `../make_speed_error_report.py`, `../analyze_staleness.py`,
  `../make_fps_latency_matrix.py`, `../make_staleness_report.py`.
- Latency anchors: `../../uplink_only_spatial_map_pipeline/TRACK1_IDEAL_LOOPBACK_RESULTS.md`.
- Original constraints to update-in-parallel (do not overwrite): `../../rl_agent/AGENT_CONSTRAINTS.md`.
- Model floor anchor: `../../rl_agent/PERMODEL_KNOB_MATRIX_ZSTD.md` (no-AE u8 ≈ 0.95 m).

## Output artifacts (write to `results/`)
- `UPLINK_ONLY_STALENESS_RESULTS.md` — the L decomposition (fast vs legacy), error(v) at L∈{0, 93, 181} ms,
  latency-upper / FPS-lower budget tables, master constraint (uplink-only).
- `UPLINK_ONLY_AGENT_CONSTRAINTS.md` — the updated guardrails for the agent (uplink-only L, no Y_down, sensor-prep
  term, fast-rasterizer effect).
- Plots: error-vs-speed at the three L values; FPS×L budget; the freshness-age breakdown (sensor prep vs split vs map).
- Keep raw CSVs + a run log. State clearly what was reused vs re-measured.

## Review rubric (the sign-off pass will check)
- L is the full capture→map age (fast rasterizer); no downlink term; loopback-labeled.
- GT origin convention verified; floor ~1.1 m recovered at v≈0; direct-vs-closed-form agree.
- Sensor prep is a first-class, quantified term; fast-vs-legacy staleness delta reported (~v×0.106 s).
- Budgets/constraints recomputed consistently; conclusions match the numbers; nothing rescued or extrapolated silently.
