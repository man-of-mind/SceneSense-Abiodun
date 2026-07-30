# Run log / artifact index — uplink-only latency-budget staleness analysis

Executed 2026-07-29 23:45 → 2026-07-30 00:15. Plan: `../PLAN.md`.
Env: venv `/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python`,
`MPLCONFIGDIR=/tmp/matplotlib-cache`, `QT_QPA_PLATFORM=offscreen`.

## What ran, in order

| # | Step | Command | Outcome |
|--:|---|---|---|
| 1 | A — `L` decomposition | `analyze_L_decomposition.py` | Reproduced the Track-1 anchors exactly from per-frame CSVs (93.3/136.1 fast, 180.7/247.5 legacy). Per-frame additivity residual **0.000 ms**. |
| 2 | B+C — validation gate, error(v), FPS, budgets | `analyze_uplink_only_staleness.py` | Gate **PASSED**: 829 obs / 6 sweep runs, `USING_ORIGIN=True` (0 missing), floor 1.162 m at v<1 mph, per-obs direct-vs-closed-form mean −0.022 m / median 0.000 m. All tables + 4 plots written. |
| 3 | Fresh run smoke test | `run_fresh_uplink_only_speedsweep.sh` (REGIME_SPEC=smoke, 15/20 frames) | **First attempt failed** (`UDPMessageSocket ... unexpected keyword 'remote_host'`) — root-caused to the exported `PYTHONPATH`, fixed, re-ran clean. |
| 4 | Fresh run, full | `run_fresh_uplink_only_speedsweep.sh` | 6/6 conditions `rc=0`. 3 × `L_*` (201 map rows each), 3 × `ACC_*` (GT 27201/27201/19201 rows, preds 1490/1561/1364). ~12 min. |
| 5 | Fold in fresh run | `analyze_fresh_run.py` | `L` conditions **pass** (570 frames, p50 67.5 ms). `ACC_*` accuracy conditions **FAILED 3 of 4 gate checks** → demoted, not used for headline numbers. |
| 6 | Regenerate with both `L` anchors | `analyze_uplink_only_staleness.py` | Budgets + headroom now reported at 68 ms (best estimate) and 93 ms (conservative anchor). |

### The one blocker hit, and its fix

Exporting `PYTHONPATH=$AB/...:$AB:$AB/rl_agent/feature_ae` (as the task's env notes specify — correct for the
*analysis* scripts) **breaks the CARLA client**. The client bootstraps its own path with:

```python
for _path in (neu_collab, abiodun):
    if _path not in sys.path: sys.path.insert(0, _path)
```

If `abiodun` is already on `PYTHONPATH` it is not re-inserted, so `neu_collab` lands ahead of it and the stale
top-level `carla_split_inference_udp_data_collect.py` (May 19, no `remote_host` kwarg) shadows the current
`abiodun/` copy. The Track-1 runner sets no `PYTHONPATH` for exactly this reason. The fresh-run script now
`unset PYTHONPATH`s with a comment.

## Environment notes

- **Reused the already-running CARLA server** (rpc-port 2000, up 2 days). The script hard-fails rather than
  starting one, and never kills other processes. No other client/OAI processes were running.
- UDP buffers were already at 8 MB (`net.core.rmem_max/wmem_max=8388608`); granted `SO_RCVBUF=16777216`.
- Host load at start ~7–9 with GPU at 90 % (CARLA itself). All 6 conditions still completed with `rc=0`.
- **No fresh OAI run** — out of scope by guardrail 4 (loopback only).

## Artifacts

**Documents**
- `UPLINK_ONLY_STALENESS_RESULTS.md` — the analysis: `L` decomposition, error(v), FPS, budgets, guardrail table, caveats.
- `UPLINK_ONLY_AGENT_CONSTRAINTS.md` — agent-facing constraint spec (supersedes `rl_agent/AGENT_CONSTRAINTS.md` for uplink-only only; that file was **not** modified).

**CSVs**
| file | contents |
|---|---|
| `L_decomposition.csv` | per-stage p50/p95, legacy vs fast, with bucket mapping |
| `L_anchors.csv` | the three `L` anchors + the core split→map non-anchor |
| `fresh_L_by_condition.csv` | fresh `L` per traffic regime (3 × 190 frames) |
| `error_vs_L_by_speed.csv` | baseline error(v) at 15 `L` values |
| `fresh_error_vs_L_by_speed.csv` | fresh error(v) incl. distribution-averaged `E_L[err]` |
| `error_vs_fps.csv` | error vs FPS, `s`∈{0.5,1} × `L`∈{0, 93 ms} |
| `budget_latency_upper.csv` | max `L` per (speed, ε), measured + closed form |
| `budget_fps_lower.csv` | `FPS_min` per (speed, ε) at both `L` anchors |
| `budget_headroom.csv` | `B(ε) − v·L` per (speed, ε) at both `L` anchors |
| `fast_rasterizer_staleness_gain.csv` | measured vs predicted staleness gain from the rasterizer fix |

**Plots** (`plots/`, each `.pdf` + `.png`)
`freshness_age_breakdown` · `error_vs_speed_by_L` · `error_vs_L_by_speed` · `fps_x_L_budget` ·
`feasibility_L_fps` · `fresh_run_L_and_error`

**Logs**
`run_log_staleness.txt` (full gate + all tables) · `run_log_fresh_run.txt` (fresh-run gate) ·
`../fresh_run.log` (launcher) · per-condition `client.log` / `map_server.log` under
`../fresh_run_20260730_000257/`.

**Raw fresh-run data**: `../fresh_run_20260730_000257/` (36 MB) — `map_ingest_metrics.csv`,
`edge_uplink_metrics.csv`, and `front_metrics/streams/*` per condition.

## Scripts (new, all under `..`)

- `analyze_L_decomposition.py` — Step A.
- `analyze_uplink_only_staleness.py` — Steps B+C incl. the validation gate.
- `run_fresh_uplink_only_speedsweep.sh` — the fresh 6-condition run.
- `analyze_fresh_run.py` — fresh-run fold-in incl. its own gate.

No existing top-level script was modified.
