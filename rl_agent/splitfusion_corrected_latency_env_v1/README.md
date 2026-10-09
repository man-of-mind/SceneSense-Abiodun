# splitfusion_corrected_latency_env_v1: corrected training-environment candidate

This is a versioned candidate for the original quality-aware Run-4 (21-D) and Run-5 (22-D) actors. It addresses the latency mismatch confirmed in `experiments/splitfusion_training_live_mismatch_v1/20261005_v1/`. Boundary definitions are B's close-out (`experiments/splitfusion_exact_gt_live_v1/closeout_01/`).

The original packages, campaigns and evidence are imported read-only and are not modified.

**Launch is blocked:**
- `CALIBRATED` mode raises `LaunchBlockedError` until an `APPROVAL.json` names the binding, this source and every blocker.
- No runner exists.

## What changed relative to `collector_v1`

| Original composition | Candidate |
|---|---|
| `R_old + 0.513 ns/B × wire + T_v2b + 1 ms`. It omits 7-channel input and reassembly → prediction, and carries the Sept-16 GT tail. | One measured joint stage vector A–H, G, d, c and R per decision, from a chain in the exact action stratum and scene neighbourhood. It covers every stage once: state/actor, input, codec, send, uplink, edge, handoff, GT, worker, return. |
| GT wait is a pooled draw added in series | Composition v2 (`SERVICE_QUEUE_V2`): GT freeze hand-off, FREEZE (preparation), BIND, CONFIRM_SEND and PUBLISH kept as separate boundaries with their own dispatch waits. A serial score channel gives PREDICTION priority 0 over TARGET priority 1; the second request scores; other frames' work carries over between decisions. GT preparation is concurrent and never added to the prediction path. All recorded chains reproduce exactly. |
| Residual independent of action, scene and channel | Whole vectors keep stage correlations. Action enters through the stratum, scene through SI/P40 k-nearest neighbour, channel through the T_v2b delta on E, inside the validated scope only. |
| Undefined-Q scenes dropped; no faults; 1 session | Undefined scenes kept. Timely undefined → excluded EVALUATOR_FAULT and a fresh-session genesis. Late or missing → TIMEOUT at 170 ms + 1 ns. |

Unchanged:
- quality grid values and interpolation
- state features and the training scaling (`live_state_v2.TRAINING_SCALING`)
- `run4_contract.resolve_reward`: Q − 0.25 L/170, timeout −1
- the inclusive 170-ms deadline
- two-tensor holds
- the MCS/SNR channel and backlog models

## Files

| File | What it holds |
|---|---|
| `chains.py` | Chain decomposition and composition, ordering, critical-path accounting |
| `outcome.py` | Live-controller outcome parity |
| `calibration.py` | Binding schema and hash-bound loader, pipeline identity, stratum/scene/uplink sampler, launch gate |
| `scene_catalog.py` | FIT view including undefined-Q scenes |
| `collector.py` | Run-4 and Run-5 collectors, session breaks, checkpoint/replay |
| `bindings/LIVE100_REGRESSION_BINDING.json` | Regression fixture, not a calibration |
| `bindings/PROPOSED_FIXED_ACTION_SCREEN_BINDING.template.json` | Binding to fill from B's sealed screen |
| `CALIBRATION_GAPS.md` | Every blocker |

## Tests

Run from `abiodun/`, CPU only. Do not run during a live window.

```bash
env -u PYTHONPATH CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  rl_agent.splitfusion_corrected_latency_env_v1.test_chains_outcome \
  rl_agent.splitfusion_corrected_latency_env_v1.test_calibration \
  rl_agent.splitfusion_corrected_latency_env_v1.test_collector
```
