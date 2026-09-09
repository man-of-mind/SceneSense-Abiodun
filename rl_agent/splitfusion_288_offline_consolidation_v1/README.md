# splitfusion_288_offline_consolidation_v1

Offline, evidence-preserving consolidation of the completed SplitFusion 288-cell
live CARLA/OAI campaign (`experiments/splitfusion_288_live_campaign_v1/20260907_full_288_retry8`,
terminal `SPLITFUSION_288_CELL_LIVE_CARLA_OAI_CAMPAIGN_COMPLETE`).

Scope: **Steps 1-4 only** - verify the committed and referenced evidence, produce
one reconciled 288-cell table, aggregate all 72 actions across the four network
profiles, and separate the latency / delivery / payload / registered-accuracy
components needed for later RL formulation.

**Not performed here** (Steps 5-6, a separate reviewer): action pruning, Pareto
decisions, feasibility masking, reward design, RL-policy design, training.

## Run

```
python3 rl_agent/splitfusion_288_offline_consolidation_v1/consolidate_288.py \
  --out experiments/splitfusion_288_offline_rl_dataset_v1/<stamp> \
  --evidence-commit <sha> --starting-head <sha>
```

The output directory is **create-only**; the analyzer refuses to overwrite. It is
read-only with respect to all campaign evidence and launches no CARLA server,
Docker container, OAI gNB/UE, RFsim, CUDA context or model inference. It reads no
training/holdout/validation/test imagery, rescores nothing and tunes no threshold.

## Fail-closed verification

The analyzer raises `EvidenceError` and stops rather than repairing evidence if
any binding or identity cannot be proven: the completion terminal and its four
declared digests, the manifest/ledger/continuation-binding hashes, the
independently **re-derived** `cell_mapping_sha256`, the 72x4 cell-ID space, each
cell's terminal / attempt manifest / every registered output hash / structural
acceptance, the full retry8->retry7->retry6->retry5 continuation chain, and the
hash-verified frozen validation provenance for all 72 actions.

## Scientific conventions

- Clock domains are established from the emitting source, never assumed:
  `WALL_HOST` (CLOCK_REALTIME, shared by the UE client, the local edge container
  and the map process), `UE_PERF` (`perf_counter_ns()` in the single UE process,
  not epoch-anchored) and `EDGE_PERF` (edge durations only).
- A stage requiring a UE_PERF instant against an edge WALL_HOST instant is
  reported **unavailable**, never estimated. That applies to the one-way feature
  uplink; a round-trip residual is offered instead, explicitly labelled.
- `front_ms` is the whole UE dispatch span, not the model front
  (`front_timing_ns.front_backbone`). Pre-front sensor preparation is never
  called `front_ms`.
- `queue_wait_ms` encloses the sensor wait and sensor preparation; it is not a
  pure prepared-queue wait. A same-clock residual is offered separately.
- The 100 ms service deadline and the 500 ms ACK timeout are reported
  separately, with both the per-sent and per-installed denominators retained.
- Zero delivery is preserved as a measured outcome, never as missing data.
- Live measured payload bytes and catalog payload estimates are kept in separate,
  labelled columns.
- **No claim of 100 ms service readiness is made.**

Quantiles use the same nearest-rank convention as the campaign runner, so the
recomputed install-AoI median/p95 are cross-checked against the registered
values for all 288 cells.
