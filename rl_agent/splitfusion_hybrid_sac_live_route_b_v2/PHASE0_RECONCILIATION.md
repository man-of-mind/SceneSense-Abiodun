# Run-4 live-qualification package v2 — Phase 0 reconciliation

**Status:** `PHASE0_RECONCILED_WITH_ONE_PHASE2_RISK`
**Date:** 2026-09-29. **Starting HEAD:** `2a922019ff599a92ec6c1cb00fa2a6fb1c144dac`.
**Scope:** read-only. No CARLA, OAI, RFsim, Docker, softmodem, map server, CUDA
or network service was started. `torch.cuda.is_initialized()` stayed `False`
throughout the symbol checks.

## 1. Supplied authority — all hashes match

| Item | Expected SHA-256 | Observed |
|---|---|---|
| `experiments/splitfusion_hybrid_sac_run4_training_analysis_v1/20260929_advisor_plots/PROVISIONAL_LIVE_CANDIDATE.md` | `d231653a…6413` | match |
| seed-43 `checkpoints/update_010000.checkpoint.json` (file) | `3db78b84…e775` | match |
| same, canonical `checkpoint_sha256` field | `aba44d2d…6672` | match |
| same, `boundary.actor_sha256` | `b61f27a9…bd3` | match |
| seed-43 `SEED_COMPLETE.json` | `5f11e37e…698f` | match |
| seed-43 `CHECKPOINT_MANIFEST.json` | `1ba8131d…d1a7` | match |

Campaign root:
`rl_agent/experiments/splitfusion_hybrid_sac_run4_v2_campaign/20260928_2a92201_three_seed_10000_v1`.
`CAMPAIGN_COMPLETE.json` has `all_seeds_complete: true`. Seed 43: `final_update 10000`,
`final_decision_count 40288`, `final_checkpoint_sha256` = the canonical digest above,
`factory_sha256 6137b855…f55f5`. Its `smoke_gate_passed` is `null`; only seed 17 runs the
smoke gate, so this is expected.

Transport/collector binding: the campaign records artifact
`rl_agent/experiments/ue_production_queue_capture_v1/20260929_model_v2b/transport_model_v2.json`
with SHA-256 `9919e5285d454ec742d877ca33af0df30277df82fe3cf665288c1102b6be286c` and
protocol `rl_agent/ue_production_transport_model_v2/EXPLORATORY_CAMPAIGN_PROTOCOL_V1.md` with
SHA-256 `5dc72797ae875d530d267d7b4268e0da0344d395a7603e4e50940bbe19c6ea3d`. Both observed files
match.

## 2. Symbols and constants (verified by import)

- `run4_contract`: `POLICY_FEATURE_ORDER` (21), `REWARD_DEADLINE_NS = 170_000_000`,
  `TRANSMIT_PERIOD_NS = 100_000_000`, `resolve_reward`, `guard_state_for_action`,
  `build_policy_features`, `PreviousOutcomeV1.from_resolution`.
- `transaction_identity.MINIMUM_HOLD_TENSORS = 2`.
- `state_adapter`: `RawUeUlDciGrantCandidateV1`, `RawUeRlcBacklogSampleV1`,
  `select_prior_new_data_ul_mcs`, `select_pre_action_rlc_backlog`.
- `modeled_smoke_orchestrator`: `read_checkpoint`, `checkpoint_from_bytes`,
  `ModeledSmokeOrchestratorV1.restore`; `smoke_runner.build_harness`.
- `models.run4_model_config`; `ConditionalHybridActor.deterministic_execution` already
  implements argmax mode, selected conditional mean, per-mode support mapping and
  `action_contract.round_half_up_q_e4`.
- `dynamic_execution_contract.load_dynamic_execution_contract()` loads (72 anchors,
  q anchors `(0, 3000, 5000, 7000, 9000, 9800)`); `resolve_q_e4(3, 4321)` returns
  `UNMEASURED_OFF_ANCHOR` with `action_id=None`.
- `reward_ticket_controller.B_REWARD_DEADLINE_NS = 200_000_000`. It is not reused.

Training-time state binding, reproduced by the live package rather than invented:

| Quantity | Training value | Source |
|---|---|---|
| Freshness, all four slots | 100 ms | `collector_v1._RealStateProvider` ← `C2.MAX_CAUSAL_AGE_NS` |
| Backlog scale | `log1p(50_000_000) = 17.72753358339242` | `collector_v1.BACKLOG_LOG1P_SCALE` |
| Camera-SI centre/scale | population mean/std of FIT catalogue camera SI | `collector_v1.__init__` |
| Clock domain | `CLOCK_MONOTONIC_RAW` | `collector_v1.CLOCK_DOMAIN` |
| Actor q support | `MODELED_SMOKE_SUPPORT.mode_q_e4_bounds` (mode 0 `[8812, 9800]` … mode 11 `[0, 9791]`) | `modeled_smoke_support` |

`build_policy_features` does not clip. The live package must therefore apply the registered
`min(1, ·)` form (`contract_v2.backlog_scaled`) only after a raw-backlog support check, so it
is bit-identical to training for every in-support value.

## 3. Restoration path

Restoration replays the event-sourced ledger from genesis through the registered factories
(`build_harness(artifact, 43)` → `read_checkpoint` → `ModeledSmokeOrchestratorV1.restore`),
retrains all 10,000 updates deterministically, and then compares the full boundary
fingerprint, including `actor_sha256`. It requires `torch.get_num_threads() == 4`. The
original seed-43 run took about 41 minutes on this host (checkpoint mtimes 21:24→22:05).
The restoration therefore costs a comparable CPU-only run. It is not a new training
decision: its result must be bit-identical.

## 4. Seams inspected

- **Live SI/P40:** `splitfusion_hybrid_sac_v1/scene_descriptors.py` and
  `splitfusion_scene_descriptor_live_v1/live_ab.py` exist.
- **Anchor-only runtime assumptions:** `envelope.pack_envelope` carries only a `uint32
  action_id`. `PreloadedSplitUERuntime.prepare` and `PreloadedSplitEdgeRuntime.process` both
  call `registry.resolve(action_id)`, which exists only for the 72 anchors.
  `ProductionSplitCodec` needs `profile.q` (float) and `transport.require_inner_agreement`
  reports `profile.action_id`. None of this is inherited for off-anchor actions; v2 needs its
  own envelope and runtime adapter.
- **UE T-tracer events:** `NRUE_MAC_DCI_GRANT` and `NRUE_MAC_RLC_BUFFER_STATUS` are declared in
  `OAI/openairinterface5g/common/utils/T/T_messages.txt:244,248`, and the UE softmodem is
  launched with `--T_port 2023`.

## 5. Phase-2 risk (recorded now, adjudicated in Phase 2)

Every retained consumer of those two UE events is **post-run**:

1. `common/utils/T/tracer/record` writes `ttracer/ue/ue.raw` during the cell;
2. after teardown, `scripts/ttracer_extract_csv_smoke.sh` replays the raw file through
   `replay` + `csv` into `NRUE_MAC_DCI_GRANT.csv` / `NRUE_MAC_RLC_BUFFER_STATUS.csv`
   (`ue_mcs_backlog_near_capacity_v1/capacity_runner.py:1180`,
   `ue_mcs_backlog_calibration_v1/runner.py:678`);
3. `ue_production_queue_capture_v1/parse.py:load_ue_traces` joins them, converting the
   wall-only `time` field to monotonic with a **median-offset `ClockBridge` built over ≥100
   dual-stamped PDCP/RLC events from the whole finished cell**.

No retained module reads the T port live, follows `ue.raw` while it is written, or builds the
wall→monotonic bridge causally. `UE_STATE_EVIDENCE_AUDIT.md` §"Required additional
instrumentation" also records the explicit UE-runtime pre-enqueue read as "the part that does
not exist today". If Phase 2 confirms this, it is the prompt's registered stop condition.

## 6. Implementation map and per-phase allow-lists

All paths are under `rl_agent/splitfusion_hybrid_sac_live_route_b_v2/` unless stated.

| Phase | Tracked files (explicit staging allow-list) | Untracked evidence |
|---|---|---|
| 0 | `PHASE0_RECONCILIATION.md` | — |
| 1 | `__init__.py`, `frozen_actor_v2.py`, `export_frozen_actor_v2.py`, `test_phase1_frozen_actor_v2.py`, `ACTOR_BINDING_V2.json`, `PHASE1_REPORT.md` | `rl_agent/experiments/splitfusion_hybrid_sac_live_route_b_v2/<run>/` (weights `.pt` is gitignored; its SHA-256 is pinned in `ACTOR_BINDING_V2.json`) |
| 2 | `ue_telemetry_provider_v2.py`, `live_state_v2.py`, `test_phase2_live_state_v2.py`, `PHASE2_REPORT.md` — or `PHASE2_BLOCKER.md` alone if blocked | — |
| 3 | `envelope_v3.py`, `continuous_runtime_v2.py`, `test_phase3_continuous_execution_v2.py`, `PHASE3_REPORT.md` | — |
| 4 | `reward_hold_controller_v2.py`, `test_phase4_reward_hold_v2.py`, `PHASE4_REPORT.md` | — |
| 5 | `readiness_v2.py`, `test_phase5_readiness_v2.py`, `LIVE_QUALIFICATION_READINESS.md`, `LIVE_BINDING_MANIFEST_V2.json`, `live_qualification_300_v2.json` | — |

No file outside this package is edited in any phase. Every pre-existing dirty or untracked
path (124 status entries, 667 hashed files) is user-owned and was hashed before any write.
