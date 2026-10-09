# Run-5 (quality-aware, 22-D) binding into the Phase-6 live path: missing seams

This is an audit only. No seam is implemented here. Line anchors refer to
`rl_agent/splitfusion_hybrid_sac_live_route_b_v2/` at `6199d49`.

Items marked **(verified)** were re-checked directly for this audit. The rest
come from a read-only survey of the `run5-snr-scaffold-v1` branch (head
`55b5e56`).

## Actor identity

- **Code:** branch `run5-snr-scaffold-v1` **(verified)**. It and Phase-6 are
  sibling branches; neither contains the other.
- **Campaign:** commit `8c00a37` (`RUN5_DEEP_CAMPAIGN_REPORT.md`). Seeds 17,
  29 and 43 each ran to update 10,000.
- **Weights:** off Git, in the `run5b-no-qperc-v1` worktree. Path:
  `abiodun_run5_wt/rl_agent/splitfusion_hybrid_sac_run5_v1/campaign_runs/run5_three_seed_10000_v1/seed_*/final_actor_010000/actor_state_dict.pt`
- **Weights SHA-256 by seed:**

  | Seed | SHA-256 |
  |---|---|
  | 17 | `693b1fd3…` |
  | 29 | `b7d9a77a…` |
  | 43 | `f17e812defd7d14204ed9c2a140d776727ecc5cf55e0b2f5a3db466e24d2defe` **(verified)** |

- **No frozen Phase-6 export exists.** The pre-registration registers all
  three actors and selects none.

## State: 22-D vs the Phase-6 21-D

- **Features 0–20 are unchanged:** same names, order, builder, scaling and
  freshness. The reward schema, 170-ms deadline, k_min = 2, the Q_perc
  feedback and the R4FB wire are also unchanged.
- **Feature 21 is new:** `effective_external_ul_snr_proxy_scaled`, defined as
  `(snr_db − 5.5) / 19.0` on [5.5, 24.5] dB **(verified)**.
  - A value outside that range forces fallback; it is never clipped.
  - It counts only if a simulator SNR command was acknowledged before state
    commit. A controller heartbeat naming that same command must also be at
    most 200 ms old at action open.
  - Its timestamps must be `CLOCK_MONOTONIC_RAW`.
  - The SNR session must equal the decision session.
- **No compliant live source exists on the UE today.**
  - The Phase-6 UE telemetry subscribes only to DCI grant, RLC buffer status
    and PDCP **(verified,** `phase6_ue_runtime_v2.py:122-124`**)**.
  - gNB PUSCH SNR is forbidden by the Run-5 contract.
  - The only allowed source is the target-SNR actuator subprocess
    (`phase6_live_child_v2.py:246`). It does not meet the contract: it writes
    its CSV from a separate process, stamps `CLOCK_MONOTONIC` rather than RAW,
    sends no heartbeat, records no clamp flag, and starts only at the first
    capture.

## Missing seams

**B** = binding only. **D** = needs a new live data source.

1. **B** — Bring the Run-5 modules onto the Phase-6 line: `run5_state_contract`,
   `run5_snr_v2`, `run5_models`, `run5_preregistration`.
2. **B** — `frozen_actor_v2.py` needs Run-5 versions of:
   - the pinned candidate and its source checks (:149-187);
   - the fixtures (:247-290), rebuilt as **22-D batch-1** fixtures. The Run-5
     fixtures are batched, and batched inference was diagnosed to flip some
     q values.
   - `binding_document` (:324);
   - the model-config check (:411);
   - the width-21 check (:441);
   - the vector-type check (:469-473);
   - the loader and export path (:476-512).
3. **B** — `export_frozen_actor_v2.py:65-77` replays Run-4's ledger. Replace it
   with a step that copies and hash-verifies the Run-5 weights.
4. **B** — A new sealed `ACTOR_BINDING` manifest.
5. **B** — `LIVE_BINDING_MANIFEST_V2.json` and `readiness_v2.py:102-180`:
   - update the actor section, feature schema, feature order and pins;
   - add pins for the lease policy, SNR scaling, SNR support and SNR
     provider.
6. **B+D** — `live_state_v2.py:79-106`: a 22-D state builder with Run-5's
   guard.
7. **D** — An in-process SNR provider in `phase6_ue_runtime_v2.py`: construct
   it at :519-544 and read it in `build_state` (:162-192).
   - The actuator must acknowledge commands, send a heartbeat every 100-ms
     tick, report clamping, and stamp `CLOCK_MONOTONIC_RAW`.
   - That means either a new IPC from the actuator process or moving the
     actuator into the UE process.
8. **B+D** — Session scoping. The engine opens new sessions
   (`phase6_decision_engine_v2.py:233-238`), but the SNR adapter is fixed to
   one session. It needs either the telemetry-style adoption (`:96`,
   `phase6_ue_runtime_v2.py:118`) or one adapter per session.
9. **B** — Decision engine:
   - the state-builder type (:132-134);
   - `guarded.boundary` must become `guarded.run4_guarded.boundary`
     (:252, :256).
10. **B** — Controller lineage hash (`phase6_live_runner_v2.py:59-67`) must
    include the Run-5 manifest and the SNR provider.
11. **B** — Preflight (`phase6_live_runner_v2.py:114-165`): add checks for the
    SNR provider and actuator, and a raw-clock gate.
12. **D** — Fallback and coverage: missing, stale or clamped SNR now counts
    against the 0.95 coverage gate. Because the actuator starts at the first
    capture, the first decisions will fall back.
13. **B** — Tests: actor, readiness, live-path, live-integration and
    ready-contract.

## The profiler needs no change for Run-5

- The policy identity comes from whichever binding the run configuration
  names.
- The actor span records whatever vector the actor receives, including its
  width.
- The replay check uses the configured loader and method.
- The reward check uses the configured contract and controller modules.

A Run-5 run would need only a new qualification JSON naming the Run-5
binding, loader and modules.

## Decisions for Abiodun

1. **Which seed.** None is pre-registered. Choosing seed 43 would be post hoc.
2. **Whether to bind at all.** The recorded modeled result has Run-5 ≈ Run-4,
   and both below the fixed action (mode 11, q 3000).
3. **SNR is the simulator's commanded target,** not a UE measurement, so any
   Run-5 live result is simulator-only.
4. **Actuator design:** change the pinned subprocess or add an in-process
   controller. This also fixes the clock and the clamp definition.
5. **Lease and lag:** is the 200-ms training lease acceptable live, given the
   one-step SNR lag?
6. **Session re-scoping:** is SNR re-scoping across decision sessions allowed?
7. **Weight custody:** the weights currently sit untracked in another
   branch's worktree.
