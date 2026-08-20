# Phase-2 map sharing

> **PARKED 2026-08-19 — no active execution or launch authority.**
> This package is preserved for later helper-recipient work. The current
> milestone is the single-UE controller in
> `../rl_agent/UE_AGENT_EXECUTION_CHECKLIST.md`. Read
> `PARKED_STATUS_2026-08-19.md` before touching this package.

This package is the recipient-specific SceneSense map-sharing core. It adapts
one source stream from the existing spatial-map snapshot, publishes a
wire-safe `scenesense.map_contribution.v1` update to one named ego, performs
class/kinematic association, rejects stale/out-of-order/cross-recipient data,
and emits constant-velocity closest-approach warnings.

`v1` is a frozen plumbing scaffold, not the live collection/safety schema. The
banked successor design is `PHASE2_PAIRED_CAUSAL_CORPUS_SPEC.md`: schema v2,
causal pre-action state, separate inference-placement/publication decisions,
designed + naturalistic paired suites, and a two-trajectory pilot gate. Do not
launch CARLA/OAI collection from this README.

The v2 offline foundation and derived collector/replay/verifier integration are
now implemented. The reviewed two-trajectory capture
`20260817_181354_pilot` passes the versioned nine-gate verifier. This is a
structural/computability PASS, not C2 performance evidence. The post-pilot
warning definitions, clustered units, calibration grid, and C2 decision rule
are in `WARNING_EVALUATION_DESIGN_FREEZE.md`. They must be satisfied before a
full collection is authorized.

The evaluation-only future-hazard adjudicator is implemented in
`adjudicate_future_hazards.py`. `hazard_adjudication_v2` is the authoritative
pilot output: it uses the matched benign recipient motion as the positive
trajectory's no-yield counterfactual and keeps realized stopping outcomes
separate. `hazard_adjudication_v1` is superseded by the intervention-paradox
fix. The ranked safety/network/compute/physical catalog is
`PHASE2_CONSTRAINT_CATALOG.md`.

The deterministic powered-corpus candidate is
`PHASE2_SUITE_AB_DESIGN.md`. Suite A means designed decision opportunities;
Suite B means naturalistic operation. Its 210 independent groups / 330 world
trajectories are already assigned to calibration, validation, and untouched
test in a hashed manifest, but collection is still unauthorized while scenario
authoring/visual review and the calibration power gate remain open. Rebuild the
offline manifest with:

```bash
/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  -m phase2_map_sharing.design_suite_manifest
```

The compact baseline is **recipient-hazard-only**, not generic saliency: it
uses causal object estimates plus the named recipient's current state to select
objects predicted to enter its safety radius. CARLA IDs and future truth are
forbidden from runtime messages; `evaluation.py` joins a separate truth stream
only when scoring warnings.

The wire codec uses the production `!IHH` chunk header already exercised by
`rl_agent/multiue_oai/endpoint.py`. Application bytes and UDP/IP on-wire bytes
are logged separately.

Run the cheap local contract check with:

```bash
/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  -m phase2_map_sharing.run_local_acceptance
```

The checked-in/latest result is a **synthetic contract validation**, never C2
evidence. C2 requires a reviewed paired-corpus pilot, then the separately
reported local and two-UE OAI RFsim evaluations. The accepted v5 policy corpus
cannot supply synchronized recipient warning lead.

Run the non-launching v2 config/storage preflight with:

```bash
/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  -m phase2_map_sharing.run_pilot_contract_preflight \
  --config phase2_map_sharing/configs/paired_causal_pilot_v1.yaml \
  --disk-root .
```

Its PASS means only that the offline contract and logical disk budget are
consistent. It deliberately reports `live_pilot_authorized=false`.

Resolve and inspect both paired collector commands without launching anything:

```bash
/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  -m data_collection.run_phase2_paired_causal_pilot \
  --validate-config \
  --config data_collection/configs/phase2_paired_causal_pilot_integration_v1.yaml
```

The command must report `live_authorized=false`,
`scenario_geometry_status=reviewed_positive_and_benign_routes`, geometry ID
`town10hd_opt_curbside_legal_opposing_v1`, population mode
`frozen_curbside_pilot_no_ambient`, one external ticker, two frozen exact ego
spawns, and unique helper/recipient UDP stacks. `--launch` on this v1 config
fails before a CARLA client is opened.

For a separately reviewed two-trajectory capture, run the evaluation-only
replay and verifier into new create-only namespaces:

```bash
/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  -m phase2_map_sharing.replay_paired_pilot \
  --batch-root <reviewed_pilot_batch> \
  --config data_collection/configs/phase2_paired_causal_pilot_reviewed_v1.yaml \
  --evaluation-name evaluation_v4

/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  -m phase2_map_sharing.verify_paired_pilot \
  --batch-root <reviewed_pilot_batch> \
  --config data_collection/configs/phase2_paired_causal_pilot_reviewed_v1.yaml \
  --evaluation-name evaluation_v4 \
  --verification-name verification_v4

/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  -m phase2_map_sharing.adjudicate_future_hazards \
  --batch-root <reviewed_pilot_batch> \
  --integration-config data_collection/configs/phase2_paired_causal_pilot_reviewed_v1.yaml \
  --adjudication-config phase2_map_sharing/configs/future_hazard_adjudication_v2.yaml \
  --evaluation-name evaluation_v4 \
  --output-name hazard_adjudication_v2
```

Replay parameters are provisional-for-computability. The verifier stops at the
first of nine failed gates, requires a registered-target capture-to-warning
chain plus exposure/fragmentation diagnostics and hash provenance, and does not
require a positive warning-lead result from two trajectories. Never overwrite
an earlier evaluation or verification namespace.

Historical launch command used for the accepted pilot (do not rerun merely
because it is documented):

```bash
/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  -m data_collection.launch_phase2_paired_causal_pilot \
  --launch-detached | tee /tmp/phase2_pilot_launch.json
```

The printed JSON gives the exact batch root and run log. For any future pilot
version, do not chain replay or verification into this command; inspect the
completion/failure sentinel first.

The renderer-domain gate is complete. Sparse Low/Epic captures demonstrated a
class tradeoff; the follow-up medium/crowded production-collector captures were
structurally matched, but their frozen <=12 m pedestrian component had zero
support. The formal weighted decision therefore remains inconclusive. For all
future primary Phase-2 work the operational setting is explicit CARLA `Epic`
with server flag `-quality-level=Epic`; Low is retained only in the existing
labelled stress artifacts and is not a second powered-corpus stratum. This
choice must not be described as proof that Epic dominates every class or that
M-prime was trained under Epic. CARLA has no quality RPC, so launch and run
manifests must record the operator-declared exact flag.

The adapter is also checked against the existing two-stream recordings:

```bash
/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  -m phase2_map_sharing.run_recorded_snapshot_smoke
```

Artifact `experiments/snapshot_adapter/20260814_222354` passes: 37 snapshots
have both streams active, 26 contributions satisfy the strict 1 s age gate,
and all messages round-trip exactly without runtime actor identity. The 11
stale rejections are retained as a useful freshness diagnostic. These legacy
recordings lack synchronized hazard truth, so warning lead is not scored.
