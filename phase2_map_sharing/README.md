# Phase-2 map sharing

This package is the recipient-specific SceneSense map-sharing core. It adapts
one source stream from the existing spatial-map snapshot, publishes a
wire-safe `scenesense.map_contribution.v1` update to one named ego, performs
class/kinematic association, rejects stale/out-of-order/cross-recipient data,
and emits constant-velocity closest-approach warnings.

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
evidence. C2 requires the paired ego-only/send-everything/hazard-only test on a
controlled CARLA occlusion through the two-UE OAI RFsim path.

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
