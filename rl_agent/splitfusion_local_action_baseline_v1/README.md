# SplitFusion LOCAL action baseline v1

This package measures the currently unregistered `LOCAL` policy branch without
changing any of the 72 SplitFusion actions.

The baseline has two deliberately separate parts:

1. `compute_baseline.py` measures full local FCOS inference from an already
   prepared normalized seven-channel host tensor through compact p025 object
   serialization. It uses the registered 300-frame fit sample. Sensor
   preparation is outside the boundary, and the desktop RTX 5090 is explicitly
   a vehicle-compute proxy rather than a vehicle-hardware claim.
2. `live_transport.py` regenerates those real object results in memory, then
   sends the compact result exactly once at 10 Hz through four fresh calibrated
   OAI/RFsim lifecycles. `map_sink.py` validates and monotonically installs the
   objects in the external-data-network namespace and returns an authoritative
   ACK. The same host-wide `CLOCK_MONOTONIC_RAW` domain makes one-way uplink,
   install, and ACK decomposition valid.

Dense segmentation is never transported and earns no edge-map credit. Raw
inputs, raw predictions, and payload bodies are not retained. The freshness
budgets are 150, 200, and 250 ms; no 100-ms gate is claimed.

CPU checks:

```bash
python3 -m rl_agent.splitfusion_local_action_baseline_v1.test_compute_baseline
python3 -m rl_agent.splitfusion_local_action_baseline_v1.test_map_sink
python3 -m rl_agent.splitfusion_local_action_baseline_v1.test_live_transport
```

The live command requires a fully cold host and the explicit execution token
recorded in `configs/splitfusion_local_action_baseline_v1.json`.
