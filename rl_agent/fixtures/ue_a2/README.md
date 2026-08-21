# UE-A2 fixed technical-smoke fixture

`ue_frame_00156944_inputs.npz` is an immutable sensor-byte fixture for the
single-UE split-inference technical smoke.

- SHA-256: `7fcfad2255c6626b8b87ff3a1c85ec7d32e17c8c2b4eee2875f5f132be423b41`
- Bytes: `2499548`
- RGB: `720x1280x3 uint8`
- Radar tensor: `4x432x768 float32`
- Camera matrix: `4x4 float64`
- Input intrinsics: `3x3 float64`

Source provenance:

`data_collection/experiments/phase2_paired_causal_v1/20260817_181354_pilot/phase2_pilot_benign_001/recipient/retained_inputs/frame_00156944_inputs.npz`

The file is reused only as a real moving-UE RGB/radar input. UE-A2 consumes no
helper/recipient pairing, scenario label, ground truth, or map-sharing logic.
Its use does not reactivate the parked Phase-2 experiment.
