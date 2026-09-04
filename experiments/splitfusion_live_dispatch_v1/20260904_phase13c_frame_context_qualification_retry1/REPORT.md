# SplitFusion SFD1 v2 frame-context qualification

Status: `SPLITFUSION_FRAME_CONTEXT_BINDING_QUALIFIED`.

This bounded run qualified dynamic world localization; it is not a latency or FPS study.

| action | family | quantizer | q | frames | max matrix error | max world error (m) |
|---:|---|---|---:|---:|---:|---:|
| 0 | noAE | UINT8 | 0.00 | 8 | 6.67595269e-06 | 8.58050951e-06 |
| 20 | AE128 | UINT8 | 0.50 | 8 | 6.67595269e-06 | 9.05585203e-06 |
| 46 | AE64 | UINT6 | 0.90 | 8 | 6.67595269e-06 | 1.05482551e-05 |
| 71 | AE32 | UINT4 | 0.98 | 8 | 6.67595269e-06 | 8.96757698e-06 |

All 32 messages used SFD1 v2, exact localhost UDP reassembly, edge-resident static calibration, and per-frame transmitted ego pose. The direct frozen reference used each recorded camera matrix only for qualification comparison.

No prediction tensors, inner payloads, SFD1 frames, datagrams, or serialized service payloads are retained.
