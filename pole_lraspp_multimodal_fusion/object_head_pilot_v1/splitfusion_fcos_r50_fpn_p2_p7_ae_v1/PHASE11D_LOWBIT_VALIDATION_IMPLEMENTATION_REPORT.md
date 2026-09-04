# Phase 11D low-bit validation implementation

`ae_phase11d_lowbit_validation.py` is one inert, family-aware future runner for the fixed 48-setting UINT6/UINT4 catalog: noAE, AE128, AE64 and AE32 × UINT6/UINT4 × q={0,.30,.50,.70,.90,.98}. It has no CLI that can alter that scientific matrix.

Before any future CUDA query it verifies the Phase-11B historical checkpoint/source provenance, the repaired noAE device-placement transition, exact hashes for all four completed same-family UINT8 validation curves, the Phase-11B and Phase-11C report/terminal pairs, dense FP32 q=0 reference, and exact live low-bit execution-source hashes. Historical checkpoint provenance and current execution source identity are recorded separately; filename-only allowances are not used.

The public path is fixed: low-bit encode, mandatory zstd L1, raw-byte `PreloadedLowBitDecoders.receive`, receiver-configured tail device, and frozen tail/p025/scorers. No packet field selects output device. Every setting is compared to its same-family, same-q completed UINT8 result and also reports absolute metrics plus dense-FP32-q0 deltas.

Phase-11C is bound as a campaign configuration decision: L1 is fixed because its aggregate L3/L5 rows were larger and higher host cost. This is not a perception result or RL action; Raspberry Pi/OAI latency confirmation remains pending.

The runner atomically writes and fsyncs a 48-identity run manifest before inference. Each setting JSON is the durable completion record; its scratch predictions are removed only afterwards and cleanup is separately marked. `--resume` validates and reuses only complete exact records, finishes only interrupted cleanup, runs only missing settings, and refuses corrupt records rather than overwriting them. The final report refuses to exist without all 48 valid records.

Classification composes the established Phase-10B helpers: all 12 same-family/same-q UINT8 preservation gates yield `FULL_PRESERVATION`; nine absolute service gates remain independent; the three segmentation requirements retain the previous layer/timestamp when not installable; and `LOCALIZATION_PRIORITY` uses the corrected primary 0–30 m AVO person recall. q=.90/.98 are forced `EMERGENCY_ONLY`; valid object-priority failures are `EMERGENCY_ONLY`; integrity failures are `INVALID`; `STATE_INFEASIBLE` remains runtime resource/network-only and is never assigned by offline degradation. No valid profile is action-deleted.
