# Phase 11D low-bit deployment validation

Terminal: `SPLITFUSION_LOWBIT_PHASE11D_VALIDATION_COMPLETE`

Fixed catalog: 4 families × UINT6/UINT4 × six registered q anchors = 48 settings. Each row is one full registered 3,345-frame validation pass through the public low-bit/zstd-1/raw-byte-dispatch/frozen-tail path.

Every row compares to the completed same-family, same-q UINT8 record and reports absolute metrics plus deltas to dense FP32 q=0. Phase-10B's corrected 0–30 m AVO person-recall classification contract is used without changing detection emission at other ranges.

Zstd L1 is fixed campaign configuration, bound to Phase-11C aggregate evidence: L3/L5 were larger and higher host cost. This is not a perception decision or RL action; Raspberry Pi/OAI latency remains pending.

The completed layout-feasibility study is hash-bound as NOT_USEFUL, so the production CURRENT_CELL_MAJOR layout and production codecs remain unchanged.

Durability is per setting: atomically fsync its record, then remove scratch (unless retained explicitly), then atomically write cleanup. Resume reuses only fully valid exact records and refuses invalid ones.

Completed settings: 48/48.
