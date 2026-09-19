# SplitFusion Phase A1a exact offline quality grid

This package produces per-frame, per-family/quantizer, exact continuous-q
evidence for the two allowlisted Route-B validation scenes.  It never names or
discovers episodes 07/08.  Every output row is labelled
`EXACT_OFFLINE_CARLA_GT_FOR_EXECUTED_FRAME_MODE_Q`; it makes no live, network,
or deployment claim.

The frozen grid is 768 selected frames × 12 family/quantizer modes × 11 exact
wire q values = **101,376 rows**. Canonical row JSON is estimated at 3–8
kB/row (planning range about 290–773 MiB before SQLite index/WAL overhead);
matched-error vectors vary, and actual database
bytes are reported by the store audit.

Raw per-class segmentation counts and localization match/error evidence are
primary.  Q_seg/Q_loc/Q_perc are recomputed by the real protocol-v2 derivation
from an exact caller-pinned canonical `RewardSpecV1`.  The repository supplies
no production calibration, so derived quality is marked
`PROVISIONAL_RECOMPUTABLE_QUALITY_CALIBRATION`. Caller-controlled provenance
can never promote a spec to frozen; only a source-reviewed registered hash can
do that, and the current allowlist is empty. Scalar reward weights are unused
by extraction and are not claimed as selected.

Transport evidence separates the exact frozen-codec inner bytes, exact SFD1-v2
common header and frame-context bytes, complete SFD1 bytes, and deployed
`!IHH` UDP chunk headers. Datagram count is computed by the real chunker over
the complete SFD1 message, not over the inner codec bytes. Registered q anchors
carry their real catalog action ID. Because SFD1 v2 cannot represent an
off-anchor continuous-q action, those rows use an explicit non-dispatchable
uint32 sentinel for byte accounting and make no dynamic-dispatch claim.

Metadata-only preflight (does not read sensor payloads, import/load torch, or
query CUDA):

```bash
python -m rl_agent.splitfusion_hybrid_sac_v1.offline_quality_grid preflight
```

Selection reads only RGB and current-sweep radar points for candidate SI/P40;
it hashes all five source payloads only for the final selected frames:

```bash
python -m rl_agent.splitfusion_hybrid_sac_v1.offline_quality_grid select \
  --output /path/to/selection.json
```

`dry-run` verifies all bound metadata, selected source files, and the exact
caller-pinned spec but does not load a model or query CUDA.  `execute` is
separately guarded by token `SPLITFUSION_EXACT_OFFLINE_GRID_A1A_101376` and is
resumable through the append-only SQLite store.  A duplicate is always refused;
resume skips only already audited exact keys with identical run bindings.
The output bundle copies the exact selection and reward-spec inputs, persists
metadata/runtime/equivalence evidence per attempt, and emits a final hashed
`run_manifest.json` plus `COMPLETE.json` only after exact expected-key
reconciliation.

The runtime gate binds both Python sources and the actual non-Python inputs
that select behavior (recovery/base configs, priors, locks, checkpoint
selection decisions and model checkpoints) by reviewed path and SHA-256. It
rehashes them before Phase-11D preflight and again immediately before model
load. Raw episode truth tables are rehashed before truth loading, while each
selected RGB/radar tensor and each segmentation/ignore mask is resolved and
rehashed again at its inference/scoring use boundary.

The exact scorer currently stages one prediction CSV and segmentation PNG per
row and calls the frozen Phase-6 composition per row. This keeps the scientific
path literal and masks out of durable output, but is likely the dominant I/O
and CPU throughput cost. Phase A1b must benchmark a bounded GPU+scorer smoke,
including rows/second and temporary-write volume, before the 101,376-row run is
authorized; no full-run duration is inferred from implementation alone.
