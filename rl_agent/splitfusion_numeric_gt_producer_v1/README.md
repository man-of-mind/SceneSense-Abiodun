# splitfusion_numeric_gt_producer_v1 — numerical GT/AVO producer candidate (offline, additive)

Owner: fresh_claude workstream of `experiments/splitfusion_gt_architecture_coordination_v1/WORK_ASSIGNMENTS.json`.
Status: **offline component, CPU-only, not live-qualified, not a wire protocol, not a scheduler.**
Pinned modules are imported unchanged. Evidence: `experiments/splitfusion_numeric_gt_producer_v1/20261005_v1/`.

## What it computes (readiness path)

| Module | Function | Numerics (unchanged pinned originals) | Not on this path |
|---|---|---|---|
| `source.py` | `NumericSourceProducer.compute` | every `LiveCompleteTargetProducer.build` validator verbatim; one depth decode + one plausibility check; `build_object_rows`; per-box `visibility_from_corners` (v1 exact strided count); training mask + literal collector paint block; static projection rows (literal `audit_visibility` loop with the original sub-functions); pinned `IncrementalConflictState.derive`; `CompleteTargetInputs` with the unchanged validator | image/file reads, 2nd/3rd depth decode, 3× catalogue digest, 3× complete-input digest, per-object full-image `isin`/`==`, per-actor radar homogeneous matrix |
| `target.py` | `numeric_target` | literal `build_contract` per-frame body: `roi_mask`, `source_state`, `*_positive_row`, `paint_scaled_box`, connected components, nearest resize, all post-loop gates | PNG encode+decode, CSV write+parse, 2nd plausibility check, per-box `isin` |
| `avo.py` | `numeric_avo` | `frozen_people_rows`, `score_live_avo_people` on the shared decode; original AVO validators | two `np.save` copies, prescore/scored document canonicalisation + hashing + writes |
| `payload.py` | `build_payload` | legacy-exact CSV string rows (DictWriter/DictReader semantics), bytes-backed read-only masks, semantic digest | — |
| `artifacts.py` | deferred | exact `build_complete_v010` dict (tables, PNG bytes, summary hashes), `restore_prepared_target` (frozen validator), both AVO documents (seven recorded clocks), binding record | — |

History: `compute` advances tracker/conflict state tentatively and returns a `Staged` frame;
`commit` makes it accepted history, `cancel` restores the pre-frame state exactly (snapshot/restore
of the pinned objects, no re-implementation). A second `compute` while one frame is pending is refused.
`history_sha256()` gives the before/after digests the coordinator's transactions expect.

## Exact interface needs (for the coordinator's DTO; nothing here is the executable contract)

1. **Fields I emit** (all present on the payload): capture identity, `sample_id`, `frame` (full manifest row incl.
   exact camera pose and `camera_inverse_matrix_json`), `camera_matrix`, `camera_inverse`, `intrinsics`,
   C-order `segmentation_bytes`/`object_ignore_bytes` (331,776 B each), legacy string tables
   `object_boxes` / `object_ignore_regions` (original order and CSV float semantics), `avo` (full qualified +
   structural lists, threshold, timestamp), `provenance` (schema, scope, completeness, `evidence_sha256`
   {owned_source, population, catalogue, conflicts}, `input_content_sha256`), `history_before_sha256`,
   `history_after_sha256`, `binding_sha256` (startup source pins + package hashes), `semantic_sha256`.
   **Policy identity, plan digest and run binding are not mine**; the coordinator must add them from the ticket.
2. **`provenance.evidence_sha256.owned_source` and `input_content_sha256` are the only costly bindings left on
   readiness.** `owned_source` needs the 100-actor `dynamic_corners` evidence canonicalised (~0.2 MB JSON);
   `input_content_sha256` hashes 5.6 MB of native arrays once. `compute(..., bind_owned_source=False)` defers
   the former; the DTO's `_metadata` currently requires both, so the default keeps them.
3. **`avo_document_sha256` cannot be produced on readiness without rebuilding the full AVO document**
   (two `np.save` copies hashed, ~4.6 MB). I recommend binding `avo` by the semantic digest at readiness and
   `avo_document_sha256` at archive (`artifacts.avo_documents` needs the seven genuine clocks).
4. **Pinned leftovers inside `conflicts.derive`** (`A.verify_sources()` re-hashing 5 source files and one catalogue
   digest per frame, ~0.5 MB of canonical JSON) cannot be removed without editing a pinned module; reported, not changed.
5. **Scorer inputs** (Codex): masks via `payload.segmentation`/`object_ignore` (read-only views), `object_boxes`
   string rows, `avo` mapping; `E.evaluate_prepared` on the deferred prepared target reproduces the saved
   evaluation evidence and Q_perc exactly (checked on all 49 frames).

## Parity evidence

`experiments/splitfusion_numeric_gt_producer_v1/20261005_v1/full49_parity/RESULT.json` —
`ALL49_NUMERIC_PRODUCER_EXACT_WITH_8_UNDEFINED_QUALITY`; 27 exact checks per frame (native inputs/provenance,
source evidence, masks, tables, AVO + rows, legacy target.json, PNG bytes, prepared digest, AVO document and
pre-score digests with the recorded clocks, evaluation evidence, Q_perc, shadow cancel/recompute, history digests).
Unit/oracle tests: `verification/TESTS.log` (13 tests: synthetic fixture vs pinned original, two retained canonical
v3 frames vs retained v010 targets and the full original builder, immutability, staging protocol, fail-closed
symmetry, hoisted-transform exactness, CSV semantics, accounting restoration).

## Commands (CPU-only, `/usr/bin/python3`, NumPy 1.26.4)

```
cd <abiodun>; export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/usr/bin/python3 -m unittest -v rl_agent.splitfusion_numeric_gt_producer_v1.test_numeric_producer
/usr/bin/python3 -m rl_agent.splitfusion_numeric_gt_producer_v1.replay_v11 --output experiments/splitfusion_numeric_gt_producer_v1/<stamp>/full49_parity --frames 49
# Serial timing: coordinator-scheduled only, never concurrently with other performance work
/usr/bin/python3 -m rl_agent.splitfusion_numeric_gt_producer_v1.benchmark --output experiments/splitfusion_numeric_gt_producer_v1/<stamp>/serial_benchmark --frames 49 --passes 2 --baseline live_composition
```

`--frames 2` runs are harness smokes and are labelled `*_NOT_PERFORMANCE`.

## Known limits

- Offline reconstructed frames (sealed fixture limits: vehicle pitch/roll placeholders, archived inverses).
- The 170 ms inclusive deadline is **not** addressed by any result here; no live timing exists for this code.
- `depth_is_plausible` still runs twice on the identical array (static audit literal kept); ~0.4 ms, deliberate.
