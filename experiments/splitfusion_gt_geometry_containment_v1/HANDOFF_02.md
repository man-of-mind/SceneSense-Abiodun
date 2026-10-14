# HANDOFF_02 — GT geometry-refusal containment v1, after correction 01 (supersedes HANDOFF.md)

**Status: READY_FOR_INDEPENDENT_OFFLINE_REVIEW** (offline only; not registered, deployed or live-qualified).
Read `CORRECTION_01.md` first, then `REPORT.md` for the original design and evidence.

## Review targets

1. **`worker_entry.py`:** the prep composition is V16 accounting V3.v2 (capability refusal, session, `AccountingTimer`,
   records) → V8 tilt AVO → V7 FREEZE lifetime → containment → V4/V3. Check that the copied V16 accounting block is
   faithful; it is compared with V16 by `WorkerEntryCompositionTest` and by `service_proof_02`'s identical accounting
   results.
2. **`contract.py`:** the V16 contract with only the trampolines, implementations, seam and `geometry_containment`
   changed. `exact_sources` is a superset of V16's.
3. **`engine.py` `bind` for a refused frame:**
   - `_validate_plan` must stay verbatim with `DeferredProducerOwner.bind_plan`'s `try` block.
   - The identity/anchor step must stay the existing `anchor_provenance`.
   - `_validate_admission` covers the payload-independent `ready._metadata` checks plus the P3-aware
     live-context/source-camera pose agreement on the retained source camera.
   - Failure semantics mirror the inherited `bind`.
4. **Unchanged since review v1 (hash-checked):** `classify.py`, the rollback invariant and FREEZE handling, `core.py`
   diagnostics, `fixtures.py`, `replay_s1.py`.

## Evidence (main checkout, this directory)

- **Current:**
  - `tests_02.log` (20/20, including the review's three counterexamples and a 17-mutation differential);
  - `dependency_tests_02.log` (86/86);
  - `review_rerun_01/` (the review's unchanged probes: PROBES_PASS 5/5);
  - `verify_engine_02/` (12/12, accounting session active);
  - `service_proof_02/` (27/27, the V16 stack as control, accounting active and exact);
  - `replay_s1_01/` (8/8, sources unchanged).
- **Superseded but preserved:** `*_01` engine and service runs, `tests_01.log`, `MANIFEST.json`.
- **Hashes:** `MANIFEST_02.json`. **Commands:** `COMMANDS.txt`.

## Minimal future registration changes (NOT done)

A new prospective revision copied from V16 (and V15 for 12 dB) would change only:

- **`gt_gc`:** `splitfusion_gt_geometry_containment_v1.contract.gc_registration(arm, budget, producer)`. This is V16's
  registration, including the unchanged `storage_accounting`, with this package's `service_entry`/`worker_entry` and
  the added `geometry_containment` section.
- **`exact_gt_sources`:** `contract.exact_sources(root)`, which is V16's set plus this package's non-test sources.
- **`launcher_binding`:** GT service/worker roles name these trampolines.
  - The startup verification keeps V16's accounting negatives and capability refusal, and adds a negative for an
    altered containment source.
  - It requires `storage_accounting_binding.json`, `avo_tilt_binding.json`, `prep_lifetime_binding.json` and
    `geometry_containment_binding.json`, all installed and restored.
- **`prerequisite_binding`:** this directory's `_02` RESULT files, `review_rerun_01`, `replay_s1_01` and the `_02` test
  logs.
- **Registration-bound offline assessment:**
  - `NO_GT_DEFERRAL_REFUSED_GT_GEOMETRY_ENVELOPE_*` is a registered GT-unavailable outcome, like V15's queue-full
    refusal.
  - **Actor replay covers ALL policy decisions, including GT-unavailable ones; exact rescoring covers SCORED decisions
    only.**
  - Report `gt_gc_trace/geometry_refusal_v1/COUNTERS.json` and the engine's `geometry_containment` counters, with
    `declined == {}` or an explanation for each entry.
  - Keep V16's accounting-timing checks: 0 dropped/errors/foreign, and per-caller counts consistent with the
    scored/refused decisions. A refused FREEZE adds exactly one `compute_unbound` call.
- **Analysis:** the `metrics.py` taxonomy beside the unchanged 80% screen. The coverage label in REPORT §6 is a
  proposal awaiting Abiodun's decision.
- **Campaign (separately authorised):** four fresh sessions in the retained 14 → 12 → 12 → 14 order, with this repair
  in all four. Not prepared here.

## Not done (by scope)

No registration, deployment, live capture, retraining, cleanup, deletion, archiving, commit or push. S1 stays
INCOMPLETE_ROUTE. The S2–S4 prospective roots of the stopped campaign are untouched; they bind the old sources and
would need re-preparation under a new revision.
