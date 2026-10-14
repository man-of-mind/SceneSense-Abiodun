> **Superseded in part by [CORRECTION_01.md](CORRECTION_01.md) and [HANDOFF_02.md](HANDOFF_02.md)** (independent review v1: the prep stack lacked V16's storage accounting V3.v2, and contained refusals skipped plan validation; both fixed and requalified in the `_02` runs). The text below is preserved as written.

# HANDOFF — GT geometry-refusal containment v1

**Status: READY_FOR_INDEPENDENT_OFFLINE_REVIEW** (offline only; not registered, deployed or live-qualified).

## Review targets

1. **`classify.py`:** is any non-envelope failure recognisable? Are the code-identity raise path and the
   message-integrity check sufficient? Note that `message_for` must stay byte-identical to `quality._envelope`'s
   f-string.
2. **`engine.py`:**
   - The rollback invariant set taken before and after `prepare_unbound`.
   - Clearing only this error's sticky failure text after verification.
   - The token-less refused unbound slot and its `bind` / `cancel_unbound` / `close` paths.
   - `traceback.clear_frames`, which releases the borrowed views before the FREEZE mapping closes.
3. **`core.py`:** diagnostics are written in the FREEZE reply path. Check the bounds (128 records, 2 sources), the
   write-error isolation, and that the fd is still open (it is closed by `worker.serve` after `handle` returns).
4. **`worker_entry.py`:** the explicit composition (AVO switch → lifetime repair → containment → V4/V3) replaces the
   `avo_tilt → V7` call chain. Records match V7/avo_tilt schemas plus `geometry_containment_binding.json`.
5. **Semantics decision to confirm:** the refusal reuses the BIND-refusal transport, so the outcome is
   `NO_GT_DEFERRAL_REFUSED_<reason>`, and the history semantics are those of any GT-unavailable fault. Timely gives
   CLEARED/genesis; late gives TIMEOUT. A refused frame is never committed to GT history, as for a hold.

## Evidence (main checkout, `experiments/splitfusion_gt_geometry_containment_v1/`)

`tests_01.log` (14/14), `dependency_tests_01.log` (27/27), `replay_s1_01/` (8/8), `verify_engine_01/` (11/11),
`service_proof_01/` (22/22). Report: `REPORT.md`; commands: `COMMANDS.txt`; hashes: `MANIFEST.json`.

## Minimal future registration changes (NOT done)

A new prospective revision copied from V16 (and V15 for 12 dB) would change only:

- **`gt_gc`:** `splitfusion_gt_geometry_containment_v1.contract.gc_registration(arm, budget, producer)`. This sets the
  trampolines to this package's `service_entry`/`worker_entry`, keeps `avo_geometry`, and adds `geometry_containment`.
- **`exact_gt_sources`:** add this package's non-test sources (`contract.exact_sources`).
- **`launcher_binding`:** GT service/worker roles name these trampolines. The startup verification adds negatives for
  an altered containment source and requires `geometry_containment_binding.json` installed and restored.
- **`prerequisite_binding`:** bind this evidence directory's RESULT/OUTPUT_SHA256 files (`service_proof_01`,
  `verify_engine_01`, `replay_s1_01`) and the test logs.
- **Registration-bound offline assessment:**
  - Treat `NO_GT_DEFERRAL_REFUSED_GT_GEOMETRY_ENVELOPE_*` as a registered GT-unavailable outcome, like V15's queue-full
    refusal. Rescore and replay remain over SCORED decisions only.
  - Report `geometry_refusal_v1/COUNTERS.json`, and require `declined == {}` or explain each entry.
- **Analysis:** the `metrics.py` taxonomy beside the unchanged screen, plus the proposed coverage label (REPORT §6),
  pending Abiodun's decision.
- **Campaign (separately authorised):** four fresh sessions in the retained 14 → 12 → 12 → 14 order, with this repair
  in all four. Not prepared here.

## Not done (by scope)

No registration, deployment, live capture, retraining, cleanup, deletion, archiving, commit or push. S1 stays
INCOMPLETE_ROUTE. The S2–S4 prospective roots of the stopped campaign are untouched; they bind the old sources and
would need re-preparation under a new revision.
