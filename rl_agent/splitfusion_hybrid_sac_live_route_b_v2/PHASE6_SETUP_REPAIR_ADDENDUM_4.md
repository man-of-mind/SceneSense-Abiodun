# Phase 6 setup-repair addendum 4: ready-record contract

**Status:** `REGISTERED_BEFORE_ANY_NEW_PHASE6_EVIDENCE`, 2026-09-30. Base commit `c5a6d04`.
The machine-readable record is `phase6_setup_repair_addendum_4.json`.

**This authorizes one metadata-only repair of the Phase-6 edge ready record. The scientific protocol is
unchanged.** Nothing else changes:
- model execution, reward, state, the 170-ms timeout and the fallback;
- transport, and the map and reward protocols;
- quality evaluation;
- P0–P8 and the verdict;
- Option-C reporting.

Addendum 2 still sets the claim scope, `SYSTEMS_INTEGRATION_QUALIFICATION_ONLY`. Addendum 3 still sets
the no-build launch of the admitted image
`sha256:2be62d533b8077ceecab5455d5377f2f952b6a50d43ff8c04dc89ce18027d6ba`.

## Cause

Startup qualification `20260930T003718Z` failed with `direct edge ready record identity/endpoint drift`.
The producer (`phase6_edge_runtime_v2.py`) omitted three fields that the consumer requires. The consumer
is the ready predicate in `adapter_direct_v1.start_direct_live_edge`, and it is unchanged.

## Repair

The ready payload is now built by one pure function, `phase6_edge_runtime_v2.ready_document`. The single
live write site uses it. It adds exactly these three fields:

| Field | Value | Meaning |
|---|---|---|
| `dense_label_map_on_radio` | `false` | The predicted dense mask stays at the edge for the asynchronous evaluator. It is not returned to the UE. |
| `object_records_on_radio` | `false` | Object records travel over the direct edge-to-map path. Only compact quality/terminal feedback travels edge-to-UE. |
| `evaluation_evidence_dir` | `str(evidence_dir)` | The existing container-mounted evidence directory the evaluator uses. `evidence_dir` is the already loaded, schema-validated `config["evidence_dir"]`: it is not hardcoded and not taken from the ignored legacy `--edge-segmentation-evidence-dir` argument. |

## Recurrence guard

`test_phase6_ready_contract_v2`:

- **Consumer parity.** It runs the unchanged adapter predicate against the real produced document, which
  passes, and against 18 violations, all refused. The violations cover every condition the predicate
  checks: schema, architecture, `tail_device`, both `false` flags (absent, null, truthy, `0`), map host
  and port, and the evidence path (absent, null, a different path). The predicate is neither copied nor
  weakened.
- **Producer content.** It checks that the identity fields and the quality-spec digest are present, that
  the map and UE endpoints are distinct, and that the record carries no scientific result or
  measurement.
- **Bounded diff.** An AST comparison against `c5a6d04` shows the runtime changed only by adding the
  builder and replacing the payload at the write site.
- **Frozen files.** The scientific files, the adapter, the shared launcher, both compose files and the
  Dockerfile are byte-identical to `c5a6d04`.

## Preserved failed attempts

Both are bound by file hash in the JSON.

| Attempt | Outcome |
|---|---|
| `20260930T000922Z_phase6_live_qualification_option_c` | setup failure: the edge image rebuilt, zero frames |
| `20260930T003718Z_phase6_edge_startup_qualification` | ready-record contract failure, zero frames |

## Sequence

1. Offline verification.
2. One bounded startup qualification. It requires:
   - the image ID before launch, after creation, after readiness and after teardown;
   - `cuda:0`;
   - every ready field reconciled, including the exact evidence path;
   - zero builds or pulls and zero frames;
   - a cold host afterwards.
3. Only if step 2 passes and the host is independently cold: one 300-frame FAVORABLE_STABLE Option-C
   qualification.
4. No retry.
