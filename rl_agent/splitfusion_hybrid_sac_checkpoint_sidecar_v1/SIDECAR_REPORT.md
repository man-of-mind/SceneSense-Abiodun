# Materialized checkpoint sidecar and selected-actor audit

**Status:** implemented and tested offline (CPU only). Starting HEAD
`f32dbf45a8ba67c47e2d78f58c925891d6310b6c`.

Nothing was launched: no CARLA, OAI, Docker, CUDA, map server, Phase 6 or retraining. No historical
Run-4 checkpoint was replayed.

## A. What the existing Run-4 checkpoints are

The Run-4 files `update_*.checkpoint.json` (`modeled_smoke_orchestrator.ModeledSmokeCheckpointV1`) are
**event-sourced reconstruction checkpoints**.

**What they keep:**
- the full transition ledger;
- the collector checkpoint;
- the preflight report;
- the bindings;
- a `BoundaryFingerprintV1`, holding the SHA-256 of:
  - the actor state;
  - the twin critic state;
  - the actor and critic optimizer states;
  - the decision q-RNG and mode-RNG states;
  - a runner-RNG *position document*;
  - the replay-buffer digests and counters.

**What they lack:** any directly loadable actor, critic or optimizer tensor state. Restoring one means
replaying training from genesis; seed 43 to update 10,000 took 2,429.7 s.

Consequences:

- The seed-43/update-10,000 actor weights exist only because `export_frozen_actor_v2` did that replay
  once and saved the result.
- The audit checks that the update-10,000 checkpoint's top-level fields are exactly the dataclass fields
  plus `schema`, and that every boundary value is a 64-hex digest or an integer.
- No historical checkpoint was modified, and none is claimed to hold weights.

## B. Sidecar for future runs (`sidecar.py`)

A Run-5 checkpoint callback, `SidecarCheckpointCallbackV1(orchestrator, root, chain=event_writer)`, writes
one directory `update_NNNNNN.sidecar/` per emitted boundary. It is published atomically and create-only:

| File | Content |
|---|---|
| `actor.pt` | actor `state_dict` |
| `online_critics.pt` | `critic_1.*`, `critic_2.*` |
| `target_critics.pt` | `target_1.*`, `target_2.*` |
| `actor_optimizer.pt`, `critic_optimizer.pt` | Adam `state_dict`s |
| `generators.pt` | CPU generator states: `decision_q`, `decision_mode`, `replay`, `trainer_target`, `trainer_actor` |
| `SIDECAR_MANIFEST.json` | canonical JSON (see below) |

The manifest records:
- the seed and full seed plan, plus its digest;
- update and decision counts;
- the event-checkpoint SHA-256 and its full boundary;
- feature/action/model schema identity (the frozen-actor binding document plus the action-identity schema
  and event schema id);
- for every artifact: SHA-256, byte size, tree digest, and the full tensor inventory (path, dtype, shape,
  SHA-256);
- deterministic actor fixture outputs, using the 20 registered fixtures.

**Reused, not re-implemented:**
- `checkpoint_io`: its error hierarchy, `_sha256_file` and canonical JSON;
- `modeled_smoke_orchestrator`: `_tensor_sha256` and `_tree_sha256`. These are the boundary hash
  functions, so sidecar and event-checkpoint hashes are directly comparable;
- `frozen_actor_v2`: `fixture_outputs` and `binding_document`;
- `models`: `validate_run4_models`.

No training code was written or changed.

**Guarantees:**
- **At the boundary only.** The writer refuses unless the captured tensors hash to the event checkpoint's
  boundary: actor, critics, both optimizers and both decision RNGs.
- **Atomic, create-only.** Each file is created `xb` and fsynced. The staging directory is fsynced,
  renamed, and the parent fsynced. An existing target is refused, and staging is removed on failure.
- **Plain data only.** Artifacts contain only tensors, containers and scalars. They are read solely with
  `torch.load(weights_only=True, map_location="cpu")`, and every tensor is a detached CPU copy.
- **External anchor required.** A reader must be given a pinned manifest digest or the event checkpoint
  itself; the checkpoint's digest and boundary must match.
- **Tamper refusal:**
  - exact file set (extra files and symlinks refused);
  - canonical manifest;
  - per-file SHA-256 and size;
  - tree digest;
  - tensor inventory;
  - boundary recomputation.
- **Identity refusal:** a foreign seed, update count, seed plan, event checkpoint, schema id/version or
  feature/action/model identity raises `SidecarIdentityError`.
- **No mutation on failure.** `apply_training_state` validates exact tensor names, dtypes and shapes,
  optimizer parameter groups and hyperparameters, per-parameter moment shapes, and generator state shapes
  before touching anything. It then applies the state and re-checks the boundary. If any step fails, it
  restores a snapshot, and tests show the target is left exactly as before.
- **Direct actor cold-load.** `load_actor_from_sidecar` builds a fresh actor, loads it strictly, and checks
  the boundary hash and the 20 fixture outputs. It needs no training replay.
- **Registered-checkpoint gate.** `require_materialized_checkpoints` fails with `SidecarIncompleteError`
  if any `update_*.checkpoint.json` lacks a cold-loadable `actor.pt`. It checks presence for every update
  before parsing anything. **This is the requirement the earlier tests missed.** Run against the
  historical seed-43 checkpoints it fails, as it should, reading only directory listings.
- **Import is side-effect free.** Importing does no I/O and starts no runtime. This is tested with a
  Python audit hook in a subprocess, and the hook itself was checked to detect file opens.

**Scope limits (explicit):**
- Replay-buffer contents are not materialized; exact continuation is the sidecar plus a replay-buffer
  rebuild from the event ledger.
- Counters are recorded and checked, but `apply_training_state` does not write them into a live
  orchestrator.
- The replay and trainer generators are private attributes of the frozen Run-4 runtime. They are read
  and set in place through `get_state`/`set_state`, with no Run-4 edit.
- These two generators are not in the Run-4 boundary. Their correctness is therefore shown by equality
  with an event-sourced restore of a synthetic run (below), not by a boundary hash.

## C. Selected-actor audit (`audit_run4_selected_actor_v1.py`, `RUN4_SELECTED_ACTOR_AUDIT.json`)

Verdict **PASS**, 12/12 checks. There was no retraining, no replay, and CUDA was never initialized.

- Seed 43, update 10,000.
- `actor_state_dict.pt` exists as a regular file, SHA-256
  `d064013d011b67dcd2c7c23acc3c396afe6750be0d43ef0204f2fbecbb9b8e29`.
- The weights-only reload tree digest, and the deployment loader's boundary, both equal
  `b61f27a9bcd3512ecf52bc35854f6a723d550092db51cf3055347297039cebd3`.
- The 13-tensor inventory equals the sealed manifest (self-digest `7f17a025…5140`).
- The 20 deterministic fixtures equal the sealed manifest.
- The update-10,000 event checkpoint has file SHA-256 `3db78b84…e775` and canonical SHA-256
  `aba44d2d…6672`. Its recorded `boundary.actor_sha256` equals the value above.

## D. Phase-6 undefined-quality memo

See `PHASE6_UNDEFINED_QUALITY_DECISION_MEMO.md`. It corrects the Phase-6 report's "about 40% undefined"
statement: the whole-grid rate is 4.8%, and the live Route-B rate is 1.5–2.2%, grouped in one stretch of
the route. No option is selected.

## E. Tests

All runs used `env -u PYTHONPATH CUDA_VISIBLE_DEVICES=`.

| Suite | Result |
|---|---|
| `test_sidecar.SyntheticSidecarTest` | 17 OK (≈3 s) |
| `test_sidecar.OrchestratorSidecarIntegrationTest` | 5 OK (104 s) |
| Run-4 `test_checkpoint_io` + `test_models` + `test_modeled_smoke_orchestrator` | 53 OK (446 s) |
| `splitfusion_hybrid_sac_live_route_b_v2` package, including Phase-1 frozen actor and Phase-5 sealed-manifest verification | 115 OK |

The integration test drives the real Run-4 orchestrator with the existing synthetic fake collector (no
historical data):
1. It runs 0 → 100 with the sidecar callback chained after an event-checkpoint writer.
2. It cold-loads the update-100 actor and checks the boundary digest and fixture equality with the live
   actor.
3. It applies the sidecar to a fresh orchestrator. The actor, critics, both optimizers and decision RNGs
   equal the event boundary, and all five generator states equal both the live source and an
   event-sourced restore of that run.
4. It checks the registered-checkpoint gate passes, then fails once one `actor.pt` is removed.
5. It checks a foreign seed or event checkpoint is refused.

## Preservation

These were re-hashed after the work, and all are byte-identical:
- the Run-4 three-seed campaign (all 22 checkpoint JSONs, manifests, `SEED_COMPLETE`);
- the seed-43/update-10,000 actor export;
- the committed Phase-6 runner and every tracked file of the live-v2 package (108 files);
- every user-owned dirty and untracked file;
- the OAI submodule diff;
- the quality-grid database, opened `immutable=1`.

## Files

Created, all in `rl_agent/splitfusion_hybrid_sac_checkpoint_sidecar_v1/`:
- `__init__.py`
- `sidecar.py`
- `test_sidecar.py`
- `audit_run4_selected_actor_v1.py`
- `RUN4_SELECTED_ACTOR_AUDIT.json`
- `PHASE6_UNDEFINED_QUALITY_DECISION_MEMO.md`
- `SIDECAR_REPORT.md`

Nothing outside the package was changed.
