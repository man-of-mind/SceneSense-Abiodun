# Run-4 durable checkpoint boundary

`checkpoint_io.py` now provides the crash-safe storage half of persistence.
It atomically publishes a directory containing `checkpoint.pt` and
`manifest.json`, refuses overwrite, verifies SHA-256/length/schema/runner
binding, and loads the payload only with `torch.load(weights_only=True)`.

The stored journal deliberately contains no private attestation token.  Each
row carries the exact decision, prediction and successor staging plus the
expected transition, environment-transition and journal digests.  Reconciled
action identities are re-issued against the frozen action catalogue on load.

## Why full-object `torch.save` is forbidden

A fresh-process test proves that normal pickle loading loses the private
catalog/transition sentinel identities; the loaded action or transition is no
longer attested.  `weights_only=True` correctly refuses the opaque object
graph.  Treating either behavior as a successful restore would weaken the
contract.

## Portable restoration

`persistent_runner.py` now exposes `reissue_portable_journal(...)` on a
pristine runner. The API accepts:

- `runner_binding_sha256`, `session_uuid`, `ue_id`;
- exact `genesis_staged` and the initial-kernel checkpoint digest;
- an ordered tuple of `PortableJournalReplayRowV1` values carrying only
  `(decision, prediction, successor_staged)` and three expected digests.

The runner stages and resets genesis, calls its existing validated
`_execute_prebuilt` primitive for every contiguous row, and compares the newly
built transition, environment transition and journal digests. No serialized
transition object, reconciliation token or attestation is accepted or copied.

`checkpoint_io.restore_runner(...)` uses a disposable runner to perform that
reissuance. It then constructs a fresh `PersistentRunnerCheckpointV1` from the
newly attested rows plus the verified model, optimizer, provider and RNG
material. Construction must reproduce `source_checkpoint_sha256` exactly.
Finally, the existing runner `restore(...)` path independently replays the
checkpoint and rechecks factory binding, causal sequence, replay accounting,
model state, optimizer state, all five RNG streams, provider states and final
checkpoint equality.

The tests include a fresh-Python-process restore and an uninterrupted versus
durably resumed optimizer comparison: both paths train to update 250, one is
written and reloaded, and their independently continued update-500 checkpoints
are bit-identical. CUDA and live services are never initialized.
