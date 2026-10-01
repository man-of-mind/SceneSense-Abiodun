# Setup-repair addendum 1: edge repository mount from a worktree

This addendum was registered after attempt 1 and before attempt 2. The gates
G1–G9 and the scientific configuration are unchanged.

## Attempt 1: `20261001T223947Z_run4_instrumentation_qualification_300`

**Classification:** `SETUP_REFUSED_ZERO_FRAMES`. Attempt 1 is not a
qualification result.

- The child stopped with
  `EdgeImageError: edge repository mount source/mode drift`.
- That stop came from the unchanged Phase-6 `inspect_container` check, after
  `docker compose up` and before the edge became ready.
- No route started and no frame was transmitted. There is no
  `PHASE6_UE_EVIDENCE.json`.
- Teardown was complete:
  - tracer, application, CARLA and radio were stopped;
  - the host was cold afterwards;
  - no profiler processes were left;
  - the actor audits passed before and after.
- The sampler ran correctly through the radio, CARLA and edge bring-up:
  - 1,184 samples, 0 dropped;
  - interval P50 100.000 ms, P95 100.171 ms, max 106.564 ms;
  - 4.25 % of one core.
- The evidence is preserved as written.

## Cause

`receiver_container/docker-compose.yaml` mounts `../../abiodun` at
`/work/abiodun`.

- From the main checkout, that path is the checkout itself.
- From a worktree at `neu_collab/abiodun_run4_single_gpu_profiler_v1/`, it
  resolves to the **main checkout**. That checkout is on another branch and
  has no profiled edge module.

The Phase-6 check correctly refused to run the edge from foreign code. Running
from a clean worktree, as the task requires, is therefore impossible without
re-binding this one mount.

## Repair (process-local, additive)

`seams_ue_v1.install_repository_mount_override` makes two changes:

- It writes `compose_repository_mount_override.yaml`, create-only, into the
  attempt's profiler directory.
- It appends that file as the last `-f` file of the no-build launch command.

The override re-binds only `/work/abiodun`, to this worktree, read-only.
`docker compose config` was checked offline: the merged service has exactly
one `/work/abiodun` mount, and the X11, `torch_cache` and FCOS mounts are
unchanged.

Everything else stays as it was:

- the admitted image ID;
- `--no-build --pull never --force-recreate`;
- the network and NVIDIA runtime checks;
- the variable filtering;
- the unchanged post-create mount check, which now verifies source = this
  worktree (read-only).

No pinned file is edited, and the Phase-6 package is still byte-identical to
`6199d49`. This is the same class of repair as the radio-scratch re-anchoring
under the OAI launcher's ROOT, which is also needed only from a worktree.

## Attempt 2

Attempt 2 is the single registered 300-frame instrumentation qualification. It
has no retry.
