# Phase 6 setup-repair addendum 3: start the edge without rebuilding

**Status:** `REGISTERED_BEFORE_ANY_NEW_PHASE6_EVIDENCE`, 2026-09-30. Base commit `4f2e68d`.
The machine-readable record is `phase6_setup_repair_addendum_3.json`.

**This changes only how the edge image is started and how its provenance is proven. The scientific
protocol is unchanged.** These stay exactly as they were:
- the actor, the 21-D state and the fallback;
- the reward, the 170-ms timeout, k_min and the hold;
- action execution, and the SFD4, map and reward protocols;
- quality evaluation;
- gates P0–P8 and the verdict rule;
- Option-C exclusion and rollover handling.

Addendum 2 still sets the claim scope: `SYSTEMS_INTEGRATION_QUALIFICATION_ONLY`.

## Why

The first authorized attempt, `20260930T000922Z_phase6_live_qualification_option_c`, failed at setup
with zero frames sent. P0–P8 were never evaluated. It is preserved byte-for-byte, and every file hash
is listed in the JSON.

The cause:
- The shared launcher runs `docker compose up -d --build --force-recreate`.
- The host build cache and the base image had been pruned.
- BuildKit therefore rebuilt from step `[3/7]`, a multi-GB torch download.
- The adapter's 180-s launch timeout cancelled the build.

## Repair

`phase6_edge_launch_v2.py` reproduces `scripts/receiver_container_fusion_back_up.sh` exactly:
- the same network check and NVIDIA-runtime probe;
- the same edge-state validation;
- the same variable derivation and the same `sudo VAR=… docker compose` environment filtering;
- the same two compose files.

It differs only in the `up` line:

```
docker compose -f docker-compose.yaml -f docker-compose.fusion-back.yaml \
  up -d --no-build --pull never --force-recreate
```

It never builds, pulls, retags or downloads. The shared launcher, both compose files and the Dockerfile
are untouched.

`phase6_live_child_nobuild_v2.py` runs the unchanged `phase6_live_child_v2`. It adds one seam that
routes only `adapter_direct_v1`'s launcher call through the no-build path. All adapter readiness,
`cuda:0`, preload and ready-record checks are unchanged.

## Image authority

The only admitted image is
`sha256:2be62d533b8077ceecab5455d5377f2f952b6a50d43ff8c04dc89ce18027d6ba`:
- created 2026-09-05T20:55:11-07:00, amd64/linux;
- 16 RootFS layers, list digest `29e796b4…70df`;
- config digest `54ef3489…cc85`;
- `RepoDigests` is empty, which is why the full ID is the binding.

The ID is checked at three points:

1. **Before OAI/CARLA start.** The runner resolves `oai-perception-rx:latest`, requires `.Id` to equal
   the admitted ID, and refuses otherwise. Evidence: `edge_image_prelaunch.json`.
2. **Before `compose up`.** The launcher resolves the tag again.
3. **After container creation, before readiness is accepted.** `oai-perception-rx`'s `.Image` must equal
   the admitted ID. The mounts must be exact:
   - repository → `/work/abiodun`, read-only;
   - per-attempt state → `/work/torch_cache`, read-write;
   - FCOS checkpoint, read-only, with its pinned SHA-256.

   Evidence: `run4_phase6/edge_image_launch.json`.

The tag is resolved once more after the run, as a recorded field only. It feeds no cleanup gate.

## Sequence

1. Offline tests.
2. A bounded edge-startup qualification: core network and edge only, no CARLA and no scientific frame,
   then teardown to a cold host. If it fails, preserve the evidence and stop.
3. Only if step 2 passes: one fresh 300-frame FAVORABLE_STABLE Option-C qualification.
4. No further attempt.
