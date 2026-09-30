# L10319 remote edge lifecycle handoff

Status: **offline seam implemented; no service or live run executed**.

The additive lifecycle is in `remote_edge_lifecycle_v1.py`. It produces a
standalone, one-service Compose document for `oai-perception-rx`; it never
reuses the legacy Compose file's literal `../../abiodun` bind. The caller must
provide the absolute current L10319 worktree, a unique per-attempt output root,
the registered FCOS constructor weight, and the scientific invocation. State
and evaluator evidence have separate attempt-owned host mounts.

The launch is exactly `--no-build --pull never --no-deps` and selects only the
edge service. The generated lifecycle has no executor, no live-run command,
and no CN, RAN, CARLA, or frame-budget command. Teardown is scoped to the
attempt's Compose project and may run only after the created container's
project label, image ID, binding labels, ID, and mounts have been checked.

## Bound L10319 facts

`REMOTE_RUNTIME_BINDING_L10319_V1.json` records the measured facts:

- GPU: NVIDIA GeForce RTX 5090 Laptop GPU, UUID
  `GPU-b8c4646c-abb0-5d63-679b-49622ce057b6`, 24463 MiB, driver 610.43.02;
- remote OCI manifest/image/container identity `ac143760...`;
- source OCI config identity `2be62d53...`;
- canonical portable inspect hash `7f8a1457...`;
- all seven ignored artifacts under their original relative paths.

The frozen Phase-6 edge has a legacy exact-name check for the desktop RTX
5090. `remote_edge_entry_v1.py` handles only that compatibility issue. Before
calling the frozen service it measures the actual GPU through `nvidia-smi`,
requires all four facts to match the binding, verifies PyTorch sees that same
measured model, and writes create-only evidence. It supplies the legacy name
only while the frozen check executes and restores the real query immediately
afterward. It does not admit an arbitrary CUDA GPU or change model numerics.

## Startup gate

1. Create the attempt, state, and evidence directories and seed the unchanged
   campaign and Run-4 edge configuration.
2. Serialize the plan's `compose_document` exactly and check its canonical hash
   against `compose_sha256`.
3. Execute each preflight and compare its return code with
   `expected_returncode`. The existing CN network must already exist; this
   lifecycle never creates it.
4. Validate the independently collected image observation, then launch only
   the edge service.
5. Validate the created container observation before trusting it or tearing it
   down. Wait for and validate the frozen ready record.
6. Preserve the ready record, GPU-entry evidence, image/container inspect,
   bounded logs, and teardown evidence create-only.
7. Stop. Startup readiness does not authorize a one-frame or 300-frame run.

The direct-map endpoint is `10.21.16.222:39320`; the edge stays at
`192.168.70.140`. Cross-host monotonic timestamps must never be subtracted.
The 170-ms policy verdict remains W10275 action-open to W10275 feedback on one
clock. L10319 may report its own same-clock edge sub-stage durations; any
cross-host one-way decomposition is diagnostic only and must state clock and
network uncertainty.
