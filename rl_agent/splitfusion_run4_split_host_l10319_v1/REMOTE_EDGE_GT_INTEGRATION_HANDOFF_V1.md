# Run-4 split-host remote edge and GT ingress handoff

Status: offline implementation complete; no service, network, Docker, CUDA, CN,
RAN, CARLA, or live qualification was started.

## Integration points

- `remote_edge_lifecycle_v1.py` starts only the pre-existing remote edge image,
  with `--no-build --pull never --no-deps`, from the exact current worktree.
- `remote_edge_gt_entry_v1.py` composes the measured-GPU shim with a persistent
  TCP GT listener at `192.168.70.140:51015`.
- The listener owns one `ExpectedTicketRegistryV1` and one `GtIngressStoreV1`
  for the edge service's exact `run_id`, `cell_id`, and segmentation-evidence
  directory.
- A process-local subclass of the frozen `Run4EvaluatorV2` authorizes the exact
  verified `EvaluationTicketV2` identity immediately before delegating to the
  unchanged evaluator `submit`. No frozen Phase-6 or GT transport file changes.
- The listener binds/listens before evaluator construction completes. Its health
  is checked again immediately before the frozen pre-warm code publishes edge
  READY. A dead listener therefore prevents READY.
- Shutdown drains/closes the unchanged evaluator first, while GT remains
  available, then stops and joins the listener. Wrapper-finally cleanup also
  covers a worker failure path that the frozen service does not close itself.

## Clock boundary

The GT seam performs no clock subtraction. L10319-local sub-stage durations may
be measured on one L10319 clock. The policy deadline remains the W10275-local
action-open-to-feedback interval. Cross-host one-way latency is not derived.
Wall-clock synchronization/offset should be bounded and reported separately
before interpreting cross-host wall timestamps.

## Bound runtime facts

- Remote edge: `192.168.70.140`; direct map: `10.21.16.222:39320`.
- Remote GPU: RTX 5090 Laptop GPU, UUID
  `GPU-b8c4646c-abb0-5d63-679b-49622ce057b6`, 24463 MiB, driver 610.43.02.
- OCI manifest/container image ID: `sha256:ac143760...901c`.
- Source config digest: `sha256:2be62d...d6ba`.
- Portable inspect hash: `7f8a1457...3d91`.

The lifecycle package still cannot authorize a full live run and contains no
executor. A coordinator must create the per-attempt state/evidence directories,
write the generated standalone Compose document, validate the recorded image,
container, GT-ready, and edge-ready evidence, and own project-scoped teardown.

## Offline verification

Focused tests cover exact identity authorization before evaluator submission,
READY ordering, listener success/failure and idempotent shutdown, current
worktree/endpoint CLI binding, and the absence of cross-host time subtraction.
