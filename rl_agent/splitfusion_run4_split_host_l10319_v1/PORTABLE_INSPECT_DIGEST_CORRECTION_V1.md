# Portable image-inspect digest correction

Status: corrected before another startup attempt; no service or live run was
started by this correction.

The original `7f8a1457...3d91` value first appeared in commit `4d0e184`
without retained source evidence. Startup attempt 002 correctly refused it at
preflight. The attempt remains only on L10319 at
`/home/shr_aisvcs/run4_split_host_attempts/edge-startup-002`:

- `REMOTE_EDGE_STARTUP_QUALIFICATION.json` SHA-256
  `f8fdd8832b7162fcd3bec3528fef190f7ffac1d19d6f709039fddb561d2aa9a1`;
- `REMOTE_EDGE_STARTUP_PLAN.json` SHA-256
  `54019167715830f94c4ea307e463bae4ecaed9441796dbc75cd2dd3d8dbec358`.

Independent current `docker image inspect` observations of the source image
and the transferred L10319 image produced byte-equivalent values for the seven
selected fields. Their canonical SHA-256 is
`c5a6690923255163ccb090db0c6f278b1761e4c52e516f024ba88c970ee918d1`.
`PORTABLE_IMAGE_INSPECT_SELECTED_FIELDS_V1.json` retains the local-equivalent
selected document, and the offline regression recomputes the digest from it.

Only this selected-field digest changed. The OCI manifest digest `ac143760...`,
source config digest `2be62d53...`, remote image/container ID `ac143760...`,
and all seven artifact hashes remain mandatory and unchanged.
