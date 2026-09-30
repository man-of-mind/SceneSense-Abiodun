# Run-4 split-host handoff (offline scaffold only)

Status: **BLOCKED**. This package neither launches nor authorizes CARLA, OAI,
Docker, CUDA, a route, or a firewall change.

## Frozen topology

- W10275 (`10.21.16.222`): CARLA, gNB/UE RFsim, UE front/actor, and map.
- L10319 (`10.21.16.162`): CN, ext-DN and edge tail.
- Preserve Docker subnet `192.168.70.128/26` and AMF `.132`, UPF `.134`,
  ext-DN `.135`, edge `.140`. W10275 routes that subnet through L10319.
- The edge publishes object-map UDP directly to `10.21.16.222:39320`.
- A runtime copy of the gNB config binds N2 and N3 to
  `10.21.16.222/24` and targets AMF `192.168.70.132`. The source config is
  never edited.

`contract.network_plan()` produces exact argv vectors. A reviewed executor
must first save the old route, `net.ipv4.ip_forward`, and tagged-rule presence.
It may add only the two source/destination-scoped `DOCKER-USER` rules bearing
`scenesense-run4-l10319-v1`; it must remove only rules it added and restore the
measured prior route/forwarding value. There is no NAT, chain flush, policy
change, or firewall disable.

## Required facts before any startup

1. Populate a remote binding from **measured** L10319 GPU model, UUID, total
   VRAM and driver. There are intentionally no GPU defaults and no RTX 5090
   assumption.
2. Verify exact edge image ID
   `sha256:2be62d533b8077ceecab5455d5377f2f952b6a50d43ff8c04dc89ce18027d6ba`.
3. Transfer and hash all seven edge-host files in `contract.ARTIFACTS`:
   perception, ranker, AE128/64/32, the Phase-15 FCOS constructor weight, and
   `checkpoints/fusion_object_best.pt` required by the compose mount. Git does
   not carry them. The Run-4 actor is the separate `LOCAL_ACTOR_ARTIFACT`; it
   stays on W10275 and must not be required on L10319.
4. Qualify, without CARLA, bidirectional W10275-to-container routing, N2/N3,
   ext-DN reachability, edge UDP receipt, direct map UDP, and compact feedback.
5. Implement/review the remaining Phase-6 split-process GT and lifecycle seam.
   The current single-host child owns map, edge and teardown; moving services
   without an explicit ownership/cleanup contract risks killing the wrong host
   or losing reward GT.

Only after these facts exist can `readiness_report()` become structurally
ready. Even then, it reports `live_run_authorized: false`; a separate explicit
authorization and bounded handshake are required before 300 frames.
