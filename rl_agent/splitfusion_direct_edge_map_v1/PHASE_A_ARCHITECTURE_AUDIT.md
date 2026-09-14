# Phase A — SplitFusion edge/map architecture audit

Audited at HEAD `b88176da81ab044171032b25c2eab3baeb1c28b4` (2026-09-14).
Every claim below is traced to a file and line in the repository as it stands at that commit.

## A.0 Process/namespace inventory

| Component | Where it runs | Identity | Started by |
|---|---|---|---|
| CARLA server | host | `CarlaUE4` process | `data_collection/run_route_b_collection_supervised_v1.py` |
| UE bridge (front/ranker/AE, SFD1-v2 sender) | **host process**, sockets bound to `10.0.0.2` (`oaitun_ue1`) | `LivePilotCellRuntime` | `rl_agent/ue_route_b_split_cell_adapter_v1.py:1225` |
| Edge inference service (frozen tail/decoders) | **docker container `oai-perception-rx`**, static IP `192.168.70.140` on external bridge `oai-cn5g-public-net` | `run_edge_service` | `start_live_edge` -> `scripts/receiver_container_fusion_back_up.sh` |
| Spatial map application | **host process** | `spatial_map_server_moving_ego_uplink_only_baseline.py` | `start_map_process` (`ue_route_b_split_cell_adapter_v1.py:775`) |
| Install-feedback ledger | **host process**, inside the UE adapter | `InstallFeedbackLedger` | `ue_route_b_split_cell_adapter_v1.py:1242` |
| OAI CN5G / gNB / UE softmodem | host + `oai-cn5g-public-net` containers | `scripts/cn_start.sh`, `gnb_start.sh`, `ue_start.sh` | operator, **not** the adapter |

Network facts (verified):

- `receiver_container/docker-compose.yaml:30-34` attaches the edge to `public_net`
  (`external: true`, `name: oai-cn5g-public-net`) at `ipv4_address: 192.168.70.140`.
- `OAI/oai-cn5g/docker-compose.yaml:169-177` defines that network: `subnet: 192.168.70.128/26`,
  bridge interface name `oai-cn5g`, **no explicit `gateway:` key**.
- Docker's IPAM assigns the first usable address of the subnet to the host bridge when no gateway
  is declared. Proven on this host by creating a probe network with the same prefix length:
  `--subnet 192.168.99.128/26` produced `Gateway 192.168.99.129` and host interface
  `sfprobe 192.168.99.129/26`. Therefore the host address on `oai-cn5g` is **`192.168.70.129`**.
- `10.0.0.2` is the UE tunnel address on `oaitun_ue1`. Traffic sent from the edge container to
  `10.0.0.2` is routed CN5G-bridge -> UPF (`192.168.70.134`) -> GTP -> gNB -> RFsim -> UE,
  i.e. **over the simulated radio downlink**.

## A.1 UE feature transmission

`live_pilot_runtime.py` `LivePilotCellRuntime.submit` (lines 795-870):

- socket `self.sender` bound to `(runtime["ue_bind_host"], 0)` = `10.0.0.2:ephemeral` (line 730).
- destination `self.remote = (runtime["edge_remote_host"], runtime["edge_receive_port"])`
  = `192.168.70.140:51002` (line 738).
- payload fragmented by `chunk_payload(..., chunk_bytes=12500)` with the production `!IHH` header.
- two capture-based deadline gates (`UE_AFTER_PREPARATION`, `UE_BEFORE_SEND`) refuse stale captures
  before the radio is used; refused captures are recorded `STALE_BEFORE_SEND` (line 757).

## A.2 Edge feature reception / reassembly

`run_edge_service.receive_loop` (lines 1185-1250):

- `receiver` bound `0.0.0.0:51002` inside the container (line 1109).
- `ChunkReassembler(timeout_s=2.0, max_chunks=4096)` -> complete message (line 1197).
- `unpack_envelope` -> action allowlist check -> frame-context check ->
  `EDGE_AFTER_REASSEMBLY` deadline gate.
- admission into `LatestFramePendingSlot` (depth one, latest-frame-first); a displaced frame
  increments `edge_pending_replacements`, a refused one `edge_admission_refused_not_freshest`.

## A.3 Edge result construction

`run_edge_service.process_loop` (lines 1252-1390):

- `EDGE_BEFORE_DECODE` gate, `edge.process(...)` (frozen tail), `tail.take_snapshot()`,
  `_finite_tree`, frame-identity check, `EDGE_BEFORE_PUBLICATION` gate.
- dense 720x1280 label map is written to the edge's own evidence mount by
  `EdgeEvaluationEvidenceWriter` and is **not** placed on the radio (Phase-15 recovery).
- builds one JSON document `EDGE_RESULT_SCHEMA = splitfusion_edge_result.v2` that contains
  **both** an `object_map_update` block (`splitfusion_object_map_update.v1`, field `records`)
  **and** an `edge_terminal_ack` block.

## A.4 `object_map_update` transport — THE DEFECT

The object records take a three-hop detour:

1. **edge container -> UE, over the radio downlink.**
   `live_pilot_runtime.py:1395`
   `sender.sendto(chunk, (str(result_host), int(result_port)))`
   with `result_host`/`result_port` supplied by
   `ue_route_b_split_cell_adapter_v1.py:673`
   `"--result-host", str(runtime["ue_bind_host"]), "--result-port", str(runtime["camera_result_port"])`
   -> **`10.0.0.2:51004`**. Defaults in `live_pilot_runtime.py:1451-1452` are the same.
   `runtime.ue_bind_host = 10.0.0.2` in `rl_agent/configs/ue_288_campaign_v1.yaml` and in
   `rl_agent/configs/splitfusion_16_cell_live_carla_oai_pilot_v1.json`.

2. **UE re-serialises and forwards to the map.**
   `live_pilot_runtime.py:962-984`: the UE result loop reassembles the edge result, lifts
   `update["records"]` into a new `fusion_object_spatial_map.v1` document, and does
   `self.map_socket.sendto(zlib.compress(...), self.map_remote)` where
   `self.map_remote = (map_host, map_port)` = `("127.0.0.1", 39310)`
   (`ue_route_b_split_cell_adapter_v1.py:1225`).

3. **Map installs.**

Consequences that the corrected architecture removes:

- the object map update is carried by the **5G downlink** and is subject to downlink loss,
  fragmentation and reassembly (`result_datagrams_received`, `result_incomplete_reassemblies_expired`);
- map installation cannot happen at all unless the UE is reachable and its result loop is alive;
- the measured post-tail install delay
  (`counterfactual_288.py:300 _post_install_ns = map_installed_at - edge_tail_complete_wall_s`)
  contains a full edge->UE radio hop plus a UE re-serialisation, and this quantity is the
  `predicted_post_publication_install_ns` input of the 288-cell counterfactual
  (`counterfactual_288_final_v3.py:323`). Every published map-freshness/AoI number therefore
  inherits the detour.

## A.5 Spatial-map ingestion and installation

`uplink_only_spatial_map_pipeline/spatial_map_server_moving_ego_uplink_only_baseline.py`:

- `udp_listener_thread` (line 1663) binds `(cfg.udp_host, cfg.udp_port)`; the adapter passes
  `--udp-host 127.0.0.1 --udp-port 39310` (`ue_route_b_split_cell_adapter_v1.py:785-786`).
  **Loopback-only: it is not reachable from the edge container today.**
- `zlib.decompress` -> `json.loads` -> `_normalize_packet` -> schema must equal
  `fusion_object_spatial_map.v1`.
- installation under the authoritative lock (lines 1716-1723):
  `with state_lock: latest_streams[stream_id] = normalized;
   installed_frame_history[(stream_id, frame_id)] = normalized` (bounded to
  `installed_frame_history_size = 256`).
- `install_timestamp = time.time()` is taken immediately before the lock (line 1711).

Validation gaps in the existing ingest path: it checks only the schema string. It does **not**
check run/campaign ID, cell ID, action/profile identity, capture-timestamp freshness, finiteness
of object records, or duplicate/superseded installation.

## A.6 Map-install feedback to the UE

`_emit_install_feedback` (line 387):

- sends from the **same** UDP ingest socket to
  `(cfg.install_feedback_host, cfg.install_feedback_port)` = `127.0.0.1:<feedback_port>`
  (`ue_route_b_split_cell_adapter_v1.py:787`).
- ordering is already correct: `ACK_INSTALLED` is emitted only after the `state_lock` insert
  (line 1725, after lines 1716-1723). `NACK_REJECTED` is emitted for schema/decode failures.
- schema `scenesense.map_install_feedback.v1`, carrying frame/capture/action identity and
  `install_timestamp`. It carries **no object records** already.
- received by `InstallFeedbackLedger` (`rl_agent/ue_map_install_feedback_v1.py`) bound
  `127.0.0.1:<feedback_port>`; `record_expired` writes the UE-local `TIMEOUT_NO_ACK` terminal.

So the *feedback* leg is structurally sound. The defect is confined to A.4: the **object records**
travel edge -> UE -> map instead of edge -> map.

## A.7 Chosen direct edge->map endpoint (decision)

**Endpoint:** the host's address on the CN5G bridge, resolved at launch from
`docker network inspect oai-cn5g-public-net` `IPAM.Config[0].Gateway` (= `192.168.70.129`),
on a new port `39320`.

Why this is the correct direct path:

- it is a **container-to-host** datagram on the directly-connected `oai-cn5g` bridge; the route
  is a link-scope route on the edge container's `eth0`, so it never enters the UPF, never enters
  a GTP tunnel, and never touches `oaitun_ue1` or RFsim;
- it is reachable from inside the container without any added route or port publication;
- it is not `127.0.0.1` (unreachable from the container) and not `0.0.0.0` (which would also accept
  the legacy loopback detour and make the audit weaker).

Rejected alternatives: `10.0.0.2` (the defect being removed — traverses the radio);
`192.168.70.140` (that is the edge itself, not the map); host-network mode for the edge container
(would change the measured transport identity that the SFD1 pins exist to protect).

**Feedback path:** the map emits the compact ACK/agent-credit message to the UE's registered
control endpoint `10.0.0.2:51014`. Honest limitation, recorded now and repeated in every report:
`10.0.0.2` is a **local** address on this host (`oaitun_ue1`), so a datagram sent to it *from a host
process* is delivered by the kernel locally and does **not** traverse the radio. The map runs on the
host, therefore the measured ACK latency is a host-local controller-observation delay and is a
**lower bound** on an over-the-air control-plane delay. It is measured and reported separately from
physical map freshness, and is never included in map-installation AoI.

## A.8 Implementation constraints derived from the audit

1. `live_pilot_runtime.py`, `ue_route_b_split_cell_adapter_v1.py`,
   `spatial_map_server_moving_ego_uplink_only_baseline.py`, `live_pilot_target_snr_runtime.py`
   and `ue_map_install_feedback_v1.py` are SHA-256 pinned by the campaign configs and the
   Phase-14A binding. `edge_runtime.py` / `context_tail.py` are pinned by the SFD1-v2 authority
   (`runtime_binding.json`, `frame_context_binding.json`).
   => **no in-place edits.** All new behaviour lands in a new versioned package plus a new config.
2. The frozen models, split points, ranker/AE/quantizer/q settings, SFD1 payload semantics,
   network profiles/seeds, 100 ms cadence, latest-only scheduling, dense-mask edge-only evidence
   and p025 filtering are reused by importing the existing modules unchanged.

PHASE_A_ARCHITECTURE_AUDIT_COMPLETE
