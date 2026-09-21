# Cooperative demo vs. production direct edge-to-map: compatibility audit v1

Date: 2026-09-21. Branch: `spatial-map-coop-v1`. Offline, read-only audit.

**Headline: the two paths are NOT plug-compatible.** The underlying p025 object
records are semantically compatible and normalize without modification. Every
outer layer — transport, framing, schema, identity space, clock domain, timing
provenance, terminal semantics and pose availability — differs. Connecting the
two endpoints directly would require a translation this audit explicitly refuses
to perform silently.

## Reference material

Read-only, from the master worktree (`.../neu_collab/abiodun`, branch `master`,
HEAD `18b1af3`). The production package **does not exist at this branch's
commit**, so it cannot be imported here; it is pinned by digest instead
(`reference_pins.py`).

| File | SHA-256 |
| --- | --- |
| `rl_agent/splitfusion_direct_edge_map_v1/protocol.py` | `158ecab3a8b3f2bd7efab2862a9ac5e12343fab38dad4760f295d28dff5df0da` |
| `rl_agent/splitfusion_direct_edge_map_v1/map_ingest.py` | `dc0830c179f51874fb8d6038cd583e703649c394c9ff1d6385f757883462cb1c` |
| `rl_agent/splitfusion_direct_edge_map_v1/edge_publisher.py` | `8e54f896e49b0656c02f6667b5ac91dbe8eb82338d562caa468622a95f7584c7` |

Compared against, in this branch: `spatial_map_coop/multi_ue_v1/contracts.py`,
`spatial_map_coop/multi_ue_v1/live_two_ue_action50_v1.py`,
`spatial_map_coop/spatial_map_server_moving_ego.py`.

Framing constants were additionally read from
`phase2_map_sharing/transport.py` in the master worktree.

## Compatibility table

| Dimension | Cooperative demo (this branch) | Production direct edge-to-map v1 | Compatible? |
| --- | --- | --- | --- |
| **Transport** | HTTP `POST /api/multi_ue/v1/live_updates` to a localhost Flask app (`spatial_map_server_moving_ego.py:1879`). Single JSON body, TCP, uncompressed. | UDP datagram burst, edge container → host address on the CN5G bridge (`192.168.70.129:39320`). `DirectMapPublisher` refuses UE tunnel, loopback, `0.0.0.0` and UE result ports. | **No.** Different L4, different addressing, different audit. |
| **Framing** | None. One HTTP body = one update. | `zlib.compress(level=1)` over canonical JSON, then `chunk_payload` with an 8-byte `!IHH` header (`message_id` u32, `index` u16, `total` u16), default `chunk_bytes=12500`. Reassembled by `ChunkReassembler` with a timeout and expiry accounting. | **No.** The demo has no compression, chunking, reassembly or expiry path. |
| **Schema / version** | `scenesense.multi_ue_live_view_update.v1` envelope wrapping `splitfusion_edge_result.v2` wrapping `splitfusion_object_map_update.v1`. **Unversioned** — no `protocol_version` field. | `splitfusion_direct_object_map_update.v1`, flat, with `protocol_version == 1` strictly enforced. | **No.** Three nested schemas vs. one flat versioned schema. |
| **Run identity** | Absent. The demo has no run concept. | `run_id`, mandatory, and validated against the map's frozen `expected_run_id` (`map_ingest._validate_binding`). | **No.** Normalized as `None`, never fabricated. |
| **Cell identity** | Absent. | `cell_id`, mandatory, validated against `expected_cell_id`. | **No.** Normalized as `None`. |
| **UE identity** | `ue_id` in the envelope, cross-checked against the server's `--multi-ue-source STREAM=UE` map. | **Not carried at all.** The production document has no UE identity; the edge knows a `stream_id` only. | **No.** Must come from explicit configuration on both paths. |
| **Session identity** | `session_id`, must equal the server's `--multi-ue-session-id`. | Absent; `run_id`/`cell_id` play the campaign-scoping role. | **No.** |
| **Stream identity** | `stream_id`, e.g. `two-ue/ue-a`. Registered per UE/session; rollover refused. | `stream_id`. Used as the supersession key together with run/cell/frame. | **Partially.** Same field name and role, different namespace and no UE binding. |
| **Frame / sequence identity** | `frame_id` only, cross-checked across envelope, edge result, object update and every record. | `frame_id` **and** `sequence_id`. Supersession key is `(run_id, cell_id, stream_id, frame_id)`. | **Partially.** `sequence_id` has no demo counterpart. |
| **Capture clock** | `capture_timestamp_ns` in the **CARLA simulation** domain (`CLOCK_DOMAIN = "carla_simulation_ns"`; the server refuses any other value). | `capture_timestamp_ns` in the **unix wall** domain — `map_ingest` computes `age_s = ingest_at - capture_ns/1e9` against `time.time()`. | **No, and not translatable.** No measured offset contract exists. |
| **CARLA sim time on the production path** | Native. | `carla_timestamp` field exists but `live_pilot_runtime_direct_v1.py` passes **`0.0`**. There is no usable simulation clock in the production document. | **No.** Cannot be used to bridge the domains. |
| **Receipt clock** | The demo sets `received_timestamp_ns = capture_timestamp_ns` in the sim domain — i.e. it deliberately declines to mix clocks. The Flask wall receipt is kept separately as `received_at`. | `map_ingest_at` / `first_datagram_at` / `reassembly_complete_at`, all `time.time()` unix wall seconds. | **No.** |
| **Action identity** | `action_id` (demo locks 50), cross-checked envelope/edge/update. No `profile_id` requirement. | `action_id` mandatory, `0 <= action_id < 72`, plus a per-cell allowlist; `profile_id` and `decoder_identity` mandatory. | **Partially.** Same catalog, stricter production binding. |
| **Object record location & format** | p025 records: `world_x/world_y/world_z`, `size_x/y/z`, `yaw_sin`/`yaw_cos`, `score`, `class_name`, `candidate_identity`. | **Identical p025 record layout**, carried in `records` with `record_count` and `record_scalar_count` cross-checks and a whole-subtree finiteness assertion. | **Yes.** This is the one genuinely compatible layer. |
| **Edge timings** | None. The demo carries no edge stage stamps whatsoever. | `edge_timing` with `tail_complete_wall_s`, `reassembly_complete_wall_s`, `admission_wall_s`, `compute_start/finish_wall_s`, `publication_ticket_ready_wall_s`, `publish_start_wall_s`, `serialization_start_wall_s` — unix wall seconds. | **No.** Normalized as absent, never zero-filled. |
| **Map-install timings** | None. The demo returns HTTP `202` with a disposition string. | `install_timestamp`, `map_ingest_at`, `feedback_emit_at`, `map_age_at_install_ms`, `install_latency_from_publish_ms`, `install_latency_from_tail_ms`, plus lock/association stamps. Install is under the map's `state_lock` and precedes feedback by construction. | **No.** |
| **Ego / sensor pose** | **Present.** `ego_pose.location`/`rotation` measured, mandatory, finiteness-checked. | **Absent.** Protocol v1 carries no pose. `FrameContextV1.ego_world` exists upstream in SFD1 but is not put on the wire. | **No.** See objective C below. |
| **Sensor FoV** | **Present.** `horizontal_fov_deg` (120°), used for the FoV wedges. | **Absent.** `StaticCameraCalibration` holds intrinsics upstream but is not carried. | **No.** |
| **Terminal outcome** | None. HTTP `202` + `{"status": "ACCEPTED", "disposition": ...}`. Not a terminal contract. | Exactly one terminal per obligation from `TERMINAL_OUTCOMES` (7 values), via `splitfusion_direct_map_feedback.v1` or `splitfusion_edge_terminal_control.v1` or the UE-local timeout; disjoint by construction. | **No.** |
| **Agent credit** | None. | 8 `AGENT_CREDITS` values; `classify_agent_credit` credits a deliberately replaced frame `CREDIT_SUPERSEDED_BY_FRESHER`, never a network failure. | **No.** |
| **Record-free UE path** | Not applicable — the demo's only message carries records. | `assert_no_object_records` walks every UE-bound document and refuses `records`, `objects`, `semantic_labels*`, `label_map`, `dense_mask`; control messages are capped at 8192 bytes. | **No counterpart.** |

## What this means operationally

1. **Do not connect the two endpoints.** The demo's Flask route would reject a
   production document on schema alone, and the production `DirectMapPublisher`
   constructor would refuse the demo's localhost endpoint by design.
2. **The HTTP demo transport is not a radio.** Its POST duration is
   `demo-control delivery time`; its inference is a `localhost model path`.
   Neither may be labelled 5G, radio, uplink or network latency.
3. **Cross-clock AoI for the demo transport is `N/A - incompatible clock
   domains`,** not a number. Enforced in `live_metrics_v1.cross_clock_age_ms`.
4. **UE identity is configuration, never inference.** The production document
   carries no UE identity at all, so `(run_id, cell_id, stream_id) ->
   (session_id, ue_id)` must be supplied explicitly. `identity_config.py`
   refuses substring matching, ordinal assignment and wildcard fallback.

## Objective C: the pose/FoV gap

Protocol v1 carries no pose and no FoV, so the 40 m ego-follow view and the FoV
wedges cannot be driven from a production document. Two options were considered.

**Option 1 — versioned pose sidecar (implemented here, offline only).**
`scenesense.coop_pose_sidecar.v1`, keyed by
`(run_id, cell_id, stream_id, frame_id, capture_timestamp_ns)`. Requires no
change to the frozen v1 object protocol, which is why it is the only one
implementable from this branch. Weakness: pose and objects are two messages and
can be lost independently, so the viewer must tolerate "objects present, pose
absent" — and must *say* `pose unavailable — ego-follow and FoV disabled` rather
than draw a wedge.

**Option 2 — `splitfusion_direct_object_map_update.v2` (recommended, not
implemented).** Carry the SFD1 `FrameContextV1.ego_world` pose and the camera
horizontal FoV atomically inside the object update. One message, one identity, no
join, no partial-arrival state. The data already exists at
`rl_agent/splitfusion_live_dispatch_v1/frame_context.py`. Not implemented because
**the frozen production v1 protocol in the main worktree must not be modified
from this branch.**

Either way: no CARLA ground-truth actor is queried for a pose. The sidecar
parser refuses a document declaring `CARLA_ACTOR_GROUND_TRUTH`,
`CARLA_WORLD_SNAPSHOT` or `SIMULATOR_TRUTH` provenance, because putting simulator
truth behind the perception path would silently convert a perception
demonstration into a ground-truth one.

## Claim boundaries

* **Implemented and offline-tested:** the normalized type, both pure adapters,
  the identity configuration, the pose-sidecar contract, the metrics/labelling
  rules and the Route-B binding.
* **Compatible by pure contract/fixture testing only:** the p025 record layer.
  Verified against canonical fixture documents shaped by the pinned production
  files, not against bytes captured from a live edge.
* **Still requires master-based integration:** a bridge worktree that imports
  `rl_agent.splitfusion_direct_edge_map_v1.protocol`, calls
  `validate_object_map_update` **first**, and then hands the document to
  `from_direct_object_map_update`. The production validator is the authority and
  is deliberately not duplicated or relaxed here.
* **Still requires live validation:** everything about running the two paths
  together. Nothing in this audit is a live, performance or UI qualification.
