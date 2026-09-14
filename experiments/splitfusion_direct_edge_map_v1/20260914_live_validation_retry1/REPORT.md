# Direct edge-to-map live validation

Four actions under FAVORABLE_STABLE, each on a fresh CARLA world and a
fresh OAI radio, torn down completely between actions.

**This is a short live validation, not a Route-B loop completion run,**
**and not an authorization for another 288-cell live campaign.**

## Per-action evidence

| Action | Family | Sent | Published | Installed | Publish->install p50 (ms) | Map age p50 (ms) | ACK delay p50 (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| 15 | noAE | 2967 | 1306 | 1274 | 5.693 | 411.4 | 2.768 |
| 30 | AE128 | 2923 | 1047 | 999 | 4.757 | 409.8 | 2.839 |
| 50 | AE64 | 2969 | 1746 | 1738 | 3.300 | 327.6 | 0.995 |
| 71 | AE32 | 2939 | 1808 | 1797 | 5.259 | 301.7 | 0.755 |

## Gates

| Gate | Holds |
|---|---|
| every_registered_action_produced_a_cell_with_data | yes |
| real_ue_feature_uplink_reached_the_edge | yes |
| object_map_updates_were_published_directly_to_the_map | yes |
| no_object_update_targeted_the_ue_result_address | yes |
| legacy_loopback_map_listener_was_never_started | yes |
| map_installation_preceded_every_feedback_emission | yes |
| ue_received_compact_record_free_feedback | yes |
| exact_frame_action_stream_identity | yes |
| no_duplicate_installation | yes |
| dense_masks_remained_edge_only | yes |
| direct_publication_install_latency_is_measured | yes |
| ack_latency_is_measured_separately_from_installation | yes |
| exactly_one_terminal_per_transmission_obligation | yes |
| superseded_frames_are_credited_as_replaced_not_lost | yes |
| at_least_the_requested_frames_were_transmitted | yes |
| clean_teardown | yes |

## Interpretation limits

- Physical map freshness ends at the map install timestamp; the ACK
  arrival at the UE is a separate controller-observation delay and is
  never included in map-installation age.
- `10.0.0.2` is a local address on this host (`oaitun_ue1`), so the
  map -> UE feedback is delivered locally by the kernel and does not
  traverse the radio. The ACK delay is therefore a lower bound on an
  over-the-air control-plane delay.
- The Route-B loop is deliberately budget-bounded; loop completion is
  neither required nor claimed.
