# SplitFusion 288-cell results — presentation talking points

## Start with the experimental question

We measured 72 split-inference actions under four time-varying channel
processes. Each action chooses a feature family, a quantizer and a dropped-cell
fraction q. The presentation asks which action preserves useful perception
while producing a payload that the current channel can deliver freshly.

## Figure 01 — payload versus validation quality

- The x-axis is logarithmic and uses explicit byte units: 10 KiB to 100 KiB is
  a tenfold increase, as is 100 KiB to 1 MiB. Equal horizontal distance means
  an equal multiplicative change, not an equal number of bytes.
- Vehicle F1, person F1 and segmentation mIoU are three different tasks.
- Segmentation mIoU is the arithmetic mean of the **vehicle pixel-class IoU**
  and **person pixel-class IoU**. It is not the average of the vehicle-detection
  IoU and person box-mask IoU shown in Figure 02.
- Therefore a high-q action near 100 KiB can have vehicle F1 around 0.8,
  person F1 around 0.5 and segmentation mIoU around 0.4 without contradiction.
- Agent connection: these curves provide the quality consequence of each
  action. Low bit width is usually cheap; extreme q buys payload at a real
  person/segmentation cost.

## Figure 02 — payload versus localization quality

- XY MAE is the mean planar Euclidean distance between a matched prediction's
  world-XY centroid and the corresponding ground-truth centroid. Matching is
  class-specific and limited to 3 m; lower is better.
- Vehicle IoU and person box-mask IoU are aggregate semantic pixel-overlap
  scores, not centroid distances and not per-object detection-box IoU. The
  person ground truth is a filled projected box rather than a silhouette.
- Person box-mask IoU genuinely peaks at 0.528 across the 72 actions. Thin,
  small person regions make pixel overlap more sensitive to boundary errors,
  and the filled-box target also limits how a predicted person silhouette can
  overlap it. The registered service threshold was 0.50, not 0.60.
- Agent connection: quality reward should not use F1 alone. A small, deliverable
  action can still incur localization or mask-quality debt.

## Figure 03 — q sweep

- Solid lines are feature payload; dashed lines are person F1. The legend now
  states both line meanings explicitly.
- q is the fraction of ranked feature cells dropped. q=0 keeps all cells and
  bypasses the ranker; q=.98 retains only about two percent.
- Agent connection: q is the strongest dynamic lever. The policy should raise
  it under congestion or stale-map pressure, but avoid permanent high-q use.

## Figure 04 — payload versus map-install rate

- Map-install rate means authoritative `MAP_INSTALLED` acknowledgements divided
  by split frames sent by the UE.
- It proves the complete feature was processed, its compact result reached the
  map service, and installation was acknowledged. It does **not** mean the
  update met the separate 100 ms reference.
- The best favorable-stable action is still below 70%. This campaign used the
  pre-optimization edge path: radio loss, fragmented-message loss, latest-frame
  replacement and an edge service slower than the 100 ms arrival interval all
  reduce installations.
- Agent connection: this is the empirical action-conditional probability of
  obtaining a usable map update under each channel state.

## Figure 05 — feature delivery

- UDP did not retransmit. “UDP datagrams received / sent” is a reception
  fraction: denominator is the UE's original datagrams and numerator is what
  the edge observed.
- A feature message is useful only when all of its fragments are reassembled.
  A modest datagram loss therefore hurts large multi-datagram actions much
  more than a one-datagram action.
- Agent connection: payload size is not just an airtime cost; it changes the
  probability of complete delivery.

## Figure 06 — action × network heatmap

- Each row is one action and each column is one network process. The color is
  map-install rate.
- The structured change across columns is the reason for adaptive selection:
  the best action is conditional on channel history rather than globally fixed.

## Figure 07 — network summary

- Campaign-wide map-install rate falls from 0.451
  in favorable stable to 0.270 in adverse
  stable.
- Zero-delivery actions rise from 8
  to 29 of 72.
- Agent connection: network state changes the feasible action set, but a
  zero-delivery observation is a measured bad outcome—not a corrupt action.

## Figure 08 — payload versus total installed-map latency

- This is the registered end-to-end AoI from camera capture to authoritative
  map installation. It includes sensor/preparation delay, UE dispatch, radio,
  edge queue/service, compact result delivery and map installation.
- Only installed frames have an AoI. A zero-delivery action has no point and
  must receive no quality credit merely because its hypothetical output is
  accurate.
- No installed update met 100 ms. The 100 ms line is a reference borrowed from
  stringent teleoperation practice, not a claim that this full perception and
  map pipeline already meets it.
- Agent connection: AoI/map freshness—not an isolated transport timer—is the
  direct state and reward quantity.

## Figure 09 — original four-action latency breakdown

- Sensor preparation compute ends before UE split dispatch begins.
- UE split dispatch includes seven-channel input assembly, FCOS front, ranker,
  optional AE encoding, quantization, zstd and chunk preparation.
- Feature uplink is the same-clock application interval from first UE datagram
  send to complete edge reassembly through OAI.
- Edge queue is waiting after reassembly. Feature reconstruction is zstd
  decompression + unpack/dequantization + optional AE decode.
- FCOS tail inference is only the model's tail launch/completion. Postprocessing
  and p025 filtering are separate.
- The old 111.6 ms “FCOS tail service span” was a misleading name for the whole
  frozen tail adapter: model tail + camera-aware postprocessing + p025 filtering
  + segmentation construction. Pure tail inference was about 21 ms by CUDA
  events and 27–29 ms by wall timing.
- The stack is a descriptive sum of per-stage medians; medians from different
  frames are not mathematically additive. Figure 08 is the authoritative E2E
  latency.

## Figure 10 — optimization result

- Before and after now use the same edge-stage categories, colors and action
  ordering as Figure 09. Upstream sensor preparation, UE dispatch and OAI
  transport are excluded because the optimization did not change them.
- Output-preserving optimization reduced direct edge service by
  38.4–45.0 ms across all four actions, with a
  mean saving of 42.2 ms.
- The direct-service saving is worker start to result publication. Edge queue
  is shown as a system consequence but is not included in that direct-service
  number.
- Perception tensors, p025 selections, segmentation labels and serialized
  records remained exact. Feature payloads and the radio bridge were unchanged.
- Camera-aware postprocessing originally decoded geometry for candidates that
  NMS later discarded; it now performs identical score/box/NMS decisions first
  and computes geometry only for survivors. Serialization originally caused
  repeated device-to-host scalar synchronizations; it now makes one aligned
  tensor transfer before constructing the same records.
- The p025 stage builds a person semantic mask, finds connected components,
  associates person boxes with them, consolidates duplicate person candidates,
  applies the locked 0.25 person threshold and calibrates vehicle scores. Its
  own saving was modest; most of the improvement came from postprocessing and
  serialization.
- Do not subtract a constant from every historical AoI. Faster service changes
  queue replacement nonlinearly. The RL simulator should retain the 288-cell
  radio/delivery evidence and replay it with the optimized service-time
  distributions.

## Transition to the agent discussion

The measurements establish three coupled consequences of an action:

1. perception and localization quality if an update succeeds;
2. payload-dependent delivery probability under the current channel;
3. resulting map freshness after queueing and processing.

That motivates a recurrent policy that observes recent channel/delivery/map
state and selects family, quantizer and q to maximize useful fresh-map utility,
not simply accuracy and not simply minimum payload.

## Compact split-action reward

Use the following display-math form in the presentation:

$$
r_t = I_t\,Q(a_t)\,\exp\!\left(-\frac{\operatorname{AoI}_t}{\tau}\right)
      - \lambda_B\frac{B(a_t)}{B_{\max}}
$$

- $I_t$ is 1 when the selected split update is installed in the spatial map
  and 0 otherwise. An undelivered prediction therefore receives no perception
  utility.
- $Q(a_t)$ is the normalized perception/localization quality associated with
  the selected action.
- $\exp(-\operatorname{AoI}_t/\tau)$ is a smooth freshness discount. It is
  1 for a new update, about 0.368 when AoI equals $\tau$, and about 0.135 at
  twice $\tau$. A smaller $\tau$ represents a freshness-sensitive application;
  a larger $\tau$ tolerates older map information.
- $B(a_t)$ is the selected action's feature payload. $B_{\max}$ is a fixed
  normalization constant, not the instantaneous network capacity. For this
  catalog it is the largest registered median feature payload: 3,580,215 bytes
  (about 3.41 MiB, action 0). Consequently $B(a_t)/B_{\max}$ is dimensionless
  and lies in approximately $[0,1]$.
- $\lambda_B$ controls how strongly the agent trades perception utility for
  lower communication cost.

Speaker summary: an action is valuable only if it produces an installed update;
its value then decreases smoothly as that update becomes older, while larger
feature payloads pay an explicit communication penalty.

## Figure 11 — final live edge optimization

- Both bars use the same predicted-install scheduler and fresh live CARLA/OAI lifecycles; only the edge implementation changes.
- Median edge service fell by 6.1–8.6 ms (7.7–14.8%). This is a real, repeatable direction of improvement, but it is below the hoped-for 10–20 ms.
- The changes overlap independent CPU/GPU work, remove one duplicate full-tensor finite scan, and avoid an immediate JSON serialize/parse cycle.
- Pure FCOS `decode_tail` remains about 21 ms. The displayed edge service ends only after the compact result has been sent.
- Agent connection: the optimized service distribution belongs in the simulator. Do not subtract one constant from all 288 historical samples.

| Action | v2 edge (ms) | v3 edge (ms) | Saving | v2 E2E AoI (ms) | v3 E2E AoI (ms) |
|---:|---:|---:|---:|---:|---:|
| 30 | 79.1 | 73.0 | 6.1 (7.7%) | 288.6 | 286.1 |
| 50 | 64.6 | 58.3 | 6.3 (9.8%) | 211.9 | 202.1 |
| 71 | 57.8 | 49.2 | 8.6 (14.8%) | 170.3 | 163.2 |

## Figure 12 — final live latency intervals

- The six causal intervals are shown separately, followed by the authoritative capture-to-install AoI. They are not stacked because medians from different installed frames do not add exactly.
- Feature uplink includes the live OAI application path from first UE datagram send to complete edge reassembly. It is unaffected by the edge code change in design; small run-to-run differences are expected.
- Edge queue medians are near zero under the predicted-install/latest-only scheduler. The remaining latency is distributed across UE preparation/dispatch, radio transfer, edge service and result installation.
- One action-71 frame installed within 100 ms in the v3 run, but a single event is not 100-ms service qualification.
- Action 15 is deliberately absent: two v3 live attempts failed closed after a non-finite camera-aware geometry output. It is a numerical reliability finding, not usable latency evidence, and must not be hidden or averaged into the successful actions.

## Honest meeting conclusion

The complete optimization program substantially reduced the original edge bottleneck, and the final pass adds another measured 6–9 ms. End-to-end latency is still mostly 160–290 ms for these three examples, so the system is suitable for cooperative map awareness and early warning—not as the sole hard real-time emergency-braking authority. The recurrent policy should optimize installed-map utility and freshness under this measured frontier.

Action-15 retained failure statement: `DiagnosticError: the instrumented edge exited during capture: {"failures": ["pipeline offer: ValueError: pipeline is not accepting work", "pipeline drain: PipelineWorkerError: pipeline worker failed"], "pipeline_fatal_error": "RuntimeError: v3 camera-aware postprocess output contains non-finite values", "pipeline_terminal_reason_counts": {"PREDICTED_MAP_INSTALL_HORIZON_EXCEEDED": 1, "PROCESSING_FAILED": 1, "RESULT_PUBLISHED": 66}, "terminal_reason": "STOP_REQUESTED"}`.
