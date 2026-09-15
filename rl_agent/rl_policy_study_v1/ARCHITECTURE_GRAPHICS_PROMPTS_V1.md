# Conference-figure generation prompts v1

These are the two requested prompts only. They deliberately leave numerical
latency callouts editable until the revised latency-operation boundary is
agreed. Any generated figure must be checked against the locked architecture;
image-generation text and tensor dimensions are not authoritative evidence.

## Prompt 1 — SplitFusion-FCOS model

Create a publication-quality, wide landscape vector diagram for a MobiSys/IEEE
conference paper titled **“SplitFusion-FCOS: RGB–Radar Split Inference for
Cooperative Perception.”** Use a clean white background, restrained navy,
teal and orange palette, consistent thin arrows, readable sans-serif type,
small professional icons and tensor-grid illustrations—not a page of generic
rectangular boxes. Use flat vector artwork, no photorealism, no 3-D perspective,
no decorative AI brain, no gradients that reduce readability, and no invented
metrics.

Show the following left-to-right scientific data path:

1. A connected vehicle containing an RGB camera icon and a forward radar fan
   with sparse points. Draw RGB as three aligned planes labelled R, G, B with
   content shape `3 × 432 × 768`.
2. Show **radar rasterization** visually: sparse altitude/azimuth/depth/radial-
   velocity returns are projected into the camera plane and painted into four
   aligned grids labelled **occupancy**, **inverse range**, **radial velocity**
   and **stationary age**, shape `4 × 432 × 768`. Do not label any radar plane
   “intensity.” Add a short note: “sparse points → dense camera-aligned planes.”
3. Show channel concatenation of RGB(3) and radar(4) into one fused tensor
   `7 × 432 × 768`, followed by 16 zero-valued bottom-padding rows to form
   `7 × 448 × 768`.
4. Inside a clearly shaded **UE / vehicle front** region, show the seven-channel
   ResNet-50 stem and C2 front. Illustrate the fused C2 tensor at the split
   boundary with shape `256 × 112 × 192`.
5. Attach an integrated **adaptive compression treatment** directly to C2—not
   as an unrelated subsystem. Show its three selectable knobs: representation
   `{noAE, AE128, AE64, AE32}`, quantization `{UINT8, UINT6, UINT4}`, and spatial
   drop fraction `q ∈ {0, .30, .50, .70, .90, .98}`. Label the combination as
   “72 registered split actions.”
6. Send the encoded feature through an OAI 5G uplink icon with UE, gNB and edge
   server symbols. Mark this as the single learned-feature network boundary.
7. Inside an **edge / model tail** region, show reconstruction/dequantization,
   ResNet C3–C5, and an FPN staircase P2–P7. From the pyramid, show two visually
   distinct heads: (a) a semantic decoder producing a colored background /
   vehicle / person mask; and (b) FCOS class + centerness + 2-D box feeding a
   depth/ray/geometry head that outputs class, confidence, box, world XYZ,
   dimensions and yaw.
8. Send compact object records directly from the edge to a spatial-map icon.
   Show record-free terminal/reward metadata returning to the UE on a thin
   dashed feedback arrow; do not show the object records travelling back over
   the radio.
9. Add four small editable latency brackets beneath the main path labelled
   “UE action path,” “5G feature uplink,” “edge service,” and “map install.” Use
   placeholder `— ms` rather than inventing numbers.

Make the split point, tensor dimensions and data/control arrows immediately
legible at presentation scale. Ensure all spelling is exact.

## Prompt 2 — recurrent PPO agent and closed loop

Create a publication-quality, wide landscape vector architecture figure titled
**“Recurrent PPO Control Loop for Adaptive SplitFusion.”** Match the visual
style of a top systems/ML conference: white background, crisp flat icons,
restrained navy/teal/orange/magenta palette, tensor grids and signals, minimal
but readable text. Do not use only boxes, do not use a decorative brain icon,
and do not invent latency values or claim training results.

Organize the figure into three horizontal lanes: **perception data path**,
**causal policy/control path**, and **delayed outcome/credit path**.

Perception lane:

- Show camera RGB planes `3 × 432 × 768` and the four camera-aligned radar
  raster planes—occupancy, inverse range, radial velocity, stationary age—
  concatenated into the fused seven-channel tensor, padded to `7 × 448 × 768`.
- Feed the tensor through the UE model front to C2. Place the adaptive action
  knobs on the C2 compression module: representation `{noAE, AE128, AE64,
  AE32}`, quantizer `{8, 6, 4 bits}`, and registered q/drop fraction. Then show
  OAI 5G uplink → edge reconstruction → FCOS/FPN tail → direct spatial-map
  installation.
- Do not draw the full raw seven-channel tensor entering PPO directly. If scene
  context is shown, route only a compact **causal scene/risk summary** through
  an explicitly labelled small encoder; raw perception remains the model input.

Policy/control lane:

- Draw telemetry entering from an **OAI UE-softmodem** icon. Separate PHY
  signals—PUSCH SNR, UL MCS, and qualified HARQ/delivery history—from MAC
  signals—BSR/UL buffer occupancy, grants or delivered throughput.
- Add service-state inputs: previous action and payload, pending ticket
  age/count, latest installed-frame lag, time since last install, recent
  supersession/failure outcomes, and optional compact scene/risk summary.
- Feed normalized inputs and availability flags into `LayerNorm + MLP
  observation encoder`, then into an LSTM cell. Show both recurrent arrows:
  hidden state `h_t` as the exposed short-term summary and cell state `c_t` as
  longer-lived channel/service memory.
- Fan `h_t` into four heads: (1) PPO policy head with **72 discrete action
  logits**, (2) value head estimating expected return, (3) cost critics for
  bytes/compute/switching, and (4) an auxiliary head predicting next observed
  SNR over roughly one decision interval. Make clear that the prediction trains
  the representation for the next epoch; future SNR is never leaked into the
  current action.
- Route the sampled action back to the C2 compression module. Label q as one of
  six registered discrete anchors for the initial policy; place “continuous q:
  future extension after evidence” in a small muted note.

Delayed outcome/credit lane:

- Show an exact pending-transition ledger keyed by `(session, UE, frame,
  action)`. Include a small two-frame inset: frame N+1 may arrive while frame N
  is still `PENDING`; missing immediate feedback is not labelled lost.
- From the edge tail/direct map installation, return a compact feedback packet
  containing terminal outcome, installed frame ID, action ID, segmentation
  quality, localization quality, action-to-map latency, bytes/compute and an
  optional replacement frame ID for `SUPERSEDED_PENDING`.
- Show that out-of-order feedback closes the matching ticket, produces reward
  for that exact action, and updates the next observation using installed-frame
  lag. Show proven reassembly failure separately from intentional supersession.
- Include this compact draft reward beneath the ledger:
  `r_j = I_install [w_S S_j + w_G G_j − w_T(1 − exp(−T_j/τ_T))] − proven-loss penalty − optional resource costs`.
  Add a note: “frame ID provides attribution and lag state; raw frame ID is not
  a scalar reward.”

Use solid arrows for feature/data flow, dashed arrows for telemetry and delayed
feedback, and a distinct recurrent-loop arrow for LSTM memory. Make the causal
ordering understandable without requiring the caption.
