# Static Split-Inference Control Knobs — presentation talking points (6 slides)
*Month-2 deliverable. Model = M' (ROI-robust fusion model). Numbers from COMPLETE_KNOB_MATRIX.md
(27 profiles) under IDEAL transport.*

---
## Slide 1 — Why compress? The split-inference bottleneck
- **Setup:** the perception model is split — the UE (vehicle) runs the front half, sends intermediate
  **features** over the 5G link, the edge runs the back half and returns detections/segmentation.
- **The problem:** the raw feature tensor is **~2,835 KB/frame** (960-ch "high" + 40-ch "low", fp16).
  At video frame rates that is far beyond a shared radio link — and we showed a bounded UDP buffer already
  drops >85% of ~1 MB frames. **Naive split inference does not survive the network.**
- **The goal (North Star):** a **network-aware controller** that shrinks payload/latency **while preserving
  task utility and safety-critical recall** (pedestrians). Month-2 = characterize the control knobs offline.
- **Speaker note:** everything downstream is about *what can we trade, and how much does it cost.*

---
## Slide 2 — The four control knobs
| knob | what it is | how it affects operation |
|---|---|---|
| **Quantization** | encode each feature channel at 8/6/4 bits instead of fp16 | uniform payload cut; lossy only at low bit-depth. Cheapest lever. |
| **Entropy coder** | lossless compression of the quantized bytes (zlib / zstd / none) | shrinks payload with **zero** accuracy change; only differs in speed. |
| **ROI drop** | zero the lowest-**objectness** fraction *q* of feature cells before sending (rank-based, task-aware) | keeps the cells around objects, drops "background" — payload ↓, **segmentation** ↓. |
| **Feature AE** | a learned autoencoder compresses the 960-ch feature to a small **bottleneck** (128/64/32) | most aggressive payload cut; preserves scene gist, costs **object detection** precision. |
- All four operate at the **split point** on the transmitted features — they do **not** change the model.
- The RL agent will pick a *combination* of these per frame.

---
## Slide 3 — The action space: 180 possible, 27 that matter
- Full grid = quant(3) × entropy(3) × ROI(5) × AE(4) = **180 combinations** — intractable and mostly redundant.
- We characterize **27 meaningful profiles**: each knob swept one-axis-at-a-time + 3 baselines
  (accuracy ceiling, uncompressed transmit, lossless-transmit) + **combined** actions (quant×ROI, AE×quant, AE×ROI).
- Why this is enough (and how it prunes for RL):
  - **Entropy is dominated** → always pick **zstd** (equal accuracy, ≤ payload, lower latency). Collapses the axis.
  - One-axis sweeps reveal each knob's cost curve; combined probes check they compose.
  - The RL **action set** is then the ~8–10 non-dominated points on the frontier, not 180.
- **Speaker note:** we don't need the full grid — the agent *learns* the (state→action) mapping; the matrix is
  the offline **action-cost oracle** we characterize it against.

---
## Slide 4 — The knob matrix (accuracy · payload · latency)
*Paste the final table from COMPLETE_KNOB_MATRIX.md. Baseline: M' clean = mIoU 0.841 / veh-IoU 0.933 /
ped-recall 0.787 / loc 1.21 m. Payload % is of the 2,835 KB uncompressed transmit. Ideal transport (8 MB
buffers, no bandwidth cap) → delivery 1.0; latency = front (UE compute) / back (edge) / transport.*

Representative rows:
| profile | payload % | mIoU | ped recall | loc m | front/back/transport ms |
|---|--:|--:|--:|--:|--:|
| uncompressed_fp16 (baseline) | 100% | 0.841 | 0.787 | 1.21 | 30 / 8 / 7 |
| quant u6 (lossless) | 26% | 0.841 | 0.790 | 1.22 | 31 / 11 / 6 |
| quant u4 | 13% | 0.840 | 0.755 | 1.32 | 28 / 9 / 5 |
| ROI 0.3 | 25% | 0.811 | 0.790 | 1.21 | 31 / 8 / 7 |
| ROI 0.7 | 13% | 0.779 | 0.778 | 1.30 | 28 / 9 / 5 |
| AE b64 | 9% | 0.836 | 0.552 | 2.15 | 25 / 11 / 2 |
- Delivery = 1.0 everywhere (ideal transport); latency is **compute-dominated**, transport is 2–7 ms.

---
## Slide 5 — What the matrix tells us (the key insight)
- **The knobs are COMPLEMENTARY, not redundant — this is why a learned policy matters:**
  - **Quant** = uniform, near-free (u6 lossless @ 26%; u4 mild cost @ 13%).
  - **ROI** preserves **detection** (recall/loc) and sacrifices **segmentation** — "keep objects, drop background."
  - **AE** preserves **segmentation** and sacrifices **detection** — "keep the scene, blur object precision."
- **No single static action wins** — the best choice depends on **what the current scene needs**:
  pedestrian-heavy frame → quant/ROI (protect recall); seg-dominated, pedestrian-free frame under bandwidth
  pressure → AE at **9% payload** is a steal.
- **Entropy coding is free but modest on raw features** — lossless zstd on raw fp16 only reaches 78% payload;
  **quantization is the primary payload lever** (uint8 → 35%), and the two compound. Always zstd (also lowest
  latency: 15 ms vs 41 ms RTT).
- **Knobs compose — but not all combinations help:**
  - **AE × quant is a big win:** quantizing the *learned* bottleneck to 4-bit is nearly free (recall 0.55→0.54)
    and **halves** AE payload → **3% (33× compression)** — the floor of the action set.
  - **quant × ROI** stacks gracefully (payload multiplies down, costs roughly add).
  - **AE × ROI is dominated** — ROI adds seg cost with no recall benefit (and high ROI collapses recall). The
    agent should not combine them.
- **Under ideal transport, payload barely moves latency** (transport 2–7 ms, compute-bound). So payload's real
  price is **reliability under a constrained channel** — which is Month-3.
- **Safety is a hard constraint:** AE's recall (0.55) is below a pedestrian floor → must be **guardrailed off**
  when vulnerable objects are present. The matrix gives the exact recall to set that floor.

---
## Slide 6 — Conclusion & next: from static costs to a network-aware policy
- **Done (Month-2):** one ROI-robust model M' + a complete, characterized action set
  (**quant × entropy × ROI × AE**) over **accuracy, payload, latency** — the offline action-cost table.
- **This directly seeds the RL formulation:**
  - **State:** scene semantics (vulnerable-object presence, foreground fraction) to choose ROI-vs-AE +
    channel state to choose aggressiveness.
  - **Action:** the non-dominated profile set (pruned via zstd + Pareto).
  - **Reward:** task utility (pedestrian recall weighted heavily) − payload/latency/reliability cost − guardrail.
  - **Guardrail:** pedestrian-recall floor blocks AE / aggressive ROI under vulnerable-object presence.
- **Next (Month-3): network stress with OAI + Sionna.** Replace the ideal transport with a **real ray-traced
  5G channel** (OpenStreetMap geometry) — time-varying bandwidth, RF loss, latency. That supplies the
  **dynamics** that make adaptation *necessary*, and the reliability/latency columns the ideal transport can't.
- **One-liner:** *we've measured what each compression action costs; next we put it under a real channel and
  let the agent learn when to spend it.*
