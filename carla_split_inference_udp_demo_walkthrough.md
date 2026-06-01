# `carla_split_inference_udp_demo.py` — Step-by-Step Implementation Walkthrough

> **Context:** This document walks through the complete split inferencing and feature compression pipeline implemented in `carla_split_inference_udp_demo.py`. It covers every stage from CARLA camera frame acquisition through to the final detection overlay. A detailed comparison with the `ail-demo-mwc2025-main` implementation follows at the end.

---

## Overview of the Architecture

The script is a **single self-contained Python file** that implements a full split inference loop. The model is **Faster R-CNN with a MobileNetV3-Large + FPN backbone** (from torchvision). The split point is placed after the FPN backbone — the first half extracts multi-scale feature maps and transmits them over UDP; the second half runs the RPN, ROI heads, and postprocessing to produce detections.

Unlike a true distributed deployment, both halves run in the **same Python process on the same machine**, communicating via localhost UDP sockets. This is intentional — the localhost UDP channel faithfully simulates a real network link, so payload size and round-trip latency measurements are meaningful and transferable to a real edge-to-server deployment.

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          Single Python Process                             │
│                                                                            │
│  CARLA World                                                               │
│     │                                                                      │
│     ▼                                                                      │
│  RGB Camera → image_queue → CameraSideSplitInference (Thread: main)        │
│                                   │                                        │
│                                   │  localhost UDP :36000 → :36001         │
│                                   ▼                                        │
│                            RemoteInferenceWorker (daemon thread)           │
│                                   │                                        │
│                                   │  localhost UDP :36002 → :36003         │
│                                   ▼                                        │
│                            CameraResultReceiver (daemon thread)            │
│                                   │                                        │
│                                   ▼                                        │
│                            DetectionResultStore → draw_overlay → cv2 window│
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: CARLA World Setup and Scene Population

**Relevant functions:** `run_demo()`, `spawn_hero_vehicle()`, `spawn_background_traffic()`, `spawn_background_pedestrians()`

Before any inference begins, the script loads a CARLA town (default: `Town10HD_Opt`) and populates it:

- A **hero vehicle** (default: `vehicle.lincoln.mkz_2017`) is spawned at a random spawn point with autopilot enabled via the CARLA Traffic Manager.
- Up to 20 **background NPC vehicles** are spawned with autopilot.
- Up to 30 **background pedestrians** are spawned with AI walker controllers that navigate randomly.

The world is switched to **synchronous mode** at a fixed `--fps` (default 10 Hz), meaning the script controls when each simulation tick occurs by calling `world.tick()`. This ensures that camera frames and world state are deterministically aligned.

```python
# run_demo() — synchronous mode setup
settings.synchronous_mode = True
settings.fixed_delta_seconds = 1.0 / args.fps
world.apply_settings(settings)
traffic_manager.set_synchronous_mode(True)
```

---

## Step 2: RGB Camera Attachment and Frame Acquisition

**Relevant functions:** `camera_image_to_bgr()`, `warmup_camera_stream()`, `put_latest()`

An RGB camera sensor is attached to the front of the hero vehicle:

```python
camera_bp = world.get_blueprint_library().find("sensor.camera.rgb")
camera_bp.set_attribute("image_size_x", str(camera_width))   # default 640
camera_bp.set_attribute("image_size_y", str(camera_height))  # default 384
camera_bp.set_attribute("fov", str(args.camera_fov))         # default 90°
camera_bp.set_attribute("sensor_tick", str(1.0 / args.fps))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=hero_vehicle)
camera.listen(lambda image: put_latest(image_queue, image))
```

The camera's callback `put_latest()` pushes the latest frame into a **maxsize=2 queue**, discarding stale frames if the inference loop is running behind — this prevents an ever-growing backlog. Each frame is a CARLA `Image` object containing raw BGRA bytes.

**Conversion to BGR numpy array:**

```python
def camera_image_to_bgr(image: carla.Image) -> np.ndarray:
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))  # BGRA
    return np.ascontiguousarray(array[:, :, :3])            # drop alpha → BGR
```

The result is a `[H, W, 3]` uint8 array in BGR channel order (OpenCV convention).

---

## Step 3: Image Tensor Construction

**Relevant class:** `CameraSideSplitInference.process()`

The BGR frame is converted to a normalized float32 image tensor:

```python
rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)   # BGR → RGB

image_tensor = (
    torch.from_numpy(rgb)
    .permute(2, 0, 1)              # [H, W, 3] → [3, H, W]
    .to(device=self.device, dtype=torch.float32)
    / 255.0                        # scale to [0.0, 1.0]
)
```

This gives an image tensor of shape **`[3, H, W]`**, float32, range [0, 1].

The tensor is then passed through **Faster R-CNN's built-in `transform` layer**, which applies ImageNet mean/std normalisation and pads the spatial dimensions to the nearest multiple of 32 (required by the FPN architecture):

```python
image_list, _ = self.model.transform([image_tensor], None)
# image_list.tensors: [1, 3, H_padded, W_padded], float32, normalised
# image_list.image_sizes: [(H_orig, W_orig)]  — kept for postprocessing
```

For a 640×384 input, the padded tensor is `[1, 3, 384, 640]` (already a multiple of 32 in both axes).

---

## Step 4: First Half of the Model — FPN Feature Extraction (The Split Point)

**Relevant class:** `CameraSideSplitInference.process()`

The padded image tensor is forwarded through the **MobileNetV3-Large + FPN backbone**:

```python
features = self.model.backbone(image_list.tensors)
# features: OrderedDict with keys "0", "1", "2", "3", "pool"
```

This is the **split point**. The backbone returns a Python `OrderedDict` of five feature map tensors at different spatial scales, each capturing different levels of abstraction:

| Key | Spatial Size (for 640×384 input) | Stride | Captures |
|-----|----------------------------------|--------|----------|
| `"0"` | ~80×48 | 8× | Fine details — edges, textures |
| `"1"` | ~40×24 | 16× | Mid-level features — parts, shapes |
| `"2"` | ~20×12 | 32× | High-level — object-like structures |
| `"3"` | ~10×6 | 64× | Very abstract, large receptive field |
| `"pool"` | ~5×3 | 128× | Global context |

Each feature map has **256 channels** (standard FPN output width). For a car occupying the lower-centre of the 640×384 frame, its spatial signature will be strongest at scales `"1"` and `"2"`. Scale `"0"` will carry fine edge details of the car's body; `"pool"` will have near-zero spatial resolution but high semantic content.

> **This is the core of split inference:** everything up to and including `self.model.backbone()` runs on the camera (front) side. Everything after — RPN, ROI heads, postprocessing — runs on the remote (back) side.

---

## Step 5: Per-Scale Feature Compression

**Relevant classes:** `SimpleFeatureCodec`, `FeatureFramePacker`, `RangeTracker`
**Relevant functions:** `serialize_feature_maps()`, `quantize()`, `tensor_to_tiled()`, `symmetric_feature_channel_flipping()`

Each of the five FPN feature maps is compressed independently. A separate `SimpleFeatureCodec` instance (with its own `RangeTracker`) is created and cached for each FPN scale key.

### Sub-step 5a — Range Tracking (EMA)

Before quantization, the codec must know the dynamic range of the feature values so it can map them to [0, 255]. Rather than using the exact per-frame min/max (which fluctuates wildly and would cause quantization instability), the script maintains a **per-scale exponential moving average (EMA) of the running min/max**:

```python
class RangeTracker:
    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self._min = float("inf")
        self._max = float("-inf")

    def update(self, current_min, current_max):
        alpha = self.alpha
        self._min = alpha * current_min + (1.0 - alpha) * min(self._min, current_min)
        self._max = alpha * current_max + (1.0 - alpha) * max(self._max, current_max)
        return self._min, self._max
```

With `alpha=0.1`, this tracker **slowly drifts toward the current frame's range** while preserving memory of the historical extremes. The tracked `rmin` and `rmax` are stored in a compact binary `TensorInfo` struct (16 bytes) and sent alongside the feature data so the server can reverse the mapping exactly.

This is why the script has a **30-frame warm-up period** (`--metrics-warmup-frames 30`) before logging metrics: during the first ~30 frames, the range trackers are still stabilising from their `±inf` initial state. Quantization quality improves as the trackers converge.

### Sub-step 5b — Quantization (float32 → uint8)

```python
def quantize(x, *, min, max, bitdepth=8):
    span = float(max - min)
    max_level = (2**bitdepth) - 1       # = 255 for 8-bit
    x = ((x - min) / span).clip(0.0, 1.0)
    x = (x * max_level).round()
    return x.to(torch.uint8)
```

Each float32 value in the feature tensor is linearly mapped to the integer range [0, 255]. This reduces the memory footprint by **4×** (float32 → uint8). There is an irreversible quantization error of approximately `±(max−min)/510`, which the detection model is trained to tolerate.

### Sub-step 5c — Channel Tiling (3D → 2D)

The quantized tensor has shape `[1, 256, H_feat, W_feat]`. It cannot be directly transmitted as-is to a UDP socket without further structuring. The tiling step reshapes it into a single 2D image:

```python
def compute_frame_resolution(shape):
    channels, height, width = shape      # e.g. (256, 48, 80)
    short_edge = int(math.sqrt(channels)) # = 16
    while channels % short_edge != 0:
        short_edge -= 1
    long_edge = channels // short_edge   # = 16
    # tiled: (16×48) × (16×80) = 768×1280
    return height_edge * height, width_edge * width
```

For FPN scale `"0"` with shape `[256, 48, 80]`, the 256 channel patches (each 48×80) are arranged in a 16×16 grid, producing a 2D image of **768×1280 pixels**. This is now a standard uint8 2D array that can be efficiently handled as bytes.

### Sub-step 5d — Symmetric Channel Flipping

```python
def symmetric_feature_channel_flipping(x, channel_resolution):
    # Alternating columns of channel-patches are horizontally flipped.
    # Alternating rows of channel-patches are vertically flipped.
    x[..., :, :, 1::2, :] = x[..., :, :, 1::2, :].flip(-1)
    x[..., 1::2, :, :, :] = x[..., 1::2, :, :, :].flip(-3)
    return x
```

This is a reversible spatial transformation applied to the tiled image. By flipping alternate tile columns horizontally and alternate tile rows vertically, neighbouring tiles in the grid become more similar to each other (because they represent spatially adjacent feature channels). This improves the effectiveness of the subsequent zlib compression. The operation is its own inverse — applying it twice returns the original data.

### Sub-step 5e — Padding to Even Dimensions

```python
pad_h, pad_w = compute_padding_2d(frame_resolution, opts.frame_div)  # frame_div=2
x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0)
```

The tiled image is zero-padded so that both dimensions are multiples of 2 (`frame_div=2`). This ensures correct byte alignment for downstream byte-level operations.

### Output of compression

Each FPN scale produces a `FeatureCodecPacket` containing:
- `feature_frame`: a uint8 numpy array (the packed 2D tiled image)
- `tensor_info_bytes`: a 16-byte binary struct encoding `(channels, height, width, rmin, rmax)`

---

## Step 6: Payload Assembly and UDP Transmission

**Relevant classes:** `UDPMessageSocket`
**Relevant function:** `serialize_feature_maps()`

All five compressed FPN feature maps are assembled into a single Python dict payload:

```python
payload = {
    "frame_id": frame_id,
    "batch_shape": (1, 3, H_padded, W_padded),
    "image_sizes": [(H_orig, W_orig)],
    "original_sizes": [(H_camera, W_camera)],
    "features": {
        "0":    {"feature_frame": <bytes>, "tensor_info": <16 bytes>},
        "1":    {"feature_frame": <bytes>, "tensor_info": <16 bytes>},
        "2":    {"feature_frame": <bytes>, "tensor_info": <16 bytes>},
        "3":    {"feature_frame": <bytes>, "tensor_info": <16 bytes>},
        "pool": {"feature_frame": <bytes>, "tensor_info": <16 bytes>},
    },
    "camera_sent_perf": time.perf_counter(),   # for round-trip timing
}
```

This dict is then **serialized using `pickle` and compressed with zlib level 1**:

```python
raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
compressed = zlib.compress(raw, level=1)
```

`zlib level=1` is the fastest compression setting — it provides some compression benefit (especially on the near-zero background regions of the tiled feature images) without adding significant latency.

The compressed blob is then **chunked** into UDP datagrams of at most `--chunk-bytes` (default 60,000 bytes). Each chunk is prefixed with a 8-byte header:

```python
HEADER_STRUCT = struct.Struct("!IHH")
# Fields: message_id (uint32), chunk_index (uint16), total_chunks (uint16)

for chunk_index in range(total_chunks):
    chunk = compressed[start:stop]
    packet = HEADER_STRUCT.pack(message_id, chunk_index, total_chunks) + chunk
    self.socket.sendto(packet, self.remote)  # → localhost :36001
```

The `message_id` counter allows the receiver to reassemble chunks from potentially out-of-order or interleaved datagrams. Stale partial messages (where some chunks never arrived) are discarded after 2 seconds.

The uncompressed float16 baseline size (for comparison in the metrics overlay) is computed as:
```python
payload_bytes_uncompressed += tensor.numel() * np.dtype(np.float16).itemsize
# i.e., number of feature values × 2 bytes each
```

---

## Step 7: Remote Inference Worker — Receiving and Decoding

**Relevant class:** `RemoteInferenceWorker.run()`, `_run_back_half()`

The `RemoteInferenceWorker` is a daemon thread listening on UDP port `:36001`. Its receive loop reassembles chunked datagrams, unpickles and decompresses the payload, then calls `_run_back_half()`.

**Deserialising feature maps:**

```python
def deserialize_feature_maps(serialized, device, batch_size, feature_codecs):
    features = OrderedDict()
    for name, payload in serialized.items():
        codec = _get_or_create_feature_codec(feature_codecs, name, device)
        tensor_info = TensorInfo.from_bytes(payload["tensor_info"])
        frame_shape = compute_packed_frame_shape(
            tensor_info.shape, batch_size=batch_size, frame_div=codec.opts.frame_div
        )
        feature_frame = (
            np.frombuffer(payload["feature_frame"], dtype=np.uint8)
            .reshape(frame_shape).copy()
        )
        features[name] = codec.decode(feature_frame, payload["tensor_info"])
    return features
```

`codec.decode()` runs the inverse pipeline in order:

1. **Strip padding** — remove zero-padded rows/columns added in step 5e
2. **Inverse channel flip** — since `symmetric_feature_channel_flipping` is its own inverse, calling it again restores the original tile order
3. **Untile** — reshape the 2D tiled image back to `[1, 256, H_feat, W_feat]`
4. **Dequantize** — recover float32 values: `float = uint8/255 × (max − min) + min`

The reconstructed features `OrderedDict` has the same keys and approximate shapes as what the encoder produced. The small quantization error is irreversible but acceptable.

---

## Step 8: Second Half of the Model — RPN, ROI Heads, Postprocessing

**Relevant class:** `RemoteInferenceWorker._run_back_half()`

With the reconstructed FPN features in hand, the back half of Faster R-CNN proceeds exactly as in a standard inference pass:

```python
# Reconstruct a dummy ImageList — only the spatial size metadata matters,
# not the actual pixel values (those are never transmitted)
dummy_images = torch.zeros(batch_shape, device=self.device)
image_list = ImageList(dummy_images, image_sizes)

with torch.inference_mode():
    # Step A: Region Proposal Network
    proposals, _ = self.model.rpn(image_list, features, None)
    # proposals: list of [N_proposals, 4] tensors — candidate bounding boxes

    # Step B: ROI heads (ROI Align + classification + bbox regression)
    detections, _ = self.model.roi_heads(
        features, proposals, image_list.image_sizes, None
    )
    # detections: list of dicts with "boxes", "labels", "scores"

    # Step C: Postprocess — rescale boxes back to original image coordinates
    detections = self.model.transform.postprocess(
        detections, image_list.image_sizes, original_sizes
    )
```

Key observations:

- **The dummy ImageList is a deliberate design choice.** The RPN and ROI heads only need the feature maps and the image size metadata (for anchor generation and coordinate rescaling). The actual pixel values are never needed on the server side — only the feature tensors matter.
- **The `original_sizes` metadata** (camera resolution: e.g. 640×384) is transmitted in the payload so that `transform.postprocess` can map the detected bounding boxes back from the padded/normalised coordinate space to the original pixel space.

Detections are then filtered by score threshold and serialised:

```python
keep = np.where(scores >= self.score_threshold)[0][:self.max_detections]
serialized_detections = [
    {
        "box":   boxes[i].round(2).tolist(),   # [x1, y1, x2, y2]
        "score": float(scores[i]),
        "label": int(labels[i]),
        "name":  COCO_LABELS[int(labels[i])],  # e.g. "car", "person"
    }
    for i in keep
]
```

---

## Step 9: Detection Results Returned to Camera Side

The remote worker sends the detection dict back over a second UDP socket (`:36002 → :36003`), using the same `UDPMessageSocket.send()` mechanism (pickle + zlib + chunked UDP). The detection payload is small — a list of `N` dicts each containing a 4-element box, a float score, and an integer label — so it typically fits in a single UDP datagram.

The `CameraResultReceiver` daemon thread listens on `:36003` and deposits each result into the `DetectionResultStore` keyed by `frame_id`.

Back in the main loop:

```python
result = result_store.wait_for(image.frame, args.result_timeout)  # default 0.35s
```

If the result arrives within the timeout, it is used to draw bounding boxes on the current frame. If the remote worker is slower than the camera tick rate (10 Hz = 100ms budget), results from older frames will be displayed with a small visual lag, but the system never blocks the simulation tick.

---

## Step 10: Overlay Rendering and Metrics

**Relevant function:** `draw_overlay()`

The annotated frame shows:
- Bounding boxes with class name and confidence score
- Front-half latency in ms
- Compressed feature payload size in KiB and number of UDP chunks
- Float16 uncompressed baseline size and compression ratio
- Back-half latency and round-trip latency (if available)

The `AsyncMetricsCollector` thread writes per-frame records to a CSV file and optionally streams data to a live matplotlib plot subprocess. At exit, an offline PNG plot of latency, payload, and detection counts is generated.

---

## Complete Data Flow

```
CARLA World
    │
    ▼  world.tick() at 10 Hz
RGB Camera [640×384, BGRA, uint8]
    │
    ▼  drop alpha, np.frombuffer → [H, W, 3] BGR uint8
    │
    ▼  BGR→RGB, permute(2,0,1), /255.0
Image Tensor [3, 384, 640] float32, range [0,1]
    │
    ▼  model.transform()  (normalize + pad)
Batch Tensor [1, 3, 384, 640] float32, ImageNet-normalized
    │
    ▼  model.backbone()  (MobileNetV3-Large + FPN)
FPN Feature Maps  (OrderedDict — 5 scales):
    "0"    [1, 256,  48,  80] float32   ← fine
    "1"    [1, 256,  24,  40] float32
    "2"    [1, 256,  12,  20] float32
    "3"    [1, 256,   6,  10] float32
    "pool" [1, 256,   3,   5] float32   ← coarse
    │
    ▼  Per-scale SimpleFeatureCodec:
    │      RangeTracker EMA → rmin, rmax
    │      Quantize float32 → uint8
    │      Tile 256 channels into 2D image
    │      Symmetric channel flip
    │      Zero-pad to even dims
    │
Packed Feature Frames (5× uint8 2D arrays) + TensorInfo structs (5× 16 bytes)
    │
    ▼  Assemble Python dict payload
    ▼  pickle.dumps + zlib.compress(level=1)
    ▼  Chunk into 60KB UDP datagrams with (msg_id, chunk_idx, total) header
    │
    │  localhost UDP :36000 → :36001
    │
    ▼  Reassemble chunks, zlib.decompress + pickle.loads
Reconstructed FPN Feature Maps  (5 scales, float32, approx.)
    │
    ▼  model.rpn()  (Region Proposal Network)
Proposals  [N×4 bounding box candidates]
    │
    ▼  model.roi_heads()  (ROI Align + classification + bbox regression)
    │
    ▼  model.transform.postprocess()  (rescale to original image coords)
    │
    ▼  Score threshold + max detections filter
Detections [ {box, score, label, name}, ... ]
    │
    ▼  pickle + zlib + UDP :36002 → :36003
    │
    ▼  DetectionResultStore.wait_for(frame_id)
    ▼  draw_overlay() → cv2.imshow()
Annotated Frame on screen
```

---

## Comparison with `ail-demo-mwc2025-main`

The table below compares the two implementations across every major design dimension.

| Dimension | `carla_split_inference_udp_demo.py` | `ail-demo-mwc2025-main` |
|---|---|---|
| **Code organisation** | Single self-contained script (~2,200 lines) | Multi-file modular framework (`models/`, `codecs/`, `pipelines/`, `serialization/`) |
| **Deployment topology** | Both halves in the same process; localhost UDP simulates the link | True client/server separation; designed for real network deployment |
| **Model** | Faster R-CNN + MobileNetV3-Large FPN (torchvision) | YOLOX (primary) or Faster R-CNN via CompressAI Vision / detectron2 |
| **Split point** | After FPN backbone (`model.backbone()`) | After YOLOX backbone or after FPN (`input_to_features()`) |
| **Feature map structure** | 5 FPN scales: "0","1","2","3","pool" — each 256 channels | Single tensor [128, 40, 40] for YOLOX; multi-scale dict for Faster R-CNN |
| **Range tracking** | **EMA `RangeTracker` (alpha=0.1)** — smoothed, stable across frames, per scale | Running min/max updated per-frame — closer to exact per-frame range |
| **Quantization** | float32 → uint8 (8-bit), per FPN scale | float32 → uint8 (8-bit) |
| **Tiling** | 256 channels → 2D grid via `compute_frame_resolution()` (sqrt factorisation) | Same algorithm, same sqrt factorisation |
| **Channel flipping** | Symmetric flip (alternating rows/cols) | Same symmetric flip — identical algorithm |
| **Padding** | `frame_div=2` (pad to even dimensions) | `frame_div=1` (no padding) |
| **Serialization** | **`pickle.dumps` + `zlib.compress(level=1)`** on a Python dict containing all 5 scales | Raw bytes for feature frame + JSON/msgpack for metadata; two separate network messages |
| **Transport** | Raw UDP with custom 8-byte header (msg_id, chunk_idx, total_chunks) | WebSocket / TCP with dedicated channel IDs (`C2S_DATA`, `C2S_MEDIA`) |
| **Transport direction** | Bidirectional — features client→server, detections server→client | Bidirectional — same pattern |
| **Video codec integration** | None — zlib only | Optional H.264/video codec on the feature image before transmission |
| **Metadata transmission** | Embedded in pickled Python dict payload (batch shape, image sizes, original sizes) | Separate `aux_data_bytes` message (JSON with TensorInfo, model metadata) |
| **Back-half inputs** | `features` + `image_sizes` + `original_sizes` → `rpn` + `roi_heads` + `postprocess` | `features` + optional `meta` → YOLOX decode head or `features_to_output()` |
| **Dummy ImageList** | Yes — explicitly constructs `ImageList(torch.zeros(...), image_sizes)` to satisfy torchvision's API | Not needed — YOLOX decode head takes features directly |
| **CARLA integration** | Built-in — spawns vehicles, pedestrians, attaches camera to hero vehicle | None — works with pre-recorded frames or video streams |
| **Metrics & observability** | Full pipeline — CSV logging, live matplotlib plot (subprocess), round-trip timing, compression ratio display | Minimal — no built-in metrics infrastructure |
| **Warm-up period** | Explicit 30-frame warm-up for range tracker stabilisation | Not handled explicitly |
| **Concurrency model** | Python threading — `RemoteInferenceWorker` and `CameraResultReceiver` as daemon threads | asyncio (`async/await`) with executor for blocking inference calls |
| **Frame drop policy** | `image_queue` maxsize=2, oldest frame dropped if queue full | Explicit frame ID joiner with per-channel packet assembly |
| **Target use case** | Proof-of-concept demo with CARLA + latency/payload measurement | Research framework for feature compression quality evaluation (mAP vs. compression) |

---

## Key Architectural Differences — In Depth

### 1. Serialization: `pickle+zlib` vs. structured binary packets

The most significant implementation difference is how the compressed feature data leaves the camera side.

In `carla_split_inference_udp_demo.py`, **all five FPN feature frames plus all metadata** are bundled into a single Python dict, serialized with `pickle`, and zlib-compressed together. This is simple and portable but has several implications:

- The zlib pass operates on the entire blob, so it can exploit redundancy between scales (e.g., similar zero regions across scales). However, pickle adds significant per-object overhead.
- The server must deserialize the entire blob before accessing any single feature map.
- The payload is opaque to any intermediate network equipment (routers, middleboxes) — it cannot be inspected, prioritised, or partially decoded.

In `ail-demo-mwc2025-main`, the feature data is sent as **two separate typed messages**: a compact metadata message (JSON, typically < 1KB) and a raw binary media message (the uint8 feature frame bytes). This separation means:

- The metadata channel can arrive and be parsed immediately, while the (larger) media channel is still in transit.
- The raw binary media packet can be passed directly into a video codec (H.264, HEVC) for additional compression, because it is already a well-structured 2D image.
- The architecture maps naturally to real network protocols (e.g., RTP for media, a control channel for metadata).

### 2. Range Tracking: EMA vs. running min/max

`carla_split_inference_udp_demo.py` uses an **exponential moving average** with `alpha=0.1` for the quantization range. This produces a smoothed, stable range estimate that changes slowly even if a single frame has an unusually large activation (e.g., a sudden bright scene). The cost is that the first ~30 frames have suboptimal quantization while the EMA converges.

`ail-demo-mwc2025-main` tracks a running min/max that can adapt more quickly to scene changes but may over-expand the range if a single outlier frame produces an extreme activation.

Both approaches avoid a per-frame exact min/max (which would waste 2 float32 values and produce an inconsistent quantization grid from frame to frame).

### 3. Concurrency: threading vs. asyncio

`carla_split_inference_udp_demo.py` uses Python's `threading` module. The main thread runs the CARLA simulation loop and the front half; two daemon threads run the back half and result receiver independently. Communication is via UDP sockets and a `threading.Condition`-protected `DetectionResultStore`.

`ail-demo-mwc2025-main` uses `asyncio` with `run_in_executor` for CPU-bound inference calls. This is better suited for a real server that may serve multiple client sessions concurrently.

### 4. Multi-scale FPN vs. single-tensor split

`carla_split_inference_udp_demo.py` must compress and transmit all five FPN scales because Faster R-CNN's RPN and ROI heads require all of them. This means the total payload is the sum of five compressed feature frames.

The YOLOX pipeline in `ail-demo-mwc2025-main` produces a single feature tensor at a fixed split point. This simpler structure requires only one codec, one TensorInfo struct, and one media packet — making the pipeline easier to reason about and the payload smaller, at the cost of using a single spatial scale for all detections.

---

*Generated from `carla_split_inference_udp_demo.py` and the `ail-demo-mwc2025-main` codebase. Key files compared: `CameraSideSplitInference`, `RemoteInferenceWorker`, `SimpleFeatureCodec`, `FeatureFramePacker`, `UDPMessageSocket`, `RangeTracker` vs. `demo/v2/models/yolox_onnx.py`, `demo/v2/codecs/feature.py`, `demo/feature_packing.py`, `demo/v2/codecs/distributed_detection.py`.*
