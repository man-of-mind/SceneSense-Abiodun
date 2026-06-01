# Handoff — Cooperative Perception over OAI 5G

**Audience:** ChatGPT Codex (or any AI agent picking up Abiodun's work in this repo).
**Date frozen:** 2026-05-22.
**User:** Abiodun Ganiyu (IDCC), abiodun.ganiyu@interdigital.com.

---

## TL;DR — where we are mid-task

We are in **Phase 1** of a 3-phase plan to migrate the team's CARLA split-inference Faster R-CNN detection demo onto a real OAI 5G data plane.

- Phase 1 (current): refactor the single-process split-inference demo so the two halves can run as separate processes. **~90% complete.** Verification + run instructions are the remaining work.
- Phase 2 (not started): make the back-half run inside a GPU-enabled docker container that sits on the OAI core network.
- Phase 3 (not started): end-to-end run with front-half on host (UE-bound), back-half in container, metrics CSV comparison vs loopback baseline.

The user is **about to switch from Claude to Codex** (credit-limit driven). Everything below is what Codex needs to continue without me.

---

## Who is the user

- **Abiodun**, recent IDCC team-member on the InterDigital × Northeastern collaboration (project ends **2026-08-29**).
- Research thread: **multi-UE cooperative perception over 5G**, extending **SCAN-AI** (single-UE) to multi-UE feature sharing for safety-critical autonomous driving.
- Engineer-style collaborator: wants concise responses, will say "stop X" or "yes exactly" — Codex should mirror that style.

## Who is everyone else (cross-team context)

- **Supervisor:** Subhramoy (or Dale) — IDCC. Likely the person to ping for strategy or system-install approval. Abiodun typically asks himself.
- **MJ, Michele, Mateo:** other team members. Mateo is an intern on a parallel parking-spot/NLM track in the same repo — don't accidentally edit his stuff.
- The **OAI / AI-RAN Alliance** community is a research-target audience for the eventual paper.

---

## HARD RULES — NEVER VIOLATE

1. **Never edit existing top-level scripts.** The team has files in `PythonAPI/neu_collab/` that they actively use. Always *copy* into `PythonAPI/neu_collab/abiodun/` and edit the copy. Same applies to OAI configs — copy first, edit copy.
2. **GStreamer, not ffmpeg, for SCAN-AI streaming.** We previously chased an ffmpeg/NVENC PoC because GStreamer's x265enc seemed slow on this box. Turned out to be a machine-specific issue (supervisor's machine runs the same GStreamer pipeline cleanly). The ffmpeg detour was removed. The original sender PoC and multi-sensor ffmpeg variant were deleted. **Do not propose ffmpeg again** unless the user explicitly asks for it.
3. **No system-wide installs (apt/sudo) without explicit user/supervisor approval.** This is a **shared service account** (`shr_aisvcs`). One exception currently approved: `nvidia-container-toolkit` install for Phase 2 — user said the machine is reserved for him, supervisor is OK with it.
4. **OAI command syntax quirks:** the rfsimulator config is a libconfig *list*. Always use `--rfsimulator.[0].serveraddr` (with the `[0]` index) — without it nr-uesoftmodem segfaults. Same for `--gNBs.[0].min_rxtxtime`.
5. **Visualization vs metrics for experiments:** the supervisor wants metrics-only (CSV → plot after) for the real Phase 3 runs. But the user wants visualization KEPT for now during refactor / first-time wiring. So both run paths should exist; default to GUI for development, easy switch to headless for actual experiments.

---

## Working directory & repo layout

```
PythonAPI/neu_collab/                         # team's shared code (don't edit)
├── carla_split_inference_udp_demo.py         # ORIGINAL OD demo, 2247 lines (DO NOT EDIT)
├── carla_split_inference_udp_segmentation_demo.py   # ORIGINAL segmentation demo
├── carla_multisensor_udp_gstreamer_*.py      # multi-sensor sender/receiver
└── abiodun/                                  # all of Abiodun's work goes here
    ├── HANDOFF_TO_CODEX.md                   # this file
    ├── notes.md                              # daily session log
    ├── brainstorm_log.md                     # raw idea capture
    ├── research_direction.md                 # MobiSys-target paper-shaped doc
    ├── installation.md                       # V2Xverse install log (currently blocked)
    ├── carla_split_inference_udp_oai.py      # NEW: OAI variant of OD demo, ~2400 lines
    ├── scan_sender_v3_oai.py                 # WORKING: GStreamer sender over 5G
    ├── scan_receiver_v3_oai.py               # WORKING: GStreamer receiver in container
    ├── scan_sender_v2_threaded.py            # earlier threaded SI/TI sender (loopback)
    ├── scan_sender_v2_threaded_lowres_siti.py
    ├── carla_multisensor_udp_opencv_receiver_v2.py
    ├── scripts/                              # OAI helper shell scripts
    │   ├── config.env                        # paths + IPs (single source of truth)
    │   ├── cn_start.sh / cn_stop.sh / cn_status.sh
    │   ├── gnb_start.sh
    │   ├── ue_start.sh
    │   ├── ue_check.sh
    │   ├── iperf2_uplink.sh / iperf3_uplink.sh
    │   ├── receiver_container_up.sh / _down.sh
    │   └── README.md
    ├── receiver_container/                   # the oai-perception-rx docker setup
    │   ├── Dockerfile                        # Ubuntu + GStreamer + python3-opencv (apt)
    │   ├── docker-compose.yaml               # joins oai-cn5g-public-net at 192.168.70.140
    │   ├── entrypoint.sh                     # adds 10.0.0.0/16 return route
    │   └── README.md
    ├── OAI/                                  # full OAI install (DO NOT EDIT the originals)
    │   ├── oai-cn5g/                         # 5G Core (docker-compose stack)
    │   └── openairinterface5g/               # gNB + UE (compiled rfsim binaries)
    ├── V2Xverse/                             # V2Xverse install (BLOCKED — see installation.md §12.3)
    └── phase3_logs/, phase3_logs_lowres/     # earlier SI/TI run outputs
```

---

## What's working as of right now

### OAI 5G stack
- Core network: 10 containers on `oai-cn5g-public-net` (subnet `192.168.70.128/26`). Includes `oai-amf` (.132), `oai-smf` (.133), `oai-upf` (.134), `oai-ext-dn` (.135).
- gNB + UE in rfsim mode (no real radio).
- UE attaches as `oaitun_ue1` with IP `10.0.0.2`.
- `iperf2`/`iperf3` UE→ext-DN verified.

### SCAN-AI video demo over 5G
- Sender: `abiodun/scan_sender_v3_oai.py` (host) — GStreamer pipeline `appsrc → videoconvert → x265enc → rtph265pay → udpsink bind-address=10.0.0.2 host=192.168.70.140 port=65000`. The `bind-address` is the load-bearing knob — without it the kernel shortcuts via the docker bridge and skips 5G entirely.
- Receiver: `abiodun/scan_receiver_v3_oai.py` (inside `oai-perception-rx` container) — GStreamer `udpsrc ... ! rtph265depay ! avdec_h265 ! videoconvert ! BGR ! appsink`, then `cv2.imshow` on the host display via X11 socket mount.
- Container: `oai-perception-rx` at `192.168.70.140`. Joins OAI's network as `external: true` so we never edit team OAI files. apt-installed `python3-opencv` (GTK backend) — avoid pip `opencv-python` (Qt plugin loader hell).
- **Confirmed end-to-end working from an AnyDesk session.** SSH sessions don't have DISPLAY so the cv2 window doesn't render there — not a bug.

### Run order
```bash
cd abiodun/scripts
./cn_start.sh                # one terminal
./gnb_start.sh               # second terminal, blocks
./ue_start.sh                # third terminal, blocks
./ue_check.sh                # sanity: tunnel + ping AMF/SMF/ext-DN
./iperf3_uplink.sh server    # in one window
./iperf3_uplink.sh client 1M 5    # in another — should hit ~1 Mbps over 5G
./receiver_container_up.sh   # builds first time (~2 min), then starts
python3 ../scan_sender_v3_oai.py   # in another terminal, CARLA must be running
```

---

## Current task in flight — split-inference OD over 5G

### Phase 1: refactor (where Codex picks up)

File: **`abiodun/carla_split_inference_udp_oai.py`** (copy of the team's `carla_split_inference_udp_demo.py`, 2247 → ~2400 lines after edits).

#### What's done
1. **Module docstring** rewritten to describe the OAI variant and the three roles (loopback / front / back).
2. **CLI flags added** in `parse_args()`:
   - `--role loopback|front|back` (default `loopback` for backwards compat)
   - `--bind-host` (local interface for socket bind; default `127.0.0.1`)
   - `--remote-host` (the other half's IP; default = bind-host)
3. **`UDPMessageSocket.__init__`** decoupled — now takes separate `host` (bind) and `remote_host` (send-to). Old single-`host` path still works as default.
4. **Lazy CARLA import.** `carla = None` at module load. `ensure_carla()` does the actual `_bootstrap_carla()` call. Called only at the top of front/loopback paths so back-role can run in a container without CARLA installed.
5. **New `run_back_only(args)` function** — builds back_model only, opens `remote_receiver` + `remote_sender` sockets with `bind_host` / `remote_host`, starts `RemoteInferenceWorker`, waits on `stop_event` for Ctrl+C, cleans up. ~60 lines, lives just before `run_demo`.
6. **`run_demo` dispatches by role** at the top:
   - `role == back` → calls `run_back_only(args)` and returns.
   - `role == front` → skips building `back_model` (kept as `None`), skips creating `remote_receiver` / `remote_sender` / `remote_worker` (all set to `None`), still opens `camera_sender` (front sends features) and `camera_receiver` (front receives detections). Starts only `result_receiver`.
   - `role == loopback` → original behavior, all 4 sockets + both halves in process.
7. **Cleanup paths guarded** with `is not None` checks so `front` mode doesn't NPE on the `finally:` block.
8. **`py_compile` passes.** `--help` correctly shows the three new flags.

#### What's left in Phase 1

| # | Task | Notes |
|---|------|-------|
| 1 | **Loopback baseline verification.** Run with `--role loopback --headless --disable-pretrained --front-device cpu --back-device cpu` against a running CARLA server, confirm at least 10 frames flow with no errors. This is the regression test — proves the refactor didn't break the original behavior. | The team's original demo is in `PythonAPI/neu_collab/carla_split_inference_udp_demo.py`; output should match. |
| 2 | **Back-role boot test.** `python3 -u carla_split_inference_udp_oai.py --role back --bind-host 127.0.0.1 --remote-host 127.0.0.1 --disable-pretrained --back-device cpu`. Should print `[back] device=cpu ...` and `[back] Press Ctrl+C to stop.` and idle. I attempted this with `timeout 8 ...` but the output was eaten; should rerun and inspect. | If `RemoteInferenceWorker` crashes on stale config, the fix is likely in the bind-host plumbing — re-check `run_back_only`. |
| 3 | **Front + back together on host (loopback IPs, two processes).** Open two terminals, start `--role back` first, then `--role front`. Should behave identically to `--role loopback` in one process. Proves the inter-process UDP path works before adding 5G. | Use default ports (36000-36003). |
| 4 | **Run instructions in `abiodun/scripts/README.md`.** Add a "Split-inference OD over 5G" section similar to the existing "First video demo over 5G" section. Cover Phase 1 (host-only) commands and the eventual Phase 3 (across 5G) commands. |  |

#### Known/anticipated issues

- **`RemoteInferenceWorker` may take 10–30s to load the pretrained Faster R-CNN weights on first run.** Smoke tests should allow at least 60s for the back role to print its "listening" line. Use `--disable-pretrained` to skip the download for testing.
- **`run_back_only` currently has `remote_receiver` constructed with `host=args.bind_host` only.** The receiver socket doesn't need a `remote_host` (it only receives), so that's correct. But if you want it to *send* back via a different IP, the constructor accepts `remote_host=` — see `UDPMessageSocket.__init__`.
- **CSV metrics warmup** uses `args.metrics_warmup_frames` (default a small number); for back-role with `--disable-data-collection` it's bypassed. Don't forget to disable that for back-role test runs.

---

## Phase 2 (after Phase 1 verifies) — GPU container for the back-half

The user chose **option C** for back-half placement: run it inside the perception container with GPU access. Reason: most architecturally faithful to the long-term cooperative-perception topology.

### Prerequisites (already approved by supervisor)
Install nvidia-container-toolkit on host:

```bash
# Stop everything first (docker daemon restart will kill running containers)
cd abiodun/scripts
./receiver_container_down.sh
./cn_stop.sh

# Install (standard NVIDIA-recommended command set)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify
docker info | grep -i runtime   # should show runc (default) AND nvidia (available)

# Bring things back up
./cn_start.sh
# gnb + ue in their own terminals
```

### Dockerfile changes needed for Phase 2

`abiodun/receiver_container/Dockerfile` currently uses `ubuntu:22.04`. To add GPU:

- **Base image:** switch to `nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04` (or whichever CUDA matches the RTX 5090 / Blackwell support — we previously verified PyTorch 2.9-nightly + cu129 works).
- **PyTorch install:** add `pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128` (or cu129). Pin numpy to be compatible (`numpy>=1.26,<2.3` matches what works with the apt python3-opencv).
- **Model weights:** Faster R-CNN MobileNet v3 320 FPN weights download at runtime (`build_detector_model` handles this via torchvision). Set `TORCH_HOME=/work/torch_cache` and mount a host volume to avoid re-downloading on container rebuild.
- **Compose change:** add the GPU device reservation to `oai-perception-rx` service:
  ```yaml
  deploy:
      resources:
          reservations:
              devices:
                  - driver: nvidia
                    count: all
                    capabilities: [gpu]
  ```
  (Or the older `runtime: nvidia` if compose v2 syntax misbehaves.)

### Container command for Phase 2
Override the container's `command:` to run the back-role:
```yaml
command: ["python3", "/work/abiodun/carla_split_inference_udp_oai.py",
          "--role", "back",
          "--bind-host", "0.0.0.0",
          "--remote-host", "10.0.0.2",
          "--back-device", "cuda"]
```

---

## Phase 3 — end-to-end run

```bash
# In one terminal — CARLA simulator (whatever the team's normal launch script is).

# In another terminal — front half on host, bound to UE IP, addressing the container.
cd PythonAPI/neu_collab/abiodun
python3 carla_split_inference_udp_oai.py \
    --role front \
    --bind-host 10.0.0.2 \
    --remote-host 192.168.70.140 \
    --camera-resolution 1080p \
    --enable-live-plot --enable-data-collection \
    --live-plot-update-interval 20 --metrics-batch-size 120 --metrics-flush-interval 1.0

# Container's docker-compose already starts the back-half (see Phase 2 command above).

# Compare the resulting metrics CSV (in --metrics-log-dir) against a loopback baseline run.
```

Metrics to look for (from the demo's existing instrumentation):
- per-frame `front_ms`, `payload_bytes`, `payload_chunks` (front-side)
- per-frame `server_ms`, `round_trip_ms` (back-side, reported via the result payload)
- detections count

A clear "5G adds X ms over loopback" number is the **deliverable for Phase 3**.

---

## Network topology cheat sheet

```
HOST (RTX 5090)                                       OAI Docker network 192.168.70.128/26
─────────────────────────────                         ──────────────────────────────────
CARLA server                                          oai-amf       .132
                                                      oai-smf       .133
PYTHON: split-inference --role front                  oai-upf       .134
  source bind=10.0.0.2 (oaitun_ue1)                   oai-ext-dn    .135  (iperf playground)
  dest=192.168.70.140                                 oai-perception-rx  .140  (our container)
      │                                                       │
      └──► oaitun_ue1 ──► nr-uesoftmodem ──► rfsim ──► nr-softmodem (gNB) ──► UPF ──► bridge ──► .140
                                                                                                  │
                                                                                                  ▼
                                                                       PYTHON: split-inference --role back
                                                                         RPN + ROI heads on GPU (Phase 2)
                                                                         detections sent back to 10.0.0.2

return: container.140 ──► UPF ──► gNB ──► UE ──► oaitun_ue1 ──► host kernel ──► front-half process
       (return route added by entrypoint.sh: 10.0.0.0/16 via 192.168.70.134)
```

---

## Watch-outs / gotchas Codex will hit

1. **DISPLAY in container.** `cv2.imshow` inside `oai-perception-rx` only works if the host has an X server *and* `DISPLAY` is set in the shell that ran `receiver_container_up.sh`. SSH sessions without `-X` produce empty `DISPLAY`. AnyDesk terminals work. If headless, use `--headless` flag in the split-inference script.
2. **NumPy 2 + pip opencv-python crash.** `_ARRAY_API not found`. Fix: use apt's `python3-opencv` (GTK GUI). Already applied in current Dockerfile.
3. **Qt platform plugin xcb fails inside container.** Same root cause — pip's opencv-python ships its own Qt. Use apt's python3-opencv.
4. **UE source-bind is load-bearing.** When the front-half sends to `192.168.70.140` *without* `bind-host=10.0.0.2`, the host kernel sees both the UE bridge and the docker bridge and shortcuts via docker. Bytes still arrive but 5G is bypassed. `bind-address` in GStreamer `udpsink`, or `--bind-host 10.0.0.2` for the split-inference script, makes the difference.
5. **Docker daemon restart kills running containers.** Always stop OAI + perception container before any `systemctl restart docker` (e.g. during nvidia-container-toolkit install).
6. **`--rfsimulator.[0].serveraddr` array index.** Without `.[0]` OAI's libconfig fails → nr-uesoftmodem segfaults. Affects ue_start.sh; gnb_start.sh already uses `--gNBs.[0]`.
7. **V2Xverse install is blocked** at the CARLA Python API step (no Python 3.10 wheel for CARLA 0.9.10.1; client 0.9.16/0.9.15 fail `rpc::rpc_error in get_sensor_token` against a 0.9.10.1 server). Notes in `installation.md` §12. Don't waste cycles retrying; supervisor is looking into another machine.
8. **Blackwell (sm_120) requires PyTorch nightly.** Verified working: `torch 2.9.0.dev20250813+cu129`. Stable PyTorch 2.5/2.6 will hang on RTX 5090.

---

## Open questions for the user / supervisor (deferred decisions)

- Should the receiver container's back-role download model weights at boot, or should we bake them into the image? (Boot-download is simpler, image-bake is faster startup; not urgent.)
- For Phase 3 5QI experiments, do we use SMF YAML (static per-DNN) or PCF (dynamic per-session)? See OAI tutorial: <https://openairinterface-docs-5b3d70.eurecom.io/projects/cn5g/Tutorials/QoS/>.
- Eventually: who consumes the multi-UE / multi-camera version? Decision deferred.

---

## Key earlier-session memories (Claude side, FYI)

These are decisions/facts I already saved to my persistent memory. Codex won't have them — listing the most load-bearing here:

- **Editing convention:** never edit team-shared scripts; copy to `/abiodun/`.
- **GStreamer over ffmpeg** for streaming. The ffmpeg PoC was deleted (`scan_sender_ffmpeg_poc.py`, `carla_multisensor_udp_ffmpeg_sender_v2.py`).
- **First demo over OAI** is the 2026-05-22 working state described above.
- **V2Xverse** is on hold pending another machine from supervisor.
- **Research direction** is the MobiSys-target paper-shaped doc at `abiodun/research_direction.md`. Strategy is **gap-finding** vs CoDriving/Coopernaut/Where2comm.

---

## What I would do next if I had more credit

1. Run the three Phase-1 verification tests (loopback baseline, back-role boot, two-process loopback).
2. Wire up the GPU Dockerfile + compose change for Phase 2.
3. Pre-write the Phase 3 run script (`abiodun/scripts/od_over_5g.sh`) so the actual demo is a one-liner.
4. After Phase 3 metrics are in, build the loopback-vs-5G comparison plot — that's the artifact for supervisor sync.

Good luck, Codex. The hard architectural work is done; what's left is mostly verification + container plumbing.
