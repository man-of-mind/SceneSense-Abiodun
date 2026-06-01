# Perception-side receiver container.
# Sits on the OAI public_net next to oai-ext-dn (192.168.70.0/26).
# Phase 1/SCAN-AI video mode receives H.265/RTP over the 5G data plane.
# Phase 2 OD mode runs the split-inference back-half on GPU.

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/work/torch_cache \
    MPLBACKEND=Agg \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# GStreamer (H.265 decode via avdec_h265) + system OpenCV + PyTorch.
# We use apt's python3-opencv (built with GTK, no Qt plugin loader gotchas)
# instead of pip's opencv-python (which ships its own Qt and needs a long
# list of libxcb-* libraries to satisfy its xcb platform plugin).
# iproute2/ping/iperf3/tcpdump are for in-container debugging
# (route checks, packet inspection on eth0).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-gi python3-gst-1.0 python3-opencv python3-numpy \
        gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav \
        iproute2 iputils-ping iperf3 tcpdump \
        libgl1 libglib2.0-0 libgtk-3-0 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Blackwell / RTX 5090 needs a PyTorch build with sm_120 support. The cu128
# nightly wheel path is the known-good track for this machine as of the Phase 2
# handoff; keep numpy below 2 because apt's python3-opencv is built against
# the NumPy 1.x C API and fails with NumPy 2.x (`_ARRAY_API not found`).
RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install "numpy>=1.26,<2" \
    && python3 -m pip install filelock typing-extensions sympy networkx jinja2 fsspec \
    && python3 -m pip install --pre torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/nightly/cu128

WORKDIR /work
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
