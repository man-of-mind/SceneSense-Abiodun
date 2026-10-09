"""Paired stage breakdown of the INSTALLED UE input function (``prepare_live_input_bgr2rgb``) on retained frames.

Same caps as the live UE child (OMP/MKL 1); CUDA H2D measured with explicit synchronization. Each stage repeats
exactly the installed code's operation on the previous stage's output.
Run: env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m rl_agent.splitfusion_ue_input_prep_v2.breakdown out.json
"""
import json
import sys
import time

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from rl_agent.splitfusion_live_parity_repair_v1 import perception as P
from rl_agent.splitfusion_ue_input_prep_v1 import test_candidate as T


def frames():
    out = []
    for frame, bgra, live in T.recorded():
        out.append((frame, T.carla_image_to_bgr(bgra), np.ascontiguousarray(live[0, 3:7, :432, :]), live))
    return out


def stages(frame_bgr, radar_tensor, device, sync):
    c, t, marks = P.CONTRACT, time.perf_counter_ns, {}
    a = t()
    ok = (isinstance(frame_bgr, np.ndarray) and frame_bgr.dtype == np.uint8 and frame_bgr.shape == (720, 1280, 3)
          and isinstance(radar_tensor, np.ndarray) and radar_tensor.dtype == np.float32
          and radar_tensor.shape == (4, 432, 768) and np.isfinite(radar_tensor).all())
    b = t(); marks["validation_isfinite"] = b - a
    rgb8 = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB); a = t(); marks["bgr2rgb_cv2_retained"] = a - b
    image = Image.fromarray(rgb8).convert("RGB"); b = t(); marks["pil_fromarray_convert_copy"] = b - a
    image = image.resize((c.content_width, c.content_height), Image.Resampling.BILINEAR); a = t(); marks["pil_bilinear_resize"] = a - b
    f = np.asarray(image, dtype=np.float32); b = t(); marks["uint8_to_float32"] = b - a
    f = f.copy(); a = t(); marks["redundant_float32_copy"] = a - b
    rgb = torch.from_numpy(f).permute(2, 0, 1) / 255.0; b = t(); marks["permute_div255_temp"] = b - a
    mean = torch.tensor(c.mean, dtype=torch.float32).view(3, 1, 1); std = torch.tensor(c.std, dtype=torch.float32).view(3, 1, 1)
    rgb = (rgb - mean) / std; a = t(); marks["sub_mean_div_std_2temps"] = a - b
    radar = torch.from_numpy(np.ascontiguousarray(radar_tensor)); content = torch.cat([rgb, radar], dim=0); b = t(); marks["cat_7ch_copy"] = b - a
    fused = F.pad(content, (0, 0, 0, c.bottom_padding), value=0.0); a = t(); marks["pad_copy"] = a - b
    out = fused.unsqueeze(0).to(device=device, dtype=torch.float32); sync(); b = t(); marks["h2d_pageable_sync"] = b - a
    return out, {k: v / 1e6 for k, v in marks.items()}, ok


def main(out_path):
    torch.set_num_threads(1)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    data = frames()
    from rl_agent.splitfusion_ue_input_prep_v1.candidate import prepare_live_input_bgr2rgb
    for _, bgr, radar, _ in data[:5]:
        prepare_live_input_bgr2rgb(bgr, radar, device); sync()
    rows, totals = [], []
    for rep in range(3):
        for frame, bgr, radar, live in data:
            out, marks, ok = stages(bgr, radar, device, sync)
            assert ok and out.cpu().numpy().tobytes() == live.tobytes()
            rows.append(marks)
            a = time.perf_counter_ns(); prepare_live_input_bgr2rgb(bgr, radar, device); sync(); totals.append((time.perf_counter_ns() - a) / 1e6)
    keys = list(rows[0])
    summary = {k: dict(p50=round(float(np.median([r[k] for r in rows])), 3), p90=round(float(np.percentile([r[k] for r in rows], 90)), 3)) for k in keys}
    summary["sum_of_stage_p50_not_a_chain"] = round(sum(v["p50"] for v in summary.values()), 3)
    summary["installed_function_total"] = dict(p50=round(float(np.median(totals)), 3), p90=round(float(np.percentile(totals, 90)), 3))
    summary.update(device=str(device), frames=len(data), repeats=3, torch_threads=torch.get_num_threads())
    open(out_path, "w").write(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
