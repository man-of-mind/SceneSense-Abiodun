# Trained LR-ASPP Split-Inference Evaluation Instructions

Use this after `experiments/pole_lraspp_training/<timestamp>_pole_lraspp_training/manifest.json`
shows `"status": "complete"`.

## Prepared Script

The prepared split-inference client is:

```bash
/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/carla_split_inference_udp_segmentation_trained_lraspp_demo.py
```

It is an additive copy of `carla_split_inference_udp_segmentation_demo.py` and
defaults to the 3-class checkpoint produced by `pole_lraspp_training`:

- `0`: background
- `1`: vehicle
- `2`: person

It can read the best checkpoint directly from a completed training experiment
using `--trained-experiment-dir`.

## Request To Give Codex

```text
Training has completed. Use:
<EXPERIMENT_DIR>

Please run the trained LR-ASPP split-inference evaluation using
carla_split_inference_udp_segmentation_trained_lraspp_demo.py.

Use the best checkpoint from that experiment directory, enable CARLA semantic
GT, collect latency/payload/mIoU metrics, keep weather unchanged, run headless,
and write outputs under:
<EXPERIMENT_DIR>/split_inference_eval/

Run a bounded evaluation of 1200 frames for these scenarios:
- baseline transport
- per-level compression probe enabled
- optional ROI drop fractions 0.0, 0.25, 0.5 if time permits

After the run, summarize:
- front-half latency
- back-half latency
- round-trip latency
- UDP payload bytes and compression ratio
- mIoU macro
- vehicle IoU
- person IoU
- pixel/object foreground behavior from the metrics CSV

Generate plots and a short report under the split_inference_eval directory.
```

## Example Command

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab

EXP="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/experiments/pole_lraspp_training/<timestamp>_pole_lraspp_training"
OUT="$EXP/split_inference_eval"
mkdir -p "$OUT"

/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  carla_split_inference_udp_segmentation_trained_lraspp_demo.py \
  --trained-experiment-dir "$EXP" \
  --town "" \
  --weather-preset unchanged \
  --headless \
  --disable-live-plot \
  --enable-data-collection \
  --enable-semantic-gt \
  --metrics-log-dir "$OUT" \
  --metrics-log-prefix trained_split_eval \
  --run-tag trained_lraspp_gt_baseline \
  --camera-resolution 720p \
  --fps 10 \
  --npc-vehicles 40 \
  --npc-pedestrians 20 \
  --max-frames 1200 \
  --metrics-warmup-frames 60 \
  --per-level-compress-probe
```

Expected successful artifacts:

- `$OUT/trained_split_eval_<run_tag>_<timestamp>.csv`
- `$OUT/trained_split_eval_<run_tag>_<timestamp>.manifest.json`
- `$OUT/analysis/summary.json`
- `$OUT/analysis/report.txt`
- `$OUT/analysis/*.png`
- `$OUT/analysis/*.pdf`

The CSV includes:

- `front_ms`
- `back_ms`
- `round_trip_ms`
- `payload_bytes`
- `payload_bytes_uncompressed`
- `miou_3class_macro`
- `miou_vehicle_iou`
- `miou_person_iou`
- per-level compressed/uncompressed byte JSON columns

## Analyze The Evaluation CSV

```bash
/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/pole_lraspp_training/scripts/analyze_trained_split_eval.py \
  --glob "$OUT/trained_split_eval_*.csv" \
  --output-dir "$OUT/analysis"
```
