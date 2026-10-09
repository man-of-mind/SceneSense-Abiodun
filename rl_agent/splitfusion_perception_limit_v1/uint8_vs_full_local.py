"""Exact UINT8-q0 vs full-local differences in the sealed paired replay (addendum evidence; read-only)."""
import hashlib, json, sys
from pathlib import Path
P = Path("experiments/splitfusion_run5_paired_baselines_v1/20261006_125848_v11_treatment/replay/REPLAY.json")
r = json.loads(P.read_text())
out = dict(replay_sha256=hashlib.sha256(P.read_bytes()).hexdigest(), frames=len(r["rows"]))
cnt_keys = ("eligible_gt", "tp", "fp", "fn", "ignored_predictions")
count_diff = err_frames = err_objects = label_frames = q_frames = within_diff = 0
max_err = max_q = max_qseg = max_qloc = 0.0
for row in r["rows"]:
    a, b = row["full_local"], row["fixed_uint8"]
    for cls in ("vehicle", "person"):
        ca, cb = a["classes"][cls], b["classes"][cls]
        count_diff += any(ca[k] != cb[k] for k in cnt_keys)
        ea, eb = ca["matched_xy_errors_m"], cb["matched_xy_errors_m"]
        tol = 1.0 if cls == "vehicle" else 1.2
        within_diff += sum(e <= tol for e in ea) != sum(e <= tol for e in eb)
        if ea != eb:
            err_frames += 1
            for x, y in zip(ea, eb):
                if x != y:
                    err_objects += 1
                    max_err = max(max_err, abs(x - y))
    label_frames += row["baseline"]["labels_sha256"] != row["parity"]["dense_labels_array_sha256"]
    qa, qb = a["quality"], b["quality"]
    if qa["q_perc"] != qb["q_perc"]:
        q_frames += 1
        max_q = max(max_q, abs(qa["q_perc"] - qb["q_perc"]))
    if qa["q_seg"] is not None:
        max_qseg = max(max_qseg, abs(qa["q_seg"] - qb["q_seg"]))
    if qa["q_loc"] is not None and qb["q_loc"] is not None:
        max_qloc = max(max_qloc, abs(qa["q_loc"] - qb["q_loc"]))
out.update(class_frames_with_count_difference=count_diff, class_frames_with_within_tol_count_difference=within_diff,
           class_frames_with_matched_error_difference=err_frames, matched_objects_with_error_difference=err_objects,
           max_abs_matched_error_difference_m=max_err, frames_with_label_difference=label_frames,
           frames_with_q_perc_difference=q_frames, max_abs_q_perc_difference=max_q,
           max_abs_q_seg_difference=max_qseg, max_abs_q_loc_difference=max_qloc,
           script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
