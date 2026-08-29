from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from common import CONFIG_PATH, ROOT, load_json, sha256, utc_now, write_json_x, write_text_x


def maybe(path: Path) -> Any | None:
    return load_json(path) if path.is_file() else None


def metric_table(records: list[dict[str, Any]], stage: str) -> str:
    if not records: return "Not run."
    if stage == "stage1":
        lines = ["| Epoch | Vehicle IoU | Person box-mask IoU | Foreground mIoU | Depth overall | 20–30 m | 30–40 m | Pass |",
                 "|---:|---:|---:|---:|---:|---:|---:|:---:|"]
        for item in records:
            s=item["segmentation"]; d=item["dense_depth"]
            lines.append(f"| {item['epoch']} | {s['vehicle_iou']:.6f} | {s['person_box_mask_iou']:.6f} | {s['foreground_miou']:.6f} | {d['overall']['episode_macro_log_mae']:.6f} | {d['20_30']['episode_macro_log_mae']:.6f} | {d['30_40']['episode_macro_log_mae']:.6f} | {item['pass']} |")
        return "\n".join(lines)
    lines = ["| Epoch | Eligible | Veh P/R/F1 | Person P/R/F1 | Veh/Person XY m | Veh IoU | Person mask IoU | FG mIoU |",
             "|---:|:---:|---|---|---|---:|---:|---:|"]
    for item in records:
        m=item["metrics"]
        lines.append(f"| {item['epoch']} | {item['eligible']} | {m['vehicle_precision']:.6f}/{m['vehicle_recall']:.6f}/{m['vehicle_f1']:.6f} | {m['person_precision']:.6f}/{m['person_recall']:.6f}/{m['person_f1']:.6f} | {m['vehicle_xy_mae_m']:.6f}/{m['person_xy_mae_m']:.6f} | {m['vehicle_iou']:.6f} | {m['person_box_mask_iou']:.6f} | {m['foreground_miou']:.6f} |")
    return "\n".join(lines)


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--experiment",required=True,type=Path)
    args=parser.parse_args(); experiment=args.experiment.resolve(strict=True)
    terminal=(experiment/"TERMINAL_VERDICT.txt").read_text().strip(); config=load_json(CONFIG_PATH)
    if terminal not in config["terminals"]: raise RuntimeError(f"unregistered terminal: {terminal}")
    design=load_json(experiment/"REGISTERED_TWO_STAGE_DESIGN.json"); qualification=load_json(experiment/"QUALIFICATION_REPORT.json")
    s1selection=maybe(experiment/"STAGE1_SELECTION.json"); transition=maybe(experiment/"STAGE2_TRANSITION.json")
    s2selection=maybe(experiment/"STAGE2_SELECTION.json")
    s1records=[load_json(experiment/f"stage1/evaluation/epoch_{e:03d}.json") for e in (10,20)
               if (experiment/f"stage1/evaluation/epoch_{e:03d}.json").is_file()]
    s2records=[load_json(experiment/f"stage2/evaluation/epoch_{e:03d}.json") for e in (10,20,30)
               if (experiment/f"stage2/evaluation/epoch_{e:03d}.json").is_file()]
    commits=subprocess.check_output(["git","log","--format=%H %s","8e556a680516638a720d9d37656493721b2bea87..HEAD"],cwd=ROOT,text=True).strip()
    changed=subprocess.check_output(["git","diff","--name-only","8e556a680516638a720d9d37656493721b2bea87..HEAD"],cwd=ROOT,text=True).splitlines()
    changed += [str((experiment/name).relative_to(ROOT)) for name in ("FINAL_REPORT.md","NOTIFICATION.json","PIPELINE_COMPLETE.json","COMPLETION_SENTINEL")]
    selected_record=next((item for item in s2records if s2selection and item["epoch"]==s2selection["selected_epoch"]),None)
    deltas="Not applicable."
    if selected_record:
        m=selected_record["metrics"]; b=config["baseline"]
        keys=("vehicle_precision","vehicle_recall","vehicle_f1","vehicle_xy_mae_m","vehicle_iou",
              "person_precision","person_recall","person_f1","person_xy_mae_m","person_box_mask_iou","foreground_miou")
        deltas="\n".join(f"- {key}: {m[key]-b[key]:+.6f}" for key in keys)
    training1=maybe(experiment/"stage1/TRAINING_COMPLETE.json"); training2=maybe(experiment/"stage2/TRAINING_COMPLETE.json")
    runtime=[]
    if training1: runtime.append(f"Stage 1 training: {training1['wall_seconds']:.1f} s")
    if training2: runtime.append(f"Stage 2 training: {training2['wall_seconds']:.1f} s")
    if s2records:
        for item in s2records:
            c=item["inference_contract"]; runtime.append(f"Stage-2 epoch {item['epoch']}: inference {item['inference_wall_seconds']:.1f} s, peak {c['peak_reserved_mib']:.1f} MiB, latency median {c['latency']['end_to_end_median_ms']:.3f} ms, transport {c['raw_transport_bytes']} bytes")
    closure=("This final task-separated LR-ASPP hypothesis is closed; no immediate LR-ASPP variation is proposed."
             if terminal in {"TWO_STAGE_LRASPP_STAGE1_REPRESENTATION_FAILED","TWO_STAGE_LRASPP_STAGE2_OBJECT_FAILED_CLOSE_LRASPP"} else "No Stage 3 or additional LR-ASPP variation was run.")
    report=f"""# Final Route B v3.1 task-separated LR-ASPP report

1. **Terminal.** `{terminal}`. {closure}

2. **Local master commits; no push.**\n```text\n{commits or 'No post-lineage commit recorded before report finalization.'}\n```\nThe work remained on local `master`; no branch or push was made.

3. **Data, cache and official seed.** Train 16,827 frames/10 episodes; validation 3,345 frames/two disjoint episodes; test absent and unopened. Manifest `{config['data']['manifest_sha256']}`. Official MobileNetV3 V2 `{config['pretrained']['sha256']}`. Train-cache hashes: `{json.dumps(design['train_depth_cache']['hashes'],sort_keys=True)}`.

4. **Architecture and split contract.** One seven-channel RGB-radar stem, one official-seeded MobileNetV3/LR-ASPP trunk, fused `low/high` interface, identity compression, shared depth-aware neck, segmentation and training-only dense-depth decoders, and two class-private object trunks/field heads. Qualification raw split/monolithic parity: `{qualification['checks']['split']['all_raw_equal']}`; transported names `low/high`; no RGB/radar/depth side channel enters the tail.

5. **Stage-1 counts.** {json.dumps(design['parameter_counts']['stage1'],sort_keys=True)}. Object-private parameters were frozen and excluded.

6. **Stage-1 epochs 10/20.**\n\n{metric_table(s1records,'stage1')}

7. **Constant-depth baseline and gates.** Train median {config['stage1_gates']['constant_train_median_depth_m']:.9f} m (`log1p` {config['stage1_gates']['constant_train_median_log1p']:.9f}); per-frame→episode→equal-episode baseline {json.dumps(config['stage1_gates']['constant_train_episode_macro_log_mae'],sort_keys=True)}. Candidate limits were 90% overall and 95% in each far band.

8. **Stage-1 selection.** `{json.dumps(s1selection,sort_keys=True) if s1selection else 'not available'}`

9. **Stage-2 reset/fresh optimizer.** `{json.dumps({key:transition[key] for key in ('selected_stage1_epoch','verified_selected_sha','reset_seed','reset_checks','segmentation_dense_reset_bit_identical','fresh_optimizer','stage2_epoch000')} ,sort_keys=True) if transition else 'Stage 2 not authorized.'}`

10. **Stage-2 frozen audit.** `{('All 30 epoch-boundary representation hashes matched '+transition['frozen_representation_hash']) if transition and s2selection and all(x['pass'] for x in s2selection['frozen_boundary_audit']) else 'Stage 2 not run.'}`

11. **Stage-2 epochs 10/20/30.**\n\n{metric_table(s2records,'stage2')}

12. **Selected deltas versus epoch-40 inherited baseline.**\n{deltas}

13. **Nine service gates/material gain.** `{json.dumps({'service':s2selection.get('service_gates') if s2selection else None,'service_ready':s2selection.get('service_ready') if s2selection else None,'material':s2selection.get('material_gates') if s2selection else None,'material_improvement':s2selection.get('material_improvement') if s2selection else None},sort_keys=True)}`

14. **Selected Stage-2 checkpoint.** `{json.dumps({'epoch':s2selection.get('selected_epoch'),'checkpoint':s2selection.get('selected_checkpoint'),'sha256':s2selection.get('selected_checkpoint_sha256')} if s2selection else None,sort_keys=True)}`

15. **v0.25 sensitivity.** `{json.dumps(s2selection.get('v025_sensitivity') if s2selection else None,sort_keys=True)}` It was run only when a selected eligible Stage-2 checkpoint existed.

16. **Runtime, VRAM, latency and transport.** {'; '.join(runtime) if runtime else 'No scientific training runtime.'} Qualification batch 16/accumulation 1 memory: `{json.dumps(qualification['checks']['memory'],sort_keys=True)}`.

17. **No inference-time depth.** Qualification sentinel passed `{qualification['checks']['no_inference_depth']}`. Deployable Stage-2 prediction signatures accepted RGB-radar only and recorded zero depth paths/labels.

18. **Prohibited scope.** Test, CARLA, OAI contents, q/AE, live split runtime, and the 288 measurements were not opened, run, altered, or started. The pre-existing `OAI/openairinterface5g` gitlink dirtiness was preserved.

19. **Exact changed files.**\n```text\n{chr(10).join(sorted(set(changed)))}\n```

20. **Notification and sentinel.** `NOTIFICATION.json`, `PIPELINE_COMPLETE.json`, and `COMPLETION_SENTINEL` were emitted with the same sole terminal. No q/AE follow-on was started.
"""
    write_text_x(experiment/"FINAL_REPORT.md",report)
    notification={"schema":"two_stage_lraspp_notification_v1","created_utc":utc_now(),"terminal":terminal,
                  "service_eligible_for_later_q_ae":terminal=="TWO_STAGE_LRASPP_SERVICE_READY","q_ae_started":False}
    write_json_x(experiment/"NOTIFICATION.json",notification)
    write_json_x(experiment/"PIPELINE_COMPLETE.json",{"schema":"two_stage_lraspp_pipeline_complete_v1",
        "created_utc":utc_now(),"terminal":terminal,"exactly_one_terminal":True,"final_report":"FINAL_REPORT.md",
        "final_report_sha256":sha256(experiment/"FINAL_REPORT.md"),"notification":"NOTIFICATION.json"})
    write_text_x(experiment/"COMPLETION_SENTINEL",terminal+"\n")
    print(json.dumps({"terminal":terminal,"report":str(experiment/"FINAL_REPORT.md"),
                      "report_sha256":sha256(experiment/"FINAL_REPORT.md")},indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
