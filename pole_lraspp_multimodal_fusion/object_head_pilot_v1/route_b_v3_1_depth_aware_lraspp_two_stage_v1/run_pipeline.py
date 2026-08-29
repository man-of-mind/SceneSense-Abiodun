from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path

from common import utc_now, write_json_x, write_text_x

PACKAGE = Path(__file__).resolve().parent


def run(script: str, experiment: Path, *extra: str) -> None:
    command = [sys.executable, str(PACKAGE / script), "--experiment", str(experiment), *extra]
    print(json.dumps({"boundary_launch": script, "command": command}), flush=True)
    subprocess.run(command, check=True)


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--experiment",required=True,type=Path)
    args=parser.parse_args(); experiment=args.experiment.resolve(strict=True); active="qualification"
    try:
        if not (experiment/"QUALIFICATION_COMPLETE").is_file(): run("qualify.py",experiment)
        active="stage1_training"
        if not (experiment/"stage1/TRAINING_COMPLETE").is_file(): run("train_stage.py",experiment,"--stage","stage1")
        active="stage1_evaluation"
        if not (experiment/"STAGE1_SELECTION.json").is_file(): run("evaluate_stage1.py",experiment)
        selection=json.loads((experiment/"STAGE1_SELECTION.json").read_text())
        if selection["stage2_authorized"]:
            active="stage2_transition"
            if not (experiment/"STAGE2_AUTHORIZED").is_file(): run("transition_stage2.py",experiment)
            active="stage2_training"
            if not (experiment/"stage2/TRAINING_COMPLETE").is_file(): run("train_stage.py",experiment,"--stage","stage2")
            active="stage2_inference"
            for epoch in (10,20,30):
                if not (experiment/f"stage2/predictions/epoch_{epoch:03d}/INFERENCE_COMPLETE").is_file():
                    run("infer_stage2.py",experiment,"--epoch",str(epoch))
            active="stage2_evaluation"
            if not (experiment/"STAGE2_SELECTION.json").is_file(): run("evaluate_stage2.py",experiment)
        active="finalization"
        if not (experiment/"FINAL_REPORT.md").is_file(): run("finalize.py",experiment)
        return 0
    except Exception as error:
        terminal=("TWO_STAGE_LRASPP_CONTRACT_INVALID" if active=="qualification"
                  else "TWO_STAGE_LRASPP_RUNTIME_FAILURE")
        if not (experiment/"TERMINAL_VERDICT.txt").exists(): write_text_x(experiment/"TERMINAL_VERDICT.txt",terminal+"\n")
        if not (experiment/"SCIENTIFIC_FAILURE.json").exists():
            write_json_x(experiment/"SCIENTIFIC_FAILURE.json",{"schema":"two_stage_lraspp_failure_v1",
                "created_utc":utc_now(),"active_phase":active,"terminal":terminal,
                "exception_type":type(error).__name__,"exception":str(error),"traceback":traceback.format_exc()})
        print(json.dumps({"terminal":terminal,"active_phase":active,"error":str(error)}),flush=True)
        return 2


if __name__=="__main__": raise SystemExit(main())
