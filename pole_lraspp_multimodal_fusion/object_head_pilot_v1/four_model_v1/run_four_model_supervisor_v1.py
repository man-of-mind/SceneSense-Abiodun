#!/usr/bin/env python3
"""W10275 supervisor: four validation-selected Route B candidates.

Phases, run strictly sequentially with exactly one GPU training process alive at
any time:

  PHASE_0_FREEZE_NOAE    verify the frozen noAE Stage-2 epoch-13 SHA-256; never retrain it
  PHASE_1_DECODER        record the already-selected duplicate-reducer decoder
  PHASE_2_AE64 .. _AE128 per family: train exactly 20 epochs, decode epochs
                         {0,4,9,14,19} under the frozen decoder, select
  PHASE_3_SUMMARY        write the summary, the DONE marker and the terminal

Design rules this wrapper enforces:

* A metric gate NEVER prevents training. Gates set a family's *status*; the
  next family is trained regardless.
* A crashed family is recorded and skipped without diagnosis or retry; the run
  continues if the GPU is still healthy.
* Nothing existing is overwritten: training refuses a populated checkpoint dir
  and each evaluation refuses an existing eval/<tag>/.
* The hard wall budget is checked before each family and before each decode; work
  that cannot finish is skipped rather than started and truncated.

State is written to ``state.json`` at every phase transition (and during
training, with the current epoch parsed from the trial log) so an observer never
needs to tail a log.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
PILOT_ROOT = HERE.parent
PKG_ROOT = PILOT_ROOT.parent
ABIODUN = PKG_ROOT.parent
for _p in (str(HERE), str(PKG_ROOT), str(ABIODUN)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PYTHON = "/usr/bin/python3"
NOAE_RUN = ABIODUN / "experiments/route_b_noae_precision_full_v1/20260825_195301"
NOAE_CKPT = NOAE_RUN / "checkpoints/curriculum_stage2_joint_v1/epoch_013.pt"
NOAE_SHA = "0882ef922edbcb8da47fe6568d8ba125e00bab71365d0370fd77268eb747dc30"
NOAE_EVAL = NOAE_RUN / "eval/curriculum_stage2_joint_v1_epoch_013"
BASE_CONFIG = PILOT_ROOT / "configs/route_b_noae_precision_pilot_v1.yaml"

FAMILIES = [
    {"name": "AE64", "bottleneck": 64, "trial": "route_b_ae64_adapt_v1"},
    {"name": "AE32", "bottleneck": 32, "trial": "route_b_ae32_adapt_v1"},
    {"name": "AE128", "bottleneck": 128, "trial": "route_b_ae128_adapt_v1"},
]
DECODE_EPOCHS = [0, 4, 9, 14, 19]
EPOCHS_PER_FAMILY = 20

# Historical service targets. Advisory only: reported MET/UNMET, never a gate.
SERVICE_TARGETS = {"vehicle_recall": 0.85, "person_recall": 0.80}

# Conservative wall-clock reserves, from the measured noAE run
# (~1.9 min/epoch training, ~2.6 min/decode).
TRAIN_RESERVE_S = 75 * 60
DECODE_RESERVE_S = 4 * 60
SUMMARY_RESERVE_S = 5 * 60


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Supervisor:
    def __init__(self, exp_dir: Path, budget_hours: float, decoder_radius_m: float) -> None:
        self.exp = exp_dir
        self.budget_s = budget_hours * 3600.0
        self.radius = decoder_radius_m
        self.t0 = time.monotonic()
        self.started = utc()
        self.state_path = exp_dir / "state.json"
        self.log_path = exp_dir / "wrapper.log"
        self.state: Dict[str, Any] = {
            "wrapper_pid": os.getpid(),
            "started_utc": self.started,
            "hard_budget_hours": budget_hours,
            "phase": "PHASE_0_FREEZE_NOAE",
            "current_family": None,
            "current_epoch": None,
            "status": "RUNNING",
            "terminal": None,
            "families": {},
            "phase_history": [],
            "updated_utc": self.started,
        }
        self.report: Dict[str, Any] = {}
        self.write_state()

    # ---------------- plumbing ----------------
    def log(self, message: str) -> None:
        line = f"[{utc()}] {message}"
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        print(line, flush=True)

    def elapsed(self) -> float:
        return time.monotonic() - self.t0

    def remaining(self) -> float:
        return self.budget_s - self.elapsed()

    def write_state(self, **updates: Any) -> None:
        self.state.update(updates)
        self.state["updated_utc"] = utc()
        self.state["elapsed_s"] = round(self.elapsed(), 1)
        self.state["remaining_s"] = round(self.remaining(), 1)
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.state_path)

    def phase(self, name: str, **updates: Any) -> None:
        self.state["phase_history"].append({"phase": name, "at": utc(),
                                            "elapsed_s": round(self.elapsed(), 1)})
        self.log(f"PHASE {name}")
        self.write_state(phase=name, **updates)

    # ---------------- GPU accounting ----------------
    @staticmethod
    def gpu_process_mib(pid: int) -> Optional[int]:
        """GPU memory of one process, so the desktop's usage is never counted as ours."""
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=20,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return None
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and parts[0].isdigit() and int(parts[0]) == pid:
                try:
                    return int(parts[1])
                except ValueError:
                    return None
        return None

    def gpu_healthy(self) -> bool:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=30,
            )
            return result.returncode == 0 and bool(result.stdout.strip())
        except (OSError, subprocess.SubprocessError):
            return False

    # ---------------- phase 0 ----------------
    def freeze_noae(self) -> Dict[str, Any]:
        self.phase("PHASE_0_FREEZE_NOAE")
        import hashlib
        digest = hashlib.sha256(NOAE_CKPT.read_bytes()).hexdigest()
        ok = digest == NOAE_SHA
        self.log(f"noAE Stage-2 epoch 13 sha256={digest} expected={NOAE_SHA} match={ok}")
        if not ok:
            raise RuntimeError("frozen noAE checkpoint SHA-256 mismatch; refusing to proceed")
        return {"family": "noAE", "epoch": 13, "checkpoint": str(NOAE_CKPT),
                "sha256": digest, "retrained": False, "status": "FROZEN"}

    # ---------------- phase 1 ----------------
    def record_decoder(self, decision: Dict[str, Any]) -> None:
        self.phase("PHASE_1_DECODER", decoder_radius_m=self.radius,
                   decoder_status=decision.get("terminal"))
        self.log(f"frozen decoder: vehicle-only predicted-world NMS radius={self.radius} m "
                 f"({decision.get('terminal')})")

    # ---------------- training ----------------
    def train_family(self, family: Dict[str, Any]) -> Dict[str, Any]:
        trial = family["trial"]
        trial_json = HERE / "configs" / f"{trial}.json"
        ckpt_dir = self.exp / "checkpoints" / trial
        record: Dict[str, Any] = {"trial": trial, "bottleneck": family["bottleneck"],
                                  "train_started_utc": utc()}
        if (ckpt_dir / "best.pt").is_file():
            record.update(status="SKIPPED_EXISTING_CHECKPOINT", train_seconds=0)
            self.log(f"{family['name']}: refusing to overwrite {ckpt_dir/'best.pt'}")
            return record

        # Budget bound handed to the trainer as well, so it self-limits if we are late.
        budget_hours = max(0.1, min(2.0, (self.remaining() - DECODE_RESERVE_S * len(DECODE_EPOCHS)
                                          - SUMMARY_RESERVE_S) / 3600.0))
        cmd = [PYTHON, str(PILOT_ROOT / "run_object_head_pilot_v1.py"),
               "--config", str(BASE_CONFIG),
               "--trial-json", str(trial_json),
               "--experiment-dir", str(self.exp),
               "--training-budget-hours", f"{budget_hours:.4f}"]
        train_log = self.exp / f"train_{trial}.log"
        self.log(f"{family['name']}: launching training (budget {budget_hours:.2f} h) -> {train_log}")

        env = dict(os.environ)
        # A CARLA-shadowing PYTHONPATH must never reach a training child.
        env.pop("PYTHONPATH", None)
        t_start = time.monotonic()
        peak_mib = 0
        with train_log.open("w", encoding="utf-8") as fh:
            proc = subprocess.Popen(cmd, cwd=str(PKG_ROOT), stdout=fh, stderr=subprocess.STDOUT, env=env)
            self.write_state(current_family=family["name"], current_epoch=None,
                             status="TRAINING", training_pid=proc.pid)
            trial_log = self.exp / "supervisor.log"
            epoch_re = re.compile(re.escape(trial) + r" epoch=(\d+) ")
            while proc.poll() is None:
                time.sleep(30)
                mib = self.gpu_process_mib(proc.pid)
                if mib:
                    peak_mib = max(peak_mib, mib)
                epoch = None
                if trial_log.is_file():
                    try:
                        matches = epoch_re.findall(trial_log.read_text(encoding="utf-8", errors="ignore"))
                        if matches:
                            epoch = int(matches[-1])
                    except OSError:
                        pass
                self.write_state(current_family=family["name"], current_epoch=epoch,
                                 status="TRAINING", gpu_peak_mib=peak_mib)
            returncode = proc.returncode
        record["train_seconds"] = round(time.monotonic() - t_start, 1)
        record["gpu_peak_mib"] = peak_mib
        record["returncode"] = returncode
        record["train_log"] = str(train_log)
        record["train_finished_utc"] = utc()

        written = sorted(ckpt_dir.glob("epoch_*.pt")) if ckpt_dir.is_dir() else []
        record["epoch_checkpoints"] = len(written)
        if returncode != 0:
            record["status"] = "TRAINING_FAILED"
            tail = ""
            if train_log.is_file():
                tail = "".join(train_log.read_text(encoding="utf-8", errors="ignore").splitlines(True)[-15:])
            record["failure_tail"] = tail
            self.log(f"{family['name']}: TRAINING FAILED rc={returncode} "
                     f"({record['epoch_checkpoints']} epoch checkpoints); recorded, no retry")
        else:
            record["status"] = "TRAINED"
            self.log(f"{family['name']}: trained in {record['train_seconds']:.0f} s, "
                     f"{record['epoch_checkpoints']} epoch checkpoints, peak {peak_mib} MiB")
        return record

    # ---------------- decode + select ----------------
    def decode_and_select(self, family: Dict[str, Any], record: Dict[str, Any],
                          baseline: Dict[str, Any], split_ids, collision_ids) -> Dict[str, Any]:
        import route_b_select_ae_v1 as sel

        trial = family["trial"]
        ckpt_dir = self.exp / "checkpoints" / trial
        records: List[Dict[str, Any]] = []
        eval_seconds = 0.0
        skipped: List[int] = []
        for epoch in DECODE_EPOCHS:
            ckpt = ckpt_dir / f"epoch_{epoch:03d}.pt"
            if not ckpt.is_file():
                skipped.append(epoch)
                self.log(f"{family['name']}: epoch {epoch} checkpoint absent; skipped")
                continue
            if self.remaining() < DECODE_RESERVE_S + SUMMARY_RESERVE_S:
                skipped.append(epoch)
                self.log(f"{family['name']}: epoch {epoch} decode skipped, "
                         f"{self.remaining():.0f} s left in the hard budget")
                continue
            tag = f"{trial}_epoch_{epoch:03d}"
            eval_dir = self.exp / "eval" / tag
            self.write_state(current_family=family["name"], current_epoch=epoch, status="DECODING")
            if not eval_dir.is_dir():
                t_eval = time.monotonic()
                env = dict(os.environ)
                env.pop("PYTHONPATH", None)
                cmd = [PYTHON, str(PILOT_ROOT / "evaluate_route_b_checkpoint_v1.py"),
                       "--experiment-dir", str(self.exp), "--checkpoint", str(ckpt),
                       "--tag", tag, "--config", str(BASE_CONFIG), "--split", "val",
                       "--python", PYTHON]
                log_file = self.exp / f"eval_{tag}.log"
                with log_file.open("w", encoding="utf-8") as fh:
                    rc = subprocess.run(cmd, cwd=str(PKG_ROOT), stdout=fh,
                                        stderr=subprocess.STDOUT, env=env).returncode
                eval_seconds += time.monotonic() - t_eval
                if rc != 0:
                    skipped.append(epoch)
                    self.log(f"{family['name']}: epoch {epoch} decode failed rc={rc}; recorded, no retry")
                    continue
            rec = sel.candidate_record(eval_dir, self.exp, self.radius, "val", epoch,
                                       split_ids, collision_ids)
            records.append(rec)
            p = rec["postprocessed"]["primary"]
            self.log(f"{family['name']}: epoch {epoch} meanF1={p['mean_f1']:.5f} "
                     f"meanXY={p['mean_xy_mae_m']:.4f} dupFP/frame={p['duplicate_fp_per_frame']:.4f}")

        outcome = sel.select_family(records, baseline)
        outcome["epochs_skipped"] = skipped
        record["eval_seconds"] = round(eval_seconds, 1)
        record["selection"] = outcome
        if record.get("status") == "TRAINED":
            record["status"] = outcome["status"]
        return record

    # ---------------- summary ----------------
    def service_targets(self, primary: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for key, target in SERVICE_TARGETS.items():
            value = float(primary[key])
            out[key] = {"target": target, "value": value,
                        "status": "MET" if value >= target else "UNMET"}
        return out

    def finish(self, terminal: str) -> None:
        self.phase("PHASE_3_SUMMARY", status="SUMMARIZING")
        self.report["terminal"] = terminal
        self.report["total_wall_seconds"] = round(self.elapsed(), 1)
        self.report["finished_utc"] = utc()
        (self.exp / "decision" / "four_model_report_v1.json").write_text(
            json.dumps(self.report, indent=2, sort_keys=True), encoding="utf-8")
        (self.exp / "DONE").write_text(
            f"terminal={terminal}\nfinished={utc()}\n"
            f"wall_seconds={self.report['total_wall_seconds']}\n", encoding="utf-8")
        self.write_state(phase="DONE", status="DONE", terminal=terminal,
                         current_family=None, current_epoch=None)
        self.log(f"TERMINAL {terminal} after {self.elapsed()/3600.0:.2f} h")
        try:
            subprocess.run(["notify-send", "Route B four-model run complete", terminal],
                           timeout=20, capture_output=True)
        except (OSError, subprocess.SubprocessError) as exc:
            self.log(f"notify-send unavailable ({exc}); not an experiment failure")

    # ---------------- main ----------------
    def run(self, decoder_decision: Dict[str, Any]) -> str:
        import route_b_select_ae_v1 as sel

        self.report["started_utc"] = self.started
        self.report["experiment_dir"] = str(self.exp)
        self.report["wrapper_pid"] = os.getpid()
        self.report["hard_budget_hours"] = self.budget_s / 3600.0
        self.report["decoder"] = decoder_decision
        self.report["service_targets_note"] = (
            "historical 0.85 vehicle / 0.80 person service recall targets are advisory; "
            "reported MET/UNMET and never used as a gate"
        )

        self.report["noae"] = self.freeze_noae()
        self.record_decoder(decoder_decision)

        split_ids, collision_ids = pp_split(self.exp)
        baseline = sel.candidate_record(NOAE_EVAL, self.exp, self.radius, "val", 13,
                                        split_ids, collision_ids)
        self.report["noae"]["metrics"] = {
            "raw": baseline["raw"], "postprocessed": baseline["postprocessed"],
            "vehicle_iou": baseline["vehicle_iou"], "person_iou": baseline["person_iou"],
            "miou": baseline["miou"],
        }
        self.report["noae"]["service_targets"] = {
            "raw": self.service_targets(baseline["raw"]["primary"]),
            "postprocessed": self.service_targets(baseline["postprocessed"]["primary"]),
        }
        (self.exp / "decision" / "baseline_noae_epoch013.json").write_text(
            json.dumps(baseline, indent=2, sort_keys=True), encoding="utf-8")

        families: Dict[str, Any] = {}
        budget_hit = False
        crashed: List[str] = []
        for family in FAMILIES:
            name = family["name"]
            self.phase(f"PHASE_2_{name}", current_family=name)
            if self.remaining() < TRAIN_RESERVE_S + SUMMARY_RESERVE_S:
                self.log(f"{name}: not started, {self.remaining():.0f} s left in the hard budget")
                families[name] = {"trial": family["trial"], "bottleneck": family["bottleneck"],
                                  "status": "NOT_STARTED_TIME_BUDGET"}
                budget_hit = True
                self.state["families"] = families
                self.write_state()
                continue
            if not self.gpu_healthy():
                self.log(f"{name}: GPU not healthy; skipping remaining families")
                families[name] = {"trial": family["trial"], "bottleneck": family["bottleneck"],
                                  "status": "SKIPPED_GPU_UNHEALTHY"}
                crashed.append(name)
                self.state["families"] = families
                self.write_state()
                break

            record = self.train_family(family)
            if record["status"] == "TRAINING_FAILED":
                crashed.append(name)
            # Decode whatever epochs exist, even after a crash: a metric gate or a
            # partial family must never stop the remaining families.
            record = self.decode_and_select(family, record, baseline, split_ids, collision_ids)
            selected = record.get("selection", {}).get("selected")
            if selected:
                record["service_targets"] = {
                    "raw": self.service_targets(selected["raw"]["primary"]),
                    "postprocessed": self.service_targets(selected["postprocessed"]["primary"]),
                }
                ckpt = Path(selected["checkpoint"])
                if ckpt.is_file():
                    import hashlib
                    record["selected_sha256"] = hashlib.sha256(ckpt.read_bytes()).hexdigest()
                    record["selected_checkpoint"] = str(ckpt)
                    record["selected_epoch"] = selected["epoch"]
            families[name] = record
            self.state["families"] = families
            self.write_state()
            (self.exp / "decision" / f"selection_{family['trial']}.json").write_text(
                json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")

        self.report["families"] = families

        available = [n for n, r in families.items()
                     if r.get("selection", {}).get("selected") is not None]
        gate_failed = [n for n, r in families.items() if r.get("status") == "VALIDATION_GATE_FAILED"]
        self.report["candidates_available"] = ["noAE"] + available
        self.report["families_with_validation_failures"] = gate_failed
        self.report["families_crashed"] = crashed

        if budget_hit and len(available) < len(FAMILIES):
            terminal = "HARD_TIME_BUDGET_REACHED"
        elif len(available) < len(FAMILIES):
            terminal = "PARTIAL_FAMILY_RUNTIME_FAILURE"
        elif gate_failed:
            terminal = "FOUR_CANDIDATES_WITH_VALIDATION_FAILURES"
        else:
            terminal = "FOUR_VALIDATION_CANDIDATES_READY"
        self.finish(terminal)
        return terminal


def pp_split(exp: Path):
    import route_b_postprocess_v1 as pp
    return pp.split_and_collision_ids(exp, "val")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--decoder-decision", required=True, type=Path)
    parser.add_argument("--budget-hours", type=float, default=7.0)
    args = parser.parse_args(argv)

    exp = args.experiment_dir.resolve()
    decision = json.loads(args.decoder_decision.read_text(encoding="utf-8"))
    radius = float(decision["selected"]["radius_m"])

    sup = Supervisor(exp, args.budget_hours, radius)
    (exp / "wrapper.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    try:
        sup.run(decision)
    except Exception as exc:  # noqa: BLE001 - the wrapper must always leave a terminal
        sup.log(f"wrapper aborted: {type(exc).__name__}: {exc}")
        sup.report.setdefault("abort", {})["error"] = f"{type(exc).__name__}: {exc}"
        sup.finish("PARTIAL_FAMILY_RUNTIME_FAILURE")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
