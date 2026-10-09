"""Create a code-reading snapshot, not an executable experiment deployment.

Original sources are only read. Selected copies are verified byte-for-byte.
No inference, imports of project modules, experiments, or external services run.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT.parent / "abiodun_run4_single_gpu_profiler_v1"
REG = DEPLOY / "rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261007_m3_route_b_v6/good_all_prospective_02/registration.json"
MODEL = "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
ACTOR_SHA = "bb9b0d256de79b4498cd622e1daa59edcf43ff3072dc9f675b265aac579ed3ce"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select():
    reg = json.loads(REG.read_text())
    files, reasons = {}, {}

    def add(relative, reason, expected=None, preferred=None):
        relative = str(relative)
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("unsafe source path: " + relative)
        options = [preferred] if preferred else [ROOT / relative, DEPLOY / relative]
        candidates = [p for p in options if p and p.is_file() and (expected is None or sha(p) == expected)]
        if not candidates:
            raise FileNotFoundError("missing or mismatched selected source: " + relative)
        chosen = candidates[0]
        if relative in files and sha(files[relative]) != sha(chosen):
            raise ValueError("conflicting source versions: " + relative)
        files[relative] = chosen
        reasons.setdefault(relative, set()).add(reason)

    # The actual most-recent runtime, not a guessed selection of version names.
    for relative, expected in reg["exact_gt_sources"].items():
        add(relative, "registered V6 runtime", expected)
    for row in reg["experiment_binds"].values():
        p = Path(row["path"])
        if p.suffix != ".py":
            continue
        for base in (ROOT, DEPLOY):
            if p.is_relative_to(base):
                relative = str(p.relative_to(base))
                # Some older dependencies were verified-archived; prefer a
                # matching live copy if the original absolute path is absent.
                add(relative, "registered V6 launcher dependency", row["sha256"])
                break

    packages = [
        MODEL + "splitfusion_fcos_r50_fpn_p2_p7_v1",
        MODEL + "splitfusion_fcos_r50_fpn_p2_p7_ae_v1",
        MODEL + "splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1",
        MODEL + "splitfusion_fcos_r50_fpn_p2_p7_service_candidate_v1",
        MODEL + "splitfusion_fcos_r50_fpn_p2_p7_person_p025_calibration_v1",
        "rl_agent/splitfusion_run5_full_v1",
        "rl_agent/splitfusion_hybrid_sac_run5_v1",
    ]
    for package in packages:
        for p in sorted((ROOT / package).glob("*.py")):
            if not p.name.startswith("test_"):
                add(p.relative_to(ROOT), "model or deployed-agent training implementation")
    explicit = [
        MODEL + "splitfusion_fcos_r50_fpn_p2_p7_v1/config.json",
        MODEL + "splitfusion_fcos_r50_fpn_p2_p7_v1/transport_schema.json",
        MODEL + "splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1/locked_config.json",
        "rl_agent/splitfusion_action_catalog_v1/splitfusion_72_action_catalog.json",
        "rl_agent/splitfusion_live_dispatch_v1/runtime_binding.json",
        "rl_agent/splitfusion_live_dispatch_v1/frame_context_binding.json",
        "rl_agent/splitfusion_live_dispatch_v1/dynamic_execution_source_binding.json",
        "rl_agent/splitfusion_hybrid_sac_v1/hybrid_sac_models.py",
        "rl_agent/splitfusion_hybrid_sac_v1/modeled_smoke_support.py",
        "rl_agent/splitfusion_hybrid_sac_v1/action_contract.py",
        "rl_agent/splitfusion_hybrid_sac_run4_v1/run4_contract.py",
        "rl_agent/splitfusion_run5_full_live_v1/actor_export/COMMITTED",
        "rl_agent/configs/network_profile_design_v2.json",
        "rl_agent/configs/splitfusion_direct_edge_map_live_validation_v1.json",
        "rl_agent/configs/oai_radio_baseline_100mhz_4d5u_v1.json",
        "data_collection/routes/town10hd_opt_route_b_full_map_loop_v1.json",
        "data_collection/routes/town10hd_opt_route_b_full_map_loop_v1.progress.csv",
        "experiments/splitfusion_phase14a_100mhz_calibration_v1/20260904_phase14a_live_calibration_retry3/target_to_rfsim_mapping.csv",
    ]
    for relative in explicit:
        add(relative, "architecture contract or runtime configuration")
    oai = ROOT / "OAI/openairinterface5g"
    changed = subprocess.check_output(["git", "diff", "--name-only", "HEAD"], cwd=oai, text=True).splitlines()
    for relative in changed:
        if Path(relative).suffix in (".c", ".h") or relative == "common/utils/T/T_messages.txt":
            add("OAI/openairinterface5g/" + relative, "local OAI telemetry or scheduler source modification")

    # Local static import closure. Dynamic imports and external artifact loads
    # are deliberately not claimed to be covered by this reading snapshot.
    visited = set()

    def module_path(module, sibling=None):
        rel = module.replace(".", "/")
        for p in (rel + ".py", rel + "/__init__.py"):
            if p in files or (ROOT / p).is_file() or (DEPLOY / p).is_file():
                return p
        if sibling and "." not in module:
            p = str(Path(sibling).parent / (module + ".py"))
            if p in files or (ROOT / p).is_file() or (DEPLOY / p).is_file():
                return p
        return None

    while True:
        pending = [p for p in files if p.endswith(".py") and p not in visited]
        if not pending:
            break
        for relative in pending:
            visited.add(relative)
            tree = ast.parse(files[relative].read_bytes(), filename=relative)
            package = str(Path(relative).parent).replace("/", ".")
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    if node.level:
                        parts = package.split(".")
                        base = ".".join(parts[:len(parts) - node.level + 1])
                        base = ".".join(x for x in (base, node.module or "") if x)
                    else:
                        base = node.module or ""
                    modules = [base] + [base + "." + alias.name for alias in node.names if alias.name != "*"]
                for module in modules:
                    p = module_path(module, relative)
                    if p and p not in files:
                        add(p, "local static-import dependency")
            for parent in Path(relative).parents:
                init = str(parent / "__init__.py")
                if parent != Path(".") and init not in files and ((ROOT/init).is_file() or (DEPLOY/init).is_file()):
                    add(init, "package initializer")
        if len(files) > 1800:
            raise RuntimeError("source snapshot exceeds intended reading scope")

    assert sha(files[reg["policy"]["weights_path"]]) == ACTOR_SHA
    return reg, files, reasons


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    reg, files, reasons = select()
    print(json.dumps(dict(source_files=len(files), source_bytes=sum(p.stat().st_size for p in files.values()),
                         extension_counts=dict(Counter(Path(p).suffix for p in files))), indent=2))
    if args.plan:
        return
    if args.output is None:
        parser.error("--output required unless --plan")
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "handoffs"):
        raise ValueError("output must be under this checkout's handoffs directory")
    output.mkdir(parents=True, exist_ok=False)
    records = []

    def copy(source, destination, reason):
        target = output / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        expected = sha(source)
        shutil.copyfile(source, target)
        if sha(target) != expected or sha(source) != expected:
            raise RuntimeError("source changed during copy: " + str(source))
        records.append(dict(path=destination, source=str(source), sha256=expected,
                            bytes=target.stat().st_size, reason=reason))

    for relative, source in sorted(files.items()):
        copy(source, "code/" + relative, sorted(reasons[relative]))
    for name in ("REPLAY_REWARD.png", "REPLAY_REWARD.csv", "TRAINING_DECISIONS.png", "TRAINING_DECISIONS.csv"):
        copy(ROOT / "experiments/splitfusion_run5_full_v1/report_01" / name,
             "agent_training_plots/" + name, "requested curves from deployed actor campaign; seeds 17,29,43")
    for name in ("__init__.py", "repair.py", "worker_entry.py", "test_repair.py"):
        copy(ROOT / "rl_agent/splitfusion_gt_freeze_lifetime_v1" / name,
             "pending_fix/splitfusion_gt_freeze_lifetime_v1/" + name,
             "offline-tested candidate; NOT part of V6 live runtime")
    copy(ROOT / "experiments/splitfusion_gt_freeze_lifetime_v1/REPAIR_STATUS.json",
         "pending_fix/REPAIR_STATUS.json", "current repair scope and limitations; no live records")
    copy(REG.parent / "capture/m3_radio/radio_state_copy/effective_gnb_100mhz_4d5u_clean_minus50.conf",
         "radio_config/effective_gnb_100mhz_4d5u.conf", "exact gNB configuration from latest V6 run; configuration only")
    copy(Path(__file__).resolve(), "build_snapshot.py", "selection and byte-copy procedure; uses original checkout paths")
    write_json(output / "SOURCE_MANIFEST.json", dict(schema="supervisor.architecture_source_snapshot.v1",
        purpose="architecture and data-structure review; not a standalone runnable release",
        created_utc=datetime.now(timezone.utc).isoformat(), source_root=str(ROOT), deployment_root=str(DEPLOY),
        main_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        includes_uncommitted_work=True, current_runtime=reg["runtime_revision"],
        actor_sha256=ACTOR_SHA, model_weight_files_included=False, actor_weight_file_included=True,
        no_experiment_run=True, no_upload=True, files=records))
    binding=json.loads(files["rl_agent/splitfusion_live_dispatch_v1/runtime_binding.json"].read_bytes())
    write_json(output / "MODEL_WEIGHT_REFERENCES.json", dict(
        note="Paths and hashes only. Large FCOS, AE and ranker checkpoints are not copied.",
        selected_checkpoints=binding["selected_checkpoints"], family_routes=binding["family_routes"]))
    write_json(output / "RUNTIME_OVERVIEW.json", {k:reg[k] for k in (
        "runtime_revision", "launcher_binding", "policy", "selection", "route", "channel",
        "frozen_deadline_ms", "map_class", "numeric_prediction_contract")})
    actor=json.loads(files["rl_agent/splitfusion_run5_full_live_v1/actor_export/ACTOR_BINDING.json"].read_bytes())
    with (output / "AGENT_FEATURES.csv").open("x", newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(["zero_based_index", "feature_name"])
        writer.writerows(enumerate(actor["binding"]["policy_feature_order"]))
    with (output / "ACTION_MODES.csv").open("x", newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(["mode_id", "family", "quantizer", "min_q_e4", "max_q_e4"])
        for mode, bounds in enumerate(actor["support"]["bounds"]):
            writer.writerow([mode, ("noAE","AE128","AE64","AE32")[mode//3],
                             ("UINT8","UINT6","UINT4")[mode%3], *bounds])
    oai=ROOT/"OAI/openairinterface5g"
    write_json(output / "EXTERNAL_DEPENDENCIES.json", dict(
        oai=dict(local_source=str(oai), head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=oai,text=True).strip(),
                 note="Only modified telemetry/scheduler sources are copied, not the complete OAI/core-network repository or container images."),
        carla="CARLA 0.10.0 installation and Python API are external; not copied.",
        perception="FCOS, AE and ranker weights are external; see MODEL_WEIGHT_REFERENCES.json.",
        training="Measured calibration chains, quality database, datasets and durable training bundles are intentionally omitted.",
        runtime="Requires original environment, deployed container images, approved registrations and source pins. This snapshot is for reading, not an authorized launch bundle."))
    print(json.dumps(dict(output=str(output), copied_files=len(records), copied_bytes=sum(r["bytes"] for r in records))))


if __name__ == "__main__":
    main()
