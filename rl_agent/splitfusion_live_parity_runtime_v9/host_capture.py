"""Fresh per-arm registrations over the retained bounded host launcher."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from rl_agent.splitfusion_live_parity_runtime_v8 import host_capture as V8
from rl_agent.splitfusion_live_parity_runtime_v8 import contract as C8
from . import contract as C
from .bindings import namespace, specialize

WORKER_CLIENT_FACTORY = "rl_agent.splitfusion_live_parity_runtime_v9.worker_observer.ObservedClient"
OFFLINE_SUITES = V8.OFFLINE_SUITES | {"integration_v9", "placement_trial", "lifecycle_and_process_scope"}
_bound = namespace(V8, C=C, WORKER_CLIENT_FACTORY=WORKER_CLIENT_FACTORY,
                   OFFLINE_SUITES=OFFLINE_SUITES, __file__=__file__)
# The immutable original handoff proof binds the unchanged wire/GT algorithm;
# new worker behavior has its own fresh regression source binding.
_proof_contract = SimpleNamespace(**dict(vars(C), bootstrap_inventory=C8.bootstrap_inventory))
_proof_bound = namespace(V8, C=_proof_contract)
_bound["verify_handoff_proof"] = _proof_bound["verify_handoff_proof"]
_original_verify = _bound["verify_prospective"]


def verify_prospective(output):
    result = _original_verify(output)
    registration = json.loads((Path(output) / "registration.json").read_bytes())
    mode = result.get("trial", {}).get("mode")
    if result.get("trial") != C.trial(mode) or registration.get("trial") != result["trial"]:
        raise ValueError("prospective main-thread trial mode differs")
    return result


_bound["verify_prospective"] = verify_prospective
specialize(V8.prepare, _bound, [
    ("def prepare(output, worktree, retained_code, *, offline, sync_probe, startup_scope, worker_proof, handoff_proof):",
     "def prepare(output, worktree, retained_code, *, offline, sync_probe, startup_scope, worker_proof, handoff_proof, trial_mode):"),
    ("reg.update(id=C.ID, child_module=C.CHILD, edge_module=C.EDGE,",
     "reg.update(trial=C.trial(trial_mode), id=C.ID, child_module=C.CHILD, edge_module=C.EDGE,"),
    ("record = dict(schema=\"splitfusion.parity_runtime.prospective.v8\", budget=C.BUDGET,",
     "record = dict(trial=C.trial(trial_mode), schema=\"splitfusion.parity_runtime.prospective.v8\", budget=C.BUDGET,")])
for _name in ("publisher_binding", "verify_offline", "verify_startup_scope", "verify_worker_proof",
              "verify_handoff_proof", "verify_prerequisite", "prepare", "launch", "postflight"):
    globals()[_name] = _bound[_name]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "launch", "postflight"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trial-mode", choices=("control", "p4"))
    for name in ("worktree", "retained-code", "offline", "sync-probe", "startup-scope", "worker-proof", "handoff-proof"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--execute", default="")
    args = parser.parse_args(argv)
    if args.operation == "prepare":
        if args.trial_mode is None:
            parser.error("prepare requires --trial-mode")
        result = prepare(args.output, args.worktree, args.retained_code, offline=args.offline,
            sync_probe=args.sync_probe, startup_scope=args.startup_scope, worker_proof=args.worker_proof,
            handoff_proof=args.handoff_proof, trial_mode=args.trial_mode)
    elif args.operation == "launch":
        launched = launch(args.output, execute_token=args.execute)
        result = postflight(args.output)
        print(json.dumps(result))
        return 0 if launched and result["passed"] else 1
    else:
        result = postflight(args.output)
    print(json.dumps(result))
    return 0 if result.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
