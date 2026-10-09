"""Add-only deployment, preserving every sealed v8 source byte."""
import argparse
import json
from pathlib import Path
from rl_agent.splitfusion_live_parity_runtime_v8 import deploy as V8
from . import contract as C
from .bindings import namespace, specialize

ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = ROOT / "experiments/splitfusion_live_parity_runtime_v8/20261004_handoff_live_v8/qualification_100/PROSPECTIVE.json"
_bound = namespace(V8, C=C, ROOT=ROOT, PREVIOUS=PREVIOUS)
specialize(V8.deploy, _bound, [
    ('"splitfusion.parity_runtime.prospective.v7"', '"splitfusion.parity_runtime.prospective.v8"'),
    ('for package in (C.PACKAGE, *C.BOOTSTRAP_PACKAGES):',
     'for package in (C.PACKAGE, *C.BOOTSTRAP_PACKAGES, "rl_agent/splitfusion_live_gt_optimization_v7"):'),
    ('schema="splitfusion.parity_runtime.deployment.v8"', 'schema="splitfusion.parity_runtime.deployment.v9"')])
deploy = _bound["deploy"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(json.dumps(deploy(args.worktree, args.report, execute=args.execute)))
