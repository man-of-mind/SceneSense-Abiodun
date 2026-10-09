"""Bind placement observation independently of the unchanged handoff wire."""
import hashlib
from pathlib import Path

from rl_agent.splitfusion_live_gt_optimization_v8 import control_contract as V8

ENVIRONMENT = dict(V8.ENVIRONMENT)
BOOTSTRAP = "rl_agent.splitfusion_live_gt_placement_v1.bootstrap"
SCHEMA = "splitfusion.dedicated_gt_placement.v1"
validate_context = V8.validate_context


def sources():
    root = Path(__file__).resolve().parents[2]
    result = V8.sources()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        if not path.name.startswith("test_"):
            if path.is_symlink() or not path.is_file():
                raise ValueError("regular placement source required")
            result[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(result.items()))
