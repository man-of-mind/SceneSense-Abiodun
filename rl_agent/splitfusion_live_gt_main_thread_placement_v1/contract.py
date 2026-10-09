"""Separate source binding for the narrower main-TID-only intervention."""
import hashlib
from pathlib import Path
from rl_agent.splitfusion_live_gt_placement_v1 import contract as V9

ENVIRONMENT = dict(V9.ENVIRONMENT)
BOOTSTRAP = "rl_agent.splitfusion_live_gt_main_thread_placement_v1.bootstrap"
SCHEMA = "splitfusion.gt_main_thread_placement.v1"
validate_context = V9.validate_context


def sources():
    root = Path(__file__).resolve().parents[2]
    result = V9.sources()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        if not path.name.startswith("test_"):
            if path.is_symlink() or not path.is_file():
                raise ValueError("regular main-thread placement source required")
            result[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(result.items()))
