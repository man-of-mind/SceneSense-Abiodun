"""Bind startup GC treatment separately from unchanged GT/wire contracts."""
import hashlib
from pathlib import Path
from rl_agent.splitfusion_live_gt_handoff_v1 import contract as H

ENVIRONMENT = dict(H.ENVIRONMENT)
BOOTSTRAP = "rl_agent.splitfusion_live_gt_optimization_v8.bootstrap"
validate_context = H.validate_context
SCHEMA = "splitfusion.gt_worker_startup_gc.v1"


def sources():
    root = Path(__file__).resolve().parents[2]
    result = H.sources()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        if not path.name.startswith("test_"):
            if path.is_symlink() or not path.is_file():
                raise ValueError("regular GC candidate source required")
            result[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(result.items()))
