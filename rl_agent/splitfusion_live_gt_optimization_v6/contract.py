"""Separate source identity for the candidate and its retained startup repair."""
import hashlib
from pathlib import Path
from rl_agent.splitfusion_live_gt_optimization_v5 import contract as V5

ENVIRONMENT = dict(V5.ENVIRONMENT)
BOOTSTRAP = "rl_agent.splitfusion_live_gt_optimization_v6.bootstrap"
validate_context = V5.validate_context


def sources():
    root = Path(__file__).resolve().parents[2]
    result = V5.sources()
    paths = sorted(Path(__file__).parent.glob("*.py"))
    for path in paths:
        if not path.name.startswith("test_"):
            if path.is_symlink() or not path.is_file():
                raise ValueError("regular depth-reuse source required")
            result[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(result.items()))
