"""Bind the metadata-only candidate separately from native GT/wire identity."""
import hashlib
from pathlib import Path
from rl_agent.splitfusion_live_gt_placement_trial_v1 import contract as P

ENVIRONMENT = dict(P.ENVIRONMENT)
BOOTSTRAP = "rl_agent.splitfusion_live_gt_metadata_v1.bootstrap"
# These are the unchanged placement scope, not a new placement experiment.
SCHEMA, MODES, TREATMENT_CPU = P.SCHEMA, P.MODES, P.TREATMENT_CPU
METADATA_SCHEMA = "splitfusion.gt_exact_metadata_trial.v1"
METADATA_MODES = ("control", "fast")


def sources():
    root = Path(__file__).resolve().parents[2]
    result = P.sources()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        if not path.name.startswith("test_"):
            if path.is_symlink() or not path.is_file():
                raise ValueError("regular metadata-candidate source required")
            result[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(result.items()))


def validate_context(context, expected):
    P.validate_context(context, expected)
    scope = context.get("runtime_metadata", {})
    if (scope.get("schema") != METADATA_SCHEMA
            or scope.get("mode") not in METADATA_MODES
            or scope.get("pid") != context.get("pid")
            or scope.get("validators_removed") is not False
            or scope.get("numerical_changes") is not False):
        raise ValueError("metadata candidate runtime context differs")
    return context
