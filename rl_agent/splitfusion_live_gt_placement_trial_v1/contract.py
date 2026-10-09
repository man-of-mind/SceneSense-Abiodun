"""Source-bound matched worker arms; native handoff wire identity is unchanged."""
import hashlib
from pathlib import Path
from rl_agent.splitfusion_live_gt_main_thread_placement_v1 import contract as V10

ENVIRONMENT = dict(V10.ENVIRONMENT)
BOOTSTRAP = "rl_agent.splitfusion_live_gt_placement_trial_v1.bootstrap"
SCHEMA = "splitfusion.gt_main_thread_live_trial.worker.v1"
MODES = ("control", "p4")
TREATMENT_CPU = 4
validate_context = V10.validate_context


def sources():
    root = Path(__file__).resolve().parents[2]
    result = V10.sources()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        if not path.name.startswith("test_"):
            if path.is_symlink() or not path.is_file():
                raise ValueError("regular matched-trial source required")
            result[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(result.items()))
