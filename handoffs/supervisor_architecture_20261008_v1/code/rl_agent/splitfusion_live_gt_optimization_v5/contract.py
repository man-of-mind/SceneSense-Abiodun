"""Small stdlib-only startup contract, separate from the frozen V4 GT body."""
import hashlib
from pathlib import Path

ENVIRONMENT = {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
               "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
BOOTSTRAP = "rl_agent.splitfusion_live_gt_optimization_v5.bootstrap"


def sources():
    root = Path(__file__).resolve().parents[2]
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.glob("*.py"))
            if not path.name.startswith("test_")}


def validate_context(context, expected_sources):
    if not isinstance(context, dict):
        raise ValueError("CPU worker effective startup context missing")
    if not expected_sources or context.get("bootstrap_sources") != expected_sources:
        raise ValueError("CPU worker bootstrap source binding differs")
    if context.get("environment") != ENVIRONMENT:
        raise ValueError("CPU worker effective environment/caps differ")
    if type(context.get("torch_threads")) is not int or context["torch_threads"] != 1:
        raise ValueError("CPU worker effective Torch intra-op cap differs")
    if context.get("cuda_initialized") is not False:
        raise ValueError("CPU worker initialized CUDA")
    # Inter-op/OpenCV counts are observed, not changed: no new tuning is
    # smuggled into the repair for the existing registered intra-op cap.
    return context
