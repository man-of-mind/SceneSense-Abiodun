"""Source-bound additive exact CPU GT scope; no launch authority or GPU work."""
from contextlib import contextmanager
import hashlib
from pathlib import Path

from rl_agent.splitfusion_live_parity_runtime_v4 import optimization_scope as V4
from rl_agent.splitfusion_live_parity_runtime_v4 import contract as C4
from rl_agent.splitfusion_live_gt_optimization_v1 import runtime as R1
from . import candidate

PACKAGE = Path(__file__).resolve().parent
settings = R1.settings
configuration_settings = V4.configuration_settings


def source_inventory():
    paths = sorted(p for p in PACKAGE.glob("*.py") if not p.name.startswith("test_"))
    if any(p.is_symlink() or not p.is_file() for p in paths):
        raise ValueError("regular v2 optimization sources required")
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def verify_sources(expected):
    if not isinstance(expected, dict) or not expected or expected != source_inventory():
        raise ValueError("GT optimization source inventory differs from prospective binding")
    R1.verify_sources(C4.OPTIMIZATION_SOURCES)


@contextmanager
def install_cpu_gt_optimizations(*, expected_sources):
    """Preserve v4 guards/prewarm/settings and put exact v2 inside its scope.

    Must enclose source ownership and the GT timing scope. The owned worker
    must drain before exit, exactly as enforced by the retained entry point.
    No write, fsync, validation, geometry or evaluation boundary is removed.
    """
    verify_sources(expected_sources)
    owned_sources = dict(expected_sources)
    state, retained = None, None
    failed = False
    try:
        with V4.install_cpu_gt_optimizations(expected_sources=C4.OPTIMIZATION_SOURCES) as retained:
            with candidate.install() as candidate_state:
                state = dict(retained, schema="splitfusion.cpu_gt_optimization.scope.v2",
                             sources=owned_sources, retained_v1_sources=dict(C4.OPTIMIZATION_SOURCES),
                             candidate=dict(candidate_state))
                yield state
    except BaseException:
        failed = True
        raise
    finally:
        if state is not None:
            # Retained scope records lifecycle/restoration after its own exit.
            for key, value in retained.items():
                if key not in ("schema", "sources"):
                    state[key] = value
            prior_error = retained.get("source_integrity_error")
            state["source_integrity_error"] = prior_error
            try:
                verify_sources(owned_sources)
            except Exception as exc:
                state["source_integrity_error"] = str(exc)
            if not failed and state["source_integrity_error"] is not None:
                raise RuntimeError(state["source_integrity_error"])
