"""Reversible, pre-worker installation of the v2 prediction contract.

Keep the retained evaluator, timers, Q_perc rules, pose wrappers, and original
Run-5 actor unchanged. This scope must surround the retained edge pose scope,
which in turn wraps install_edge_evaluation. Merely entering it launches nothing.
"""
from contextlib import contextmanager
from types import FunctionType


@contextmanager
def install_prediction_contract():
    from rl_agent.splitfusion_live_parity_repair_v1 import runtime as retained
    from .prediction_archive import PredictionArchive
    original = retained.PredictionArchive
    retained.PredictionArchive = PredictionArchive
    try:
        yield {"archive_class": PredictionArchive.__module__ + ".PredictionArchive",
               "native_prediction_preserved": True, "no_new_thread": True}
    finally:
        retained.PredictionArchive = original


@contextmanager
def install_postflight_reader():
    """The exact retained composed verifier, with only its archive reader replaced.

The inherited population, original numerical replay, pose, GT/AVO, and source
hash checks all remain mandatory. Nothing can turn a failed capture into PASS.
"""
    from rl_agent.splitfusion_live_parity_runtime_v1 import postflight as retained
    from . import readiness
    original = retained.verify_cell
    replacement = FunctionType(original.__code__, dict(original.__globals__, R=readiness),
        original.__name__, original.__defaults__, original.__closure__)
    replacement.__kwdefaults__ = original.__kwdefaults__
    def verified(cell):
        from pathlib import Path
        from .source_ownership import verify_ownership_archive
        from .postflight_checks import verify_gt_timing
        result = replacement(cell)
        population = retained._population(cell)
        ownership = verify_ownership_archive(Path(cell) / "source_ownership",
            expected_identities=population["gt"],
            expected_action_identities={frame: row["run4_identity"] for frame, row in population["sent"].items()})
        timing = verify_gt_timing(Path(cell) / "complete_gt_stage_timing", expected_frames=population["gt"])
        return {**result, "exact_source_ownership": ownership, "complete_gt_stage_timing": timing}
    retained.verify_cell = verified
    try:
        yield
    finally:
        retained.verify_cell = original
