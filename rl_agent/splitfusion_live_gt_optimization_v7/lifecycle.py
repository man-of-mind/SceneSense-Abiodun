"""Normalize the retained edge shutdown protocol inside an active handoff.

This helper is not wired into a live entry. A future registered caller must
invoke it inside the existing handoff scope. That scope remains responsible
for broker shutdown and its final success/failure decision.
"""
from types import CodeType

SCHEMA = "splitfusion.retained_edge_shutdown_normalization.v1"


def _retained_shutdown_code():
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_edge_runtime_v2 as E
    matches = [item for item in E.run_run4_edge_service.__code__.co_consts
               if isinstance(item, CodeType) and item.co_name == "shutdown"]
    if len(matches) != 1:
        raise RuntimeError("retained edge shutdown code seam differs")
    return matches[0]


def _clean_close(state):
    """Require observations already made by the genuine composed close."""
    if (not isinstance(state, dict)
            or state.get("schema") != "splitfusion.gt_packet_edge_scope.v1"
            or state.get("errors") != [] or state.get("complete") is not False):
        return False
    close = state.get("evaluator_close") or {}
    prediction = state.get("prediction_state") or {}
    pose = state.get("pose_state") or {}
    return (close.get("worker_alive") is False
        and close.get("parity_archive_complete") is True
        and close.get("parity_archive_errors") == []
        and type(close.get("submitted")) is int
        and close["submitted"] == close.get("emitted") == close.get("records")
        and prediction.get("flushed") is True and prediction.get("archive_errors") == []
        and pose.get("flushed") is True and pose.get("errors") == [])


def dispatch_with_retained_shutdown(dispatch, *args, handoff_state, **kwargs):
    """Return 0 only for the retained handler's verified clean SystemExit(0).

    Usage within a future registered edge entry::

        with install_edge_handoff(...) as state:
            result = dispatch_with_retained_shutdown(
                retained_dispatch, argv, handoff_state=state)

    Normal return values are unchanged. Nonzero/foreign exits, incomplete
    evaluator close, and all other exceptions propagate unchanged. Broker
    drain still occurs afterwards in the enclosing unmodified handoff scope;
    its errors cannot be intercepted by this dispatch helper.
    """
    try:
        return dispatch(*args, **kwargs)
    except SystemExit as exc:
        if type(exc.code) is not int or exc.code != 0:
            raise
        origin = exc.__traceback__
        while origin is not None and origin.tb_next is not None:
            origin = origin.tb_next
        if (origin is None or origin.tb_frame.f_code is not _retained_shutdown_code()
                or not _clean_close(handoff_state)):
            raise
        # Add an honest adapter receipt; never replace errors, complete, broker,
        # prediction, pose, or evaluator-close observations.
        handoff_state["retained_shutdown"] = dict(schema=SCHEMA, exitcode=0,
            origin="phase6_edge_runtime_v2.run_run4_edge_service.shutdown",
            original_close_verified_before_normalization=True,
            broker_drain_still_required=True, live_qualified=False)
        return 0
