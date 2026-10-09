"""Reversible process-local seams; frozen runtime files remain unchanged.

Call these before the retained profiler installs its wrappers. No function
here launches a service. Full live qualification remains a separate gate.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import time

from .prediction_archive import PredictionArchive


@contextmanager
def install_ue_perception():
    from rl_agent import ue_route_b_split_cell_adapter_v1 as P
    from rl_agent.splitfusion_live_dispatch_v1 import live_pilot_runtime as B
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_prewarm_v2 as PW
    from . import perception
    perception.verify_reference_sources(Path(__file__).resolve().parents[2])
    collector = P.PassiveSplitCollector
    spawn, packing = collector._spawn_sensors, B._prepare_live_input
    warm_radar_shape = PW.RADAR_SHAPE

    def corrected_spawn(self):
        # Frozen constructor invokes _spawn_sensors before starting workers.
        perception.bind_collector_geometry(self)
        return spawn(self)

    collector._spawn_sensors = corrected_spawn
    B._prepare_live_input = perception.prepare_live_input
    PW.RADAR_SHAPE = (4, perception.CONTRACT.content_height, perception.CONTRACT.content_width)
    try:
        yield {"perception_contract_sha256": perception.CONTRACT.sha256,
               "scene_descriptor_contract": "UNCHANGED", "new_threads": 0}
    finally:
        collector._spawn_sensors = spawn
        B._prepare_live_input = packing
        PW.RADAR_SHAPE = warm_radar_shape


@contextmanager
def install_edge_evaluation(*, archive_directory: Path, max_frames: int):
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_edge_runtime_v2 as E
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_map_protocol_v2 as MP
    from rl_agent.splitfusion_quality_feedback_probe_v1 import gt_evidence as G
    from rl_agent.splitfusion_run5_live_avo_v1 import live_gt
    from . import evaluation, target_transport
    evaluation.verify_sources()  # before any route/frame or worker
    archive_directory = Path(archive_directory)
    if archive_directory.exists():
        raise FileExistsError("prediction archive is create-only")
    archive = PredictionArchive(max_frames=max_frames)
    evaluator_class = E.Run4EvaluatorV2
    old = (G.read_ground_truth, E.Run4EvaluatorV2._evaluate_with,
           E.Run4EvaluatorV2.submit, E.Run4EvaluatorV2.close,
           E.Run4EvaluatorV2.GT_COMPONENTS, W.LIVE_MEASUREMENT_SEMANTICS,
           E.Run4EvaluatorV2._observe_components)
    state = {"archive": archive, "archive_errors": [], "flushed": False}

    def read(directory, *, expected_identity, timeout_s):
        start = time.monotonic()
        bundle = target_transport.read_bundle(Path(directory), expected_identity=expected_identity,
                                              timeout_s=timeout_s)
        remaining = max(0.000001, timeout_s - (time.monotonic() - start))
        avo = live_gt.read_avo_bundle(Path(directory), expected_identity=expected_identity,
                                      timeout_s=remaining)
        return {"prepared": bundle.prepared, "manifest_sha256": bundle.manifest_sha256, **avo,
                "gt_ready_wall_ns": max(bundle.manifest["ready_wall_ns"], avo["avo_ready_wall_ns"])}

    def submit(self, ticket):
        try:
            archive.add(ticket)
            ticket = archive.owned_ticket(ticket)
        except Exception as exc:
            state["archive_errors"].append(f"submit: {type(exc).__name__}: {exc}")
            with self._lock:
                self.counters["parity_archive_errors"] = self.counters.get("parity_archive_errors", 0) + 1
            self._emit(ticket, None, W.EvaluatorReason.EVALUATOR_EXCEPTION, infrastructure=True)
            return
        return old[2](self, ticket)

    def observe(self, ticket, timing):
        if self._evidence_dir is None:
            return
        handoff = timing.setdefault("gt_handoff", {})
        if "components" not in handoff:
            root = self._evidence_dir
            handoff.update(identity=dict(ticket.gt_identity), read_errors={}, components={})
            paths = {
                "parity_target_ready": target_transport.bundle_path(root, ticket.gt_identity) / target_transport.READY_NAME,
                "avo": live_gt.sidecar_path(root, ticket.gt_identity),
            }
            for name, path in paths.items():
                handoff["components"][name] = {"container_path": str(path),
                    "first_observed_wall_ns": None, "size_bytes": None}
        for item in handoff["components"].values():
            path = Path(item["container_path"])
            if item["first_observed_wall_ns"] is None and path.is_file():
                item.update(first_observed_wall_ns=time.time_ns(), size_bytes=path.stat().st_size)

    def evaluate(self, ticket, gt, timing):
        timing["evaluator_start_wall_ns"] = time.time_ns()
        value = evaluation.evaluate_prepared(identity=MP.run4_identity(ticket.envelope),
            prepared=gt["prepared"], predicted_labels=ticket.predicted_mask,
            predictions=list(ticket.records), avo=gt["avo"])
        feedback, reason = W.quality_feedback(self._spec, value.measurement, ticket.envelope)
        timing.update(evaluator_end_wall_ns=time.time_ns(), gt_ready_wall_ns=gt["gt_ready_wall_ns"],
                      target_bundle_sha256=gt["manifest_sha256"], avo_sidecar_sha256=gt["avo_sidecar_sha256"])
        self._emit(ticket, feedback, reason, measurement=value.measurement, timing=timing)
        # The full original prediction was already copied at submit. Add match
        # identities after feedback; no PNG/disk serialization on its clock.
        try:
            archive.evaluated(ticket, value.evidence)
        except Exception as exc:
            state["archive_errors"].append(f"evaluate: {type(exc).__name__}: {exc}")
            with self._lock:
                self.counters["parity_archive_errors"] = self.counters.get("parity_archive_errors", 0) + 1

    def close(self, timeout_s=None):
        result = old[3](self, timeout_s)
        if result["worker_alive"]:
            state["archive_errors"].append("evaluator worker failed to drain; refusing evidence flush")
            return {**result, "parity_archive_complete": False}
        if state["flushed"]:
            raise RuntimeError("evaluator close repeated")
        evidence = archive.flush(archive_directory)
        state["flushed"] = True
        state["manifest"] = evidence
        return {**result, "parity_archive_complete": not state["archive_errors"],
                "parity_archive": evidence, "parity_archive_errors": list(state["archive_errors"])}

    G.read_ground_truth = read
    E.Run4EvaluatorV2._evaluate_with = evaluate
    E.Run4EvaluatorV2.submit = submit
    E.Run4EvaluatorV2.close = close
    E.Run4EvaluatorV2.GT_COMPONENTS = ("parity_target.json", "avo.json")
    E.Run4EvaluatorV2._observe_components = observe
    W.LIVE_MEASUREMENT_SEMANTICS = "V010_NUMERICAL_RULES__PROSPECTIVE_LIVE_CONSERVATIVE_CONFLICTS__PERSON_AVO_GE065"
    try:
        yield state
    finally:
        # The profiler installs a subclass after this context enters. Restore
        # our exact original class, not whichever subclass now occupies E.
        (G.read_ground_truth, evaluator_class._evaluate_with,
         evaluator_class.submit, evaluator_class.close,
         evaluator_class.GT_COMPONENTS, W.LIVE_MEASUREMENT_SEMANTICS,
         evaluator_class._observe_components) = old
