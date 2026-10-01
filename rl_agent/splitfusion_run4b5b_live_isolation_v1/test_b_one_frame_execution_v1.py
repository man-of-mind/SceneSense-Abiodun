"""CPU-only proof for the distinct exactly-one operational loop."""

from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest
import uuid

from . import b_one_frame_execution_v1 as E
from . import b_ue_process_v1 as U
from . import b_validation_runner_v1 as V
from . import live_adapters_v1 as L
from . import operational_ack_v1 as A


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _request(root: Path, budget: int = 1) -> U.BUEProcessRequestV1:
    return U.BUEProcessRequestV1(
        run_id="one_frame_cpu", variant=L.ActorVariant.RUN4B,
        config_binding_sha256=_sha("config"),
        actor_boundary_sha256=_sha("actor"),
        feature_schema_sha256=_sha("features"),
        transmitted_budget=budget, deadline_ns=A.ACK_DEADLINE_NS,
        ack_semantics="TAIL_OUTPUT_READY__GT_FREE__SYNCHRONOUS_BEFORE_MAP",
        postrun_semantics="RAW_CARLA_GT_SPOOLED_LIVE__QPERC_MATERIALIZED_POSTRUN",
        clock_domain=A.CLOCK_DOMAIN,
        split_host=V.SplitHostBindingV1(
            carla_host="W10275.idcc.lab", ue_host="W10275.idcc.lab",
            cn_host="L10319.idcc.lab", edge_host="L10319.idcc.lab",
            ext_dn_host="L10319.idcc.lab",
            ack_receiver_host="W10275.idcc.lab", ack_receiver_port=51014),
        output_root=root / "output", evidence_root=root / "evidence",
        actor_manifest_path=root / "not-opened.json",
        required_authority_modules=U.REQUIRED_AUTHORITIES)


class _Pipeline:
    def __init__(self, request: U.BUEProcessRequestV1, *, include_gt: bool = False):
        self.variant = request.variant
        self.feature_schema_sha256 = request.feature_schema_sha256
        self.actor_boundary_sha256 = request.actor_boundary_sha256
        self.include_gt = include_gt
        self.calls = 0
        self.closed = False
        self.opened = 10_000_000_000

    def transmit_next(self, frame_index, previous):
        self.calls += 1
        if frame_index != 0 or previous is not None:
            raise AssertionError("one-frame loop requested another decision")
        identity = A.FrameActionIdentityV1(
            run_id="one_frame_cpu", cell_id="cell",
            stream_id="ue0_route_b",
            session_uuid=str(uuid.UUID(int=1)),
            controller_lineage_sha256=_sha("controller"),
            decision_seq=0, ticket_seq=0, frame_id=7, tensor_seq=0,
            capture_timestamp_ns=self.opened - 1_000_000,
            mode_id=11, q_e4=3000, keep_count=7000,
            anchor_action_id=None, profile_id=None,
            execution_bundle_sha256=_sha("bundle"))
        fields = ({"gt_objects": (), "gt_semantic_mask": object(),
                   "gt_recorded_monotonic_raw_ns": self.opened + 1}
                  if self.include_gt else {})
        return U.BTransmissionV1(
            identity=identity,
            action_open_monotonic_raw_ns=self.opened,
            payload_bytes=177_000, decision_frame=True, **fields)

    def close(self):
        self.closed = True


class _Receiver:
    def __init__(self, pipeline: _Pipeline):
        self.pipeline = pipeline
        self.closed = False

    def receive_until(self, identity, deadline_monotonic_raw_ns):
        ack = A.TailOutputAckV1.success(identity, b"tail-output")
        return (A.encode_ack(ack), self.pipeline.opened + 100_000_000)

    def close(self):
        self.closed = True


class OneFrameExecutionTest(unittest.TestCase):
    def test_exactly_one_success_and_sealed_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request = _request(Path(directory))
            pipeline = _Pipeline(request)
            receiver = _Receiver(pipeline)
            result = E.execute_one(request, pipeline, receiver)
            self.assertEqual(result["transmitted_frames"], 1)
            self.assertEqual(pipeline.calls, 1)
            self.assertTrue(pipeline.closed and receiver.closed)
            report = request.output_root / E.REPORT_NAME
            self.assertTrue(report.is_file())
            text = report.read_text(encoding="ascii").lower()
            self.assertIn('"live_qperc_computed":false', text)
            self.assertIn('"ground_truth_records":0', text)

    def test_refuses_300_and_never_calls_frozen_validator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request = _request(Path(directory), budget=300)
            pipeline = _Pipeline(request)
            receiver = _Receiver(pipeline)
            with self.assertRaisesRegex(E.OneFrameExecutionError,
                                        "not exactly one"):
                E.execute_one(request, pipeline, receiver)
            self.assertEqual(pipeline.calls, 0)

    def test_refuses_live_gt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request = _request(Path(directory))
            pipeline = _Pipeline(request, include_gt=True)
            receiver = _Receiver(pipeline)
            with self.assertRaisesRegex(E.OneFrameExecutionError,
                                        "live CARLA GT"):
                E.execute_one(request, pipeline, receiver)
            self.assertTrue(pipeline.closed and receiver.closed)


if __name__ == "__main__":
    unittest.main()
