"""Focused CPU-only tests for the additive GT-free edge service."""

from __future__ import annotations

import dataclasses
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest import mock

import numpy as np

from rl_agent.splitfusion_run4b5b_live_isolation_v1 import (
    b_edge_process_v1 as E,
)
from rl_agent.splitfusion_run4b5b_live_isolation_v1 import (
    b_edge_service_v2 as S,
)
from rl_agent.splitfusion_run4b5b_live_isolation_v1 import (
    branch_evidence_v1 as B,
)
from rl_agent.splitfusion_run4b5b_live_isolation_v1 import (
    live_adapters_v1 as L,
)
from rl_agent.splitfusion_run4b5b_live_isolation_v1 import (
    tail_evidence_codec_v2 as C,
)


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def succeeded(code: str) -> L.BranchCallbackResultV1:
    return L.BranchCallbackResultV1(
        status=B.BranchStatus.SUCCEEDED,
        detail_code=code,
        evidence_sha256=sha(code),
    )


def phase6_parts(*, reward_requested: bool = True):
    envelope = types.SimpleNamespace(
        reward_requested=reward_requested,
        session_uuid="00000000-0000-4000-8000-000000000001",
        controller_lineage_sha256=sha("controller"),
        decision_seq=3, ticket_seq=3, frame_id=12, tensor_seq=12,
        capture_timestamp_ns=100, mode_id=11, q_e4=3000, keep_count=2,
        anchor_action_id=67, execution_bundle_sha256=sha("bundle"),
    )
    context = types.SimpleNamespace(
        stream_id="ego_rgb", sequence_id=12, frame_id=12,
        capture_timestamp_ns=100)
    profile = types.SimpleNamespace(
        mode_id=11, q_e4=3000, keep_count=2, action_id=67,
        profile_id="split_ae32_uint4_q3000",
        execution_bundle_sha256=sha("bundle"))
    return envelope, context, profile


class FakeProcessor:
    def __init__(self, *, reward_requested: bool = True,
                 omit_output: bool = False) -> None:
        self.envelope, self.context, self.profile = phase6_parts(
            reward_requested=reward_requested)
        self.omit_output = omit_output

    def verify(self, _payload):
        return self.envelope, self.context, self.profile, {"verified": True}

    def process(self, _payload, *, edge_timing):
        evaluation = None
        if self.envelope.reward_requested and not self.omit_output:
            evaluation = types.SimpleNamespace(
                records=({"class_name": "person"},),
                predicted_mask=np.asarray([[0, 1]], dtype=np.uint8))
        return types.SimpleNamespace(
            envelope=self.envelope, context=self.context,
            profile=self.profile,
            update={"frame_id": self.context.frame_id,
                    "edge_timing": dict(edge_timing)},
            evaluation=evaluation)


class ExactDispatchHarness:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.packets: list[bytes] = []
        self.decoded: list[C.TailEvidenceV2] = []
        self.registry = E.MapDocumentRegistryV1(
            lambda document: self.events.append("MAP") or {
                "frame_id": document["frame_id"]})

        def prediction(work):
            self.decoded.append(C.decode(
                work.tail_output, expected_identity=work.identity))
            self.events.append("PREDICTION")
            return succeeded("PREDICTION_RETAINED")

        self.dispatcher = L.TailOutputDispatchV1(
            ack_sender=lambda packet: (
                self.events.append("ACK"), self.packets.append(bytes(packet))),
            queue_depth=4,
            map_callback=self.registry.callback,
            prediction_callback=prediction)
        self.dispatcher.start()
        self.seam = S.GTFreeEdgePostComputeSeamV2(
            dispatcher=self.dispatcher,
            register_map_document=self.registry.register)


class ExactSignatureTest(unittest.TestCase):
    def test_processed_frame_uses_evaluation_records_and_mask(self) -> None:
        harness = ExactDispatchHarness()
        try:
            adapter = S.BEdgePostComputeAdapterV2(
                processor=FakeProcessor(), seam=harness.seam,
                run_id="run", cell_id="cell", raw_clock=lambda: 999)
            item = adapter.verify(b"frame", received_wall_s=1.5)
            # Other repository tests may preload the old scorer. This test is
            # about signatures; the subprocess test below proves import purity.
            with mock.patch.object(S, "FORBIDDEN_RUNTIME_PREFIXES", ()):
                receipt = adapter.process(item)
            harness.dispatcher.stop()
            self.assertIsInstance(receipt, L.DispatchReceiptV1)
            self.assertEqual(harness.events[0], "ACK")
            self.assertCountEqual(harness.events[1:], ["MAP", "PREDICTION"])
            self.assertEqual(harness.decoded[0].objects,
                             ({"class_name": "person"},))
            np.testing.assert_array_equal(
                harness.decoded[0].semantic_mask,
                np.asarray([[0, 1]], dtype=np.uint8))
        finally:
            harness.dispatcher.stop()

    def test_missing_usable_output_is_refused(self) -> None:
        harness = ExactDispatchHarness()
        try:
            adapter = S.BEdgePostComputeAdapterV2(
                processor=FakeProcessor(omit_output=True), seam=harness.seam,
                run_id="run", cell_id="cell", raw_clock=lambda: 1)
            item = adapter.verify(b"frame", received_wall_s=1.0)
            with self.assertRaisesRegex(S.BEdgeServiceError, "records/mask"):
                adapter.process(item)
        finally:
            harness.dispatcher.stop()

    def test_admission_to_compute_identity_drift_is_refused(self) -> None:
        processor = FakeProcessor()
        harness = ExactDispatchHarness()
        try:
            adapter = S.BEdgePostComputeAdapterV2(
                processor=processor, seam=harness.seam,
                run_id="run", cell_id="cell", raw_clock=lambda: 1)
            item = adapter.verify(b"frame", received_wall_s=1.0)
            processor.envelope.frame_id = 13
            processor.context.frame_id = 13
            with self.assertRaisesRegex(S.BEdgeServiceError,
                                        "identity changed"):
                adapter.process(item)
        finally:
            harness.dispatcher.stop()


class LifetimeIndexTest(unittest.TestCase):
    def test_duplicate_identity_and_payload_conflicts_are_distinct(self) -> None:
        identity = E._fake_identity()
        index = S.LifetimeIngressIndexV2()
        self.assertTrue(index.admit(identity, sha("a")))
        self.assertFalse(index.admit(identity, sha("a")))
        with self.assertRaises(S.IngressPayloadConflict):
            index.admit(identity, sha("b"))
        with self.assertRaises(S.IngressIdentityConflict):
            index.admit(dataclasses.replace(identity, q_e4=3001), sha("a"))


class ImportPurityTest(unittest.TestCase):
    def test_real_symbol_load_and_dispatch_do_not_import_quality_modules(self):
        root = Path(__file__).resolve().parents[2]
        script = r'''import numpy as np, sys
from rl_agent.splitfusion_run4b5b_live_isolation_v1 import b_edge_process_v1 as E
from rl_agent.splitfusion_run4b5b_live_isolation_v1 import b_edge_service_v2 as S
from rl_agent.splitfusion_run4b5b_live_isolation_v1 import branch_evidence_v1 as B
from rl_agent.splitfusion_run4b5b_live_isolation_v1 import live_adapters_v1 as L
S.load_qualified_symbols()
events=[]
registry=E.MapDocumentRegistryV1(lambda document: {'frame_id':document['frame_id']})
def done(work):
    return L.BranchCallbackResultV1(status=B.BranchStatus.SUCCEEDED,
        detail_code='DONE', evidence_sha256='0'*64)
dispatcher=L.TailOutputDispatchV1(ack_sender=lambda packet: events.append('ACK'),
    queue_depth=2, map_callback=registry.callback, prediction_callback=done)
dispatcher.start()
seam=S.GTFreeEdgePostComputeSeamV2(dispatcher=dispatcher,
    register_map_document=registry.register)
identity=E._fake_identity()
seam.emit(E.UsableTailOutputV1(identity=identity, reward_requested=True,
    object_records=({'class_name':'person'},),
    semantic_mask=np.asarray([[0,1]],dtype=np.uint8),
    map_document={'frame_id':identity.frame_id},
    tail_ready_monotonic_raw_ns=1))
dispatcher.stop()
assert S.PROVEN_EDGE_MODULE in sys.modules
assert events[0]=='ACK'
'''
        env = dict(os.environ)
        env["PYTHONPATH"] = str(root)
        completed = subprocess.run(
            [sys.executable, "-c", script], cwd=root, env=env,
            capture_output=True, text=True, timeout=60, check=False)
        self.assertEqual(completed.returncode, 0,
                         completed.stdout + completed.stderr)


if __name__ == "__main__":
    unittest.main()
