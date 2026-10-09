from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import continuous_execution_v2 as X

from . import b_opportunity_processor_v1 as P
from . import b_route_bridge_v4 as B
from . import b_ue_process_v1 as U
from . import final_actor_gate_v2 as G
from . import live_adapters_v1 as L
from . import operational_ack_v1 as A


def sha(text): return hashlib.sha256(text.encode()).hexdigest()


class _Actor:
    def deterministic_execution(self, state):
        assert tuple(state.shape) == (1, 20)
        return SimpleNamespace(mode_index=torch.tensor([6]), q_e4=torch.tensor([4321]))


class _Contract:
    def resolve_q_e4(self, mode, q):
        return SimpleNamespace(mode_id=mode, q_e4=q, keep_count=5479,
            action_id=None, profile_id=None, execution_bundle_sha256=sha("bundle"))


class _Continuous:
    def prepare(self, profile, _input, frame):
        envelope = X.ExecutionEnvelopeV3(
            mode_id=profile.mode_id, q_e4=profile.q_e4,
            keep_count=profile.keep_count, anchor_action_id=None,
            reward_requested=True, session_uuid=frame.session_uuid,
            controller_lineage_sha256=frame.controller_lineage_sha256,
            decision_seq=frame.decision_seq, ticket_seq=frame.ticket_seq,
            frame_id=frame.frame_id, tensor_seq=frame.tensor_seq,
            capture_timestamp_ns=frame.capture_timestamp_ns,
            execution_bundle_sha256=profile.execution_bundle_sha256,
            inner_payload_sha256=hashlib.sha256(b"inner").hexdigest(),
            inner_payload=b"inner")
        return SimpleNamespace(envelope=envelope)


class _Sender:
    def __init__(self): self.packets = []
    def sendto(self, packet, remote): self.packets.append((packet, remote))


class _Processor(P.BOpportunityProcessorV1):
    def _features(self, opportunity, previous):
        self.seen_previous = previous
        return (0.0,) * 20


class OpportunityProcessorTest(unittest.TestCase):
    def test_exact_actor_action_front_sfd4_send_and_identity(self):
        identity = G.FinalActorIdentityV2(
            variant=G.RUN4B_VARIANT, actor_state_dict_sha256=sha("weights"),
            actor_tree_sha256=sha("tree"), feature_schema_id="r4b",
            feature_schema_sha256=sha("schema"),
            feature_order_sha256=sha("order"),
            feature_order=tuple(f"f{i}" for i in range(20)),
            model_binding_sha256=sha("model"),
            operational_latency_provider_sha256=sha("latency"),
            scientific_channel_sha256=sha("channel"),
            run5b_only_authority_sha256=None, selected_seed=43,
            selected_update=10000)
        request = SimpleNamespace(
            variant=L.ActorVariant.RUN4B, feature_schema_sha256=sha("schema"),
            run_id="run", actor_boundary_sha256=sha("lineage"))
        actor = P.LoadedActorBindingV1(
            G.LoadedFinalActorV2(identity, _Actor()), Path("manifest"), Path("weights"))
        sender = _Sender()
        processor = object.__new__(_Processor)
        processor.request, processor.actor = request, actor
        processor.telemetry = SimpleNamespace(
            session_uuid="00000000-0000-4000-8000-000000000001")
        processor.contract, processor.continuous = _Contract(), _Continuous()
        processor.sender, processor.remote = sender, ("127.0.0.1", 5000)
        processor.input_builder = lambda frame, radar: (frame, radar)
        processor.cell_id, processor.chunk_bytes = "cell", 1200
        opportunity = B.RouteOpportunityV4(
            sequence=9, frame_id=12, capture_timestamp_ns=1234,
            action_open_monotonic_raw_ns=5678,
            submit_kwargs={"frame_bgr": "frame", "radar_tensor": "radar",
                           "ego_pose": (1, 2, 3, 4, 5, 6),
                           "stream_id": "ego"})
        marker = object()
        sent = processor(opportunity, marker)
        self.assertIs(processor.seen_previous, marker)
        self.assertEqual((sent.identity.mode_id, sent.identity.q_e4), (6, 4321))
        self.assertEqual(sent.action_open_monotonic_raw_ns, 5678)
        self.assertGreater(len(sender.packets), 0)
        self.assertEqual({remote for _packet, remote in sender.packets},
                         {("127.0.0.1", 5000)})

    def test_prior_uses_operational_outcome_not_quality(self):
        ident = A.FrameActionIdentityV1(
            run_id="r", cell_id="c", stream_id="s",
            session_uuid="00000000-0000-4000-8000-000000000001",
            controller_lineage_sha256=sha("l"), decision_seq=1, ticket_seq=1,
            frame_id=1, tensor_seq=1, capture_timestamp_ns=1, mode_id=3,
            q_e4=4000, keep_count=5800, anchor_action_id=None, profile_id=None,
            execution_bundle_sha256=sha("b"))
        outcome = A.OperationalOutcomeV1(
            identity=ident, terminal=A.OperationalTerminal.SUCCESS,
            action_open_monotonic_raw_ns=10,
            resolution_monotonic_raw_ns=30, observed_latency_ns=20,
            state_latency_ns=20, accepted_ack_sha256=sha("a"),
            tail_output_sha256=sha("t"))
        prior = P._prior(outcome)
        self.assertEqual((prior.mode_id, prior.q_e4,
                          prior.operational_latency_ns), (3, 4000, 20))

    def test_module_has_no_live_gt_or_quality_dependency(self):
        source = Path(P.__file__).read_text(encoding="utf-8").lower()
        for forbidden in ("_ground_truth(", "semantic_gt_3class(",
                          "evaluate_exact_quality("):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__": unittest.main()
