from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from . import b_one_frame_pipeline_v1 as O
from . import b_route_bridge_v3 as V3
from . import b_route_bridge_v4 as B
from . import b_ue_process_v1 as U
from . import live_adapters_v1 as L
from . import operational_ack_v1 as A


def sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class OneFramePipelineV2Test(unittest.TestCase):
    def test_exactly_one_without_weakening_production_bridge(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            def processor(opportunity, _previous):
                identity = A.FrameActionIdentityV1(
                    run_id="r",
                    cell_id="c",
                    stream_id="s",
                    session_uuid="00000000-0000-4000-8000-000000000001",
                    controller_lineage_sha256=sha("l"),
                    decision_seq=0,
                    ticket_seq=0,
                    frame_id=9,
                    tensor_seq=0,
                    capture_timestamp_ns=10,
                    mode_id=1,
                    q_e4=2,
                    keep_count=9798,
                    anchor_action_id=None,
                    profile_id=None,
                    execution_bundle_sha256=sha("b"),
                )
                return U.BTransmissionV1(
                    identity,
                    opportunity.action_open_monotonic_raw_ns,
                    8,
                    True,
                )

            def route(bridge) -> None:
                bridge.offer_prepared(
                    B.RouteOpportunityV4(0, 9, 10, 11, {"frame_id": 9})
                )
                bridge.offer_prepared(
                    B.RouteOpportunityV4(1, 10, 11, 12, {"frame_id": 10})
                )

            bridge = O.EngineeringOneFrameBridgeV1(
                variant=L.ActorVariant.RUN4B,
                feature_schema_sha256=sha("s"),
                actor_boundary_sha256=sha("a"),
                processor=processor,
                route_driver=route,
                raw_spool_root=root / "raw",
            )
            sent = bridge.transmit_next(0, None)
            self.assertEqual(sent.identity.frame_id, 9)
            bridge.close()
            self.assertEqual(bridge.transmitted, 1)
            self.assertEqual(bridge.transmitted_budget, 1)

            with self.assertRaisesRegex(V3.BRouteBridgeError, "budget"):
                B.BRouteBridgeV4(
                    variant=L.ActorVariant.RUN4B,
                    feature_schema_sha256=sha("s"),
                    actor_boundary_sha256=sha("a"),
                    processor=processor,
                    route_driver=route,
                    raw_spool_root=root / "other",
                    transmitted_budget=1,
                )


if __name__ == "__main__":
    unittest.main()
