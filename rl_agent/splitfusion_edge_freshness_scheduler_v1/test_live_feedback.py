from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from .live_capture import (
    build_scheduler_collector_class,
    build_scheduler_runtime_class,
)
from .scheduler import SCHEMA


class _Counters:
    def __init__(self) -> None:
        self.values: dict[str, int] = {}

    def bump(self, name: str, amount: int = 1) -> None:
        self.values[name] = self.values.get(name, 0) + amount


class LiveFeedbackTest(unittest.TestCase):
    def _runtime(self):
        runtime_type = build_scheduler_runtime_class(object)
        runtime = runtime_type.__new__(runtime_type)
        runtime.metrics = {
            7: {
                "action_id": 71,
                "profile_id": "split_ae32_uint4_q9800",
                "stream_id": "ue-1",
                "capture_id": "ue-1:7",
            }
        }
        runtime.lock = threading.Lock()
        runtime.completed = 0
        runtime.counters = _Counters()
        runtime.scheduler_feedback_records = []
        runtime.scheduler_feedback_lock = threading.Lock()
        return runtime

    @staticmethod
    def _feedback(reason: str = "SUPERSEDED_PENDING") -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "run_id": "run",
            "cell_id": "cell",
            "stream_id": "ue-1",
            "frame_id": 7,
            "sequence_id": 8,
            "action_id": 71,
            "profile_id": "split_ae32_uint4_q9800",
            "capture_timestamp_ns": 1_000_000_000,
            "terminal_reason": reason,
            "outcome_class": "INTENTIONAL_FRESHNESS_DROP",
            "stage": "PENDING",
            "agent_credit": {"charge_feature_bytes": 6464},
        }

    def test_scheduler_terminal_is_identity_bound_and_not_map_installed(self) -> None:
        runtime = self._runtime()
        forwarded: list[tuple[str, str]] = []
        runtime.scheduler_feedback_callback = lambda value, capture: forwarded.append(
            (str(value["terminal_reason"]), str(capture))
        )
        runtime._accept_scheduler_terminal(
            self._feedback(),
            message_id=7,
            received_ns=20,
            received_wall=2.0,
        )
        self.assertEqual(forwarded, [("SUPERSEDED_PENDING", "ue-1:7")])
        self.assertEqual(runtime.completed, 1)
        self.assertEqual(
            runtime.metrics[7]["map_publication_status"],
            "EDGE_TERMINATED_WITHOUT_MAP_PUBLICATION",
        )
        self.assertEqual(len(runtime.scheduler_feedback_records), 1)

    def test_edge_publication_cannot_impersonate_scheduler_noninstall(self) -> None:
        runtime = self._runtime()
        runtime.scheduler_feedback_callback = lambda _value, _capture: None
        with self.assertRaisesRegex(RuntimeError, "normal map-install path"):
            runtime._accept_scheduler_terminal(
                self._feedback("RESULT_PUBLISHED"),
                message_id=7,
                received_ns=20,
                received_wall=2.0,
            )

    def test_collector_summary_separates_scheduler_drop_and_map_install(self) -> None:
        class Base:
            def __init__(base_self, **_keywords):
                base_self.campaign = {"campaign_id": "campaign"}
                base_self.cell = {"cell_id": "cell"}
                base_self.feedback_port = 65431
                base_self.attempt_dir = Path(temporary.name)
                base_self.rows_lock = threading.Lock()
                base_self.rows = [
                    {
                        "frame_id": 7,
                        "capture_wall_s": 1.0,
                        "prepare_status": "SENT",
                    },
                    {
                        "frame_id": 8,
                        "capture_wall_s": 1.1,
                        "prepare_status": "SENT",
                    },
                ]
                base_self.gt_lock = threading.Lock()
                base_self.installed_at = {8: 1.35}
                base_self.transport_counters = _Counters()
                base_self.live = SimpleNamespace(
                    scheduler_feedback_callback=None,
                    scheduler_feedback_records=[self._feedback()],
                    scheduler_feedback_lock=threading.Lock(),
                )

            def diagnostic_summary(base_self):
                return {"base": True}

            def finish(base_self):
                return True

        with tempfile.TemporaryDirectory() as path:
            temporary = SimpleNamespace(name=path)
            collector_type = build_scheduler_collector_class(Base)
            with patch(
                "rl_agent.splitfusion_edge_freshness_scheduler_v1.live_capture.socket.socket",
                return_value=MagicMock(),
            ):
                collector = collector_type()
            summary = collector.diagnostic_summary()
            self.assertEqual(summary["map_utility"]["sent_frames"], 2)
            self.assertEqual(summary["map_utility"]["ack_installed_frames"], 1)
            self.assertEqual(
                summary["freshness_scheduler"][
                    "intentional_freshness_drop_frames"
                ],
                1,
            )
            self.assertTrue(collector.finish())


if __name__ == "__main__":
    unittest.main()
