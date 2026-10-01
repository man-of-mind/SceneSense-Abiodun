"""No-launch CLI coverage for the additive B edge service."""

from __future__ import annotations

import tempfile
from pathlib import Path
import types
import unittest

from rl_agent.splitfusion_run4b5b_live_isolation_v1 import (
    b_edge_service_v2 as S,
)
from rl_agent.splitfusion_run4b5b_live_isolation_v1.test_b_edge_process_v1 import (
    request,
)


class ServiceCliTest(unittest.TestCase):
    def test_preflight_checks_request_and_exact_symbol_names(self) -> None:
        module = types.SimpleNamespace(
            Run4EdgeProcessorV2=type("Run4EdgeProcessorV2", (), {}),
            Run4MapPublisherV2=type("Run4MapPublisherV2", (), {}),
            run4_compute_on_detached_runtime=lambda: None)
        result = S.preflight(
            request()[0], importer=lambda _name: module,
            find_module=lambda _name: object())
        self.assertEqual(result["schema"], S.PREFLIGHT_SCHEMA)
        self.assertEqual(result["processor"], "Run4EdgeProcessorV2")
        self.assertFalse(result["gt_ingress"])
        self.assertFalse(result["qperc_or_reward"])
        self.assertFalse(result["legacy_quality_feedback"])

    def test_offline_fake_is_create_only_ack_first_and_gt_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "fake"
            result = S.offline_fake(request()[0], root=root)
            self.assertEqual(result["schema"], S.OFFLINE_FAKE_SCHEMA)
            self.assertEqual(result["events"][0], "ACK")
            self.assertCountEqual(result["events"][1:], ["MAP", "PREDICTION"])
            self.assertEqual(result["prediction_records"], 1)
            self.assertFalse(result["gt_qperc_reward_used"])
            with self.assertRaisesRegex(S.BEdgeServiceError, "create-only"):
                S.offline_fake(request()[0], root=root)


if __name__ == "__main__":
    unittest.main()
