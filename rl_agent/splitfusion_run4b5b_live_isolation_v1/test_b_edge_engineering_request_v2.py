"""One-frame engineering request separation and runtime acceptance tests."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from rl_agent.splitfusion_run4b5b_live_isolation_v1 import (
    b_edge_engineering_request_v2 as Q,
    b_edge_process_v1 as E,
    b_edge_runtime_v2 as R,
    live_adapters_v1 as L,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _encode(attempt: Path, *, budget: int = 1,
            ack_host: str = Q.ACK_RECEIVER_HOST) -> str:
    raw = {
        "schema": Q.SCHEMA, "purpose": Q.PURPOSE,
        "claim_scope": Q.CLAIM_SCOPE, "role": E.ROLE,
        "run_id": "one_frame_engineering",
        "variant": L.ActorVariant.RUN4B.value,
        "config_binding_sha256": _sha("config"),
        "actor_boundary_sha256": _sha("actor"),
        "feature_schema_sha256": _sha("features"),
        "transmitted_budget": budget, "deadline_ns": E.DEADLINE_NS,
        "ack_semantics": E.ACK_SEMANTICS,
        "postrun_semantics": E.POSTRUN_SEMANTICS,
        "clock_domain": E.CLOCK_DOMAIN,
        "split_host": {
            "carla_host": E.LOCAL_HOST, "ue_host": E.LOCAL_HOST,
            "cn_host": E.REMOTE_HOST, "edge_host": E.REMOTE_HOST,
            "ext_dn_host": E.REMOTE_HOST,
            "ack_receiver_host": ack_host,
            "ack_receiver_port": Q.ACK_RECEIVER_PORT,
        },
        "output_root": None, "evidence_root": None,
        "actor_manifest_path": None,
        "remote_attempt_root": str(attempt),
        "required_authority_modules": list(E.REQUIRED_AUTHORITIES),
        "old_live_quality_runtime_permitted": False,
    }
    payload = json.dumps(raw, sort_keys=True,
                         separators=(",", ":")).encode("ascii")
    return base64.urlsafe_b64encode(payload).decode("ascii")


class EngineeringRequestTest(unittest.TestCase):
    def test_one_frame_and_tunnel_ack_are_accepted_by_runtime_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = root / "campaign.json"
            campaign.write_text("{}", encoding="ascii")
            encoded = _encode(root / "attempt")
            args = R.RuntimeArgumentsV2(
                request_b64=encoded, campaign_config=campaign,
                ready_file=root / "READY.json", cell_id="cell",
                edge_port=51002, direct_map_host="10.21.16.222",
                direct_map_port=39320, queue_depth=4)
            result = R.runtime_preflight(
                args, find_module=lambda _name: object())
            self.assertEqual(result["purpose"], Q.PURPOSE)
            self.assertEqual(result["claim_scope"], Q.CLAIM_SCOPE)
            self.assertEqual(result["transmitted_budget"], 1)
            raw = R._validated_request(encoded)
            self.assertEqual(raw["split_host"]["ack_receiver_host"],
                             "10.0.0.2")
            with self.assertRaises(E.RequestError):
                E.preflight_request(encoded,
                                    find_module=lambda _name: object())

    def test_budget_or_ack_hostname_cannot_masquerade_as_engineering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(Q.EngineeringRequestError,
                                        "budget"):
                Q.decode_and_validate(_encode(root / "a", budget=300))
            with self.assertRaisesRegex(Q.EngineeringRequestError,
                                        "UE tunnel"):
                Q.decode_and_validate(_encode(root / "b",
                                              ack_host=E.LOCAL_HOST))

    def test_frozen_300_request_semantics_are_unchanged(self) -> None:
        from rl_agent.splitfusion_run4b5b_live_isolation_v1.test_b_edge_runtime_v2 import (
            _request)

        with tempfile.TemporaryDirectory() as temporary:
            raw = R._validated_request(_request(Path(temporary) / "attempt"))
            self.assertEqual(raw["transmitted_budget"], 300)
            self.assertEqual(raw["purpose"],
                             "FROZEN_300_FRAME_QUALIFICATION")
            self.assertEqual(raw["split_host"]["ack_receiver_host"],
                             E.LOCAL_HOST)


if __name__ == "__main__":
    unittest.main()
