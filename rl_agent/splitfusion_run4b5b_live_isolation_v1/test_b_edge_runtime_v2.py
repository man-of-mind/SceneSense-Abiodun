"""CPU-only tests for the executable GT-free B edge runtime."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

from rl_agent.splitfusion_run4b5b_live_isolation_v1 import b_edge_process_v1 as E
from rl_agent.splitfusion_run4b5b_live_isolation_v1 import b_edge_runtime_v2 as R
from rl_agent.splitfusion_run4b5b_live_isolation_v1 import live_adapters_v1 as L


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _request(attempt: Path, *, ack_host: str = E.LOCAL_HOST,
             ack_port: int = 51014) -> str:
    raw = {
        "schema": E.REQUEST_SCHEMA, "role": E.ROLE,
        "run_id": "runtime_test", "variant": L.ActorVariant.RUN4B.value,
        "config_binding_sha256": _sha("config"),
        "actor_boundary_sha256": _sha("actor"),
        "feature_schema_sha256": _sha("features"),
        "transmitted_budget": 300, "deadline_ns": E.DEADLINE_NS,
        "ack_semantics": E.ACK_SEMANTICS,
        "postrun_semantics": E.POSTRUN_SEMANTICS,
        "clock_domain": E.CLOCK_DOMAIN,
        "split_host": {
            "carla_host": E.LOCAL_HOST, "ue_host": E.LOCAL_HOST,
            "cn_host": E.REMOTE_HOST, "edge_host": E.REMOTE_HOST,
            "ext_dn_host": E.REMOTE_HOST,
            "ack_receiver_host": ack_host,
            "ack_receiver_port": ack_port,
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


class _FakeSocket:
    def __init__(self, events: list[object]) -> None:
        self.events = events
        self.closed = False

    def setsockopt(self, *values) -> None:
        self.events.append(("setsockopt", values))

    def bind(self, endpoint) -> None:
        self.events.append(("bind", endpoint))

    def settimeout(self, value) -> None:
        self.events.append(("timeout", value))

    def sendto(self, payload, endpoint):
        self.events.append(("sendto", bytes(payload), endpoint))
        return len(payload)

    def recvfrom(self, _size):
        raise OSError("offline fake receiver")

    def close(self) -> None:
        self.closed = True
        self.events.append("socket_close")


class _FakePublisher:
    def __init__(self, *, map_host, map_port, chunk_bytes,
                 socket_buffer_request_bytes):
        self.arguments = (map_host, map_port, chunk_bytes,
                          socket_buffer_request_bytes)
        self.published = 0
        self.closed = False

    def publish(self, document):
        self.published += 1
        return {"frame_id": document["frame_id"]}

    def close(self):
        self.closed = True


class _FakeProcessor:
    def __init__(self, **kwargs):
        self.arguments = kwargs

    def verify(self, _payload):
        raise AssertionError("offline construction must not process payload")

    def process(self, _payload, *, edge_timing):
        raise AssertionError("offline construction must not process payload")


class _FakeTensor:
    def detach(self):
        return self

    def to(self, **_kwargs):
        return self

    def contiguous(self):
        return self

    def numpy(self):
        return [[0]]


class _FakeTorch:
    uint8 = "uint8"
    cuda = types.SimpleNamespace(is_available=lambda: True)

    @staticmethod
    def device(name):
        return name


class _FakePending:
    def __init__(self):
        self.closed = False

    def offer(self, _stream, _item, *, sequence):
        del sequence
        return True, None

    def take(self, timeout):
        del timeout
        return None

    def close(self):
        self.closed = True
        return []


class RuntimeConstructionTest(unittest.TestCase):
    def _case(self, root: Path):
        attempt = root / "attempt"
        campaign = root / "campaign.json"
        campaign.write_text(json.dumps({"runtime": {
            "udp_chunk_bytes": 1200,
            "socket_buffer_request_bytes": 4096,
        }}), encoding="utf-8")
        args = R.RuntimeArgumentsV2(
            request_b64=_request(attempt), campaign_config=campaign,
            ready_file=root / "READY.json", cell_id="cell",
            edge_port=51002, direct_map_host="192.0.2.8",
            direct_map_port=39320, queue_depth=4)
        events: list[object] = []
        sockets: list[_FakeSocket] = []

        def socket_factory(*_args):
            item = _FakeSocket(events)
            sockets.append(item)
            return item

        runtime = types.SimpleNamespace(
            _codec=object(), publish_cpu=lambda computed: computed)
        edge = types.SimpleNamespace(
            runtime=runtime, autoencoders={}, tail=object())
        authorities = R.RuntimeAuthoritiesV2(
            torch=_FakeTorch,
            load_campaign=lambda path: json.loads(path.read_text()),
            service_deadline_s=lambda _campaign: 0.170,
            ack_timeout_s=lambda _campaign: 0.170,
            load_contract=lambda: object(),
            preload_edge=lambda _device: edge,
            snapshot=lambda _serialized: types.SimpleNamespace(
                semantic_labels=_FakeTensor(), records=()),
            processor_type=_FakeProcessor,
            map_publisher_type=_FakePublisher,
            compute=lambda *_args, **_kwargs: None,
            pending_type=_FakePending,
            warm_edge=lambda *_args, **_kwargs: {
                "schema": "fake-prewarm", "completed": True},
            socket_factory=socket_factory,
            raw_clock=lambda: 7)
        return args, authorities, events, sockets

    def test_build_owns_udp_endpoints_prewarm_and_clean_close(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args, authorities, events, sockets = self._case(Path(temporary))
            built = R.build_runtime(args, authorities)
            self.assertEqual(len(sockets), 2)
            self.assertIn(("bind", ("0.0.0.0", 51002)), events)
            self.assertFalse(built.ready_document["tcp_listener"])
            self.assertFalse(built.ready_document["gt_ingress"])
            self.assertFalse(built.ready_document["qperc_reward_evaluator"])
            self.assertFalse(built.ready_document["map_wait"])
            self.assertEqual(built.ready_document["ack_receiver_port"], 51014)
            built.start()
            result = built.close()
            self.assertEqual(result["failures"], ())
            self.assertTrue(all(item.closed for item in sockets))
            self.assertTrue(built.publisher.closed)

    def test_build_refuses_ack_map_endpoint_alias(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, authorities, _events, _sockets = self._case(root)
            request = _request(root / "other", ack_host=E.LOCAL_HOST,
                               ack_port=39320)
            args = R.RuntimeArgumentsV2(
                request_b64=request, campaign_config=args.campaign_config,
                ready_file=args.ready_file, cell_id=args.cell_id,
                edge_port=args.edge_port, direct_map_host=E.LOCAL_HOST,
                direct_map_port=39320, queue_depth=4)
            with self.assertRaisesRegex(R.BEdgeRuntimeError, "distinct"):
                R.build_runtime(args, authorities)

    def test_incomplete_prewarm_fails_and_closes_owned_components(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args, authorities, _events, sockets = self._case(Path(temporary))
            authorities = R.RuntimeAuthoritiesV2(
                **{**{field: getattr(authorities, field)
                      for field in authorities.__dataclass_fields__},
                   "warm_edge": lambda *_args, **_kwargs: {
                       "completed": False}})
            with self.assertRaisesRegex(R.BEdgeRuntimeError, "prewarm"):
                R.build_runtime(args, authorities)
            self.assertTrue(all(item.closed for item in sockets))


class RuntimePreflightTest(unittest.TestCase):
    def test_preflight_is_cpu_only_and_requires_create_only_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = root / "campaign.json"
            campaign.write_text("{}", encoding="ascii")
            args = R.RuntimeArgumentsV2(
                request_b64=_request(root / "attempt"),
                campaign_config=campaign, ready_file=root / "READY.json",
                cell_id="cell", edge_port=51002,
                direct_map_host="192.0.2.8", direct_map_port=39320,
                queue_depth=4)
            result = R.runtime_preflight(
                args, find_module=lambda _name: object())
            self.assertEqual(result["schema"], R.RUNTIME_PREFLIGHT_SCHEMA)
            self.assertFalse(result["tcp_listener"])
            self.assertFalse(result["gt_ingress"])
            (root / "attempt").mkdir()
            with self.assertRaisesRegex(R.BEdgeRuntimeError, "create-only"):
                R.runtime_preflight(args, find_module=lambda _name: object())

    def test_start_token_is_exact_and_preflight_does_not_call_factory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = root / "campaign.json"
            campaign.write_text("{}", encoding="ascii")
            argv = ["preflight", "--request-b64", _request(root / "attempt"),
                    "--campaign-config", str(campaign), "--ready-file",
                    str(root / "READY.json"), "--cell-id", "cell",
                    "--edge-port", "51002", "--direct-map-host", "192.0.2.8",
                    "--direct-map-port", "39320"]
            # Resolve authorities syntactically only; no torch/CUDA import.
            with mock.patch.object(
                    E, "_find_module", return_value=object()):
                self.assertEqual(R.main(argv), 0)


class ForbiddenLivePathTest(unittest.TestCase):
    def test_runtime_source_has_no_evaluator_gt_qperc_or_quality_spec_calls(self):
        source = Path(R.__file__).read_text(encoding="utf-8")
        forbidden = (
            "Run4EvaluatorV2(", "load_run4_quality_spec(",
            "read_ground_truth(", "GtIngress", "Q_perc",
            "send_quality_feedback(", "evaluator.submit(",
        )
        for token in forbidden:
            self.assertNotIn(token, source)
        self.assertIn("dispatcher.stop()", source)
        self.assertIn("publisher.close()", source)
        self.assertIn("control.close()", source)


if __name__ == "__main__":
    unittest.main()
