from __future__ import annotations

import copy
import importlib.util
import json
import queue
import struct
import sys
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

from rl_agent.ue_split_wire_contract import ROOT, resolve_registered_profile


RUNTIME_PATH = (
    ROOT
    / "uplink_only_spatial_map_pipeline"
    / "carla_fusion_staleness_scenario_uplink_only_v2.py"
)


def _load_runtime():
    spec = importlib.util.spec_from_file_location("ue_a2_runtime_test_module", RUNTIME_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot import {RUNTIME_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _wire_field_lengths(shape: tuple[int, ...], bits: int) -> tuple[int, int]:
    _batch, channels, height, width = shape
    ranges = channels * 2 * 4
    total = channels * height * width
    if bits == 8:
        data = total
    elif bits == 6:
        data = ((total + 3) // 4) * 3
    elif bits == 4:
        data = (total + 1) // 2
    else:  # pragma: no cover - the registry validates this independently
        raise AssertionError(bits)
    return ranges, data


def _valid_payload(profile) -> dict[str, object]:
    bits = int(profile.row["quantization_bits"])
    features: dict[str, dict[str, bytes]] = {}
    for level, shape in profile.expected_wire_shapes.items():
        _batch, channels, height, width = shape
        ranges, data = _wire_field_lengths(shape, bits)
        features[level] = {
            "header": struct.pack("!IIIB", channels, height, width, bits),
            "ranges": bytes(ranges),
            "data": bytes(data),
        }
    return {
        "frame_id": 17,
        "batch_size": 1,
        "model_input_size": [
            int(profile.row["input_width"]),
            int(profile.row["input_height"]),
        ],
        "feature_shapes": {
            name: tuple(shape) for name, shape in profile.expected_wire_shapes.items()
        },
        "features": features,
        "profile_identity": dict(profile.wire_identity),
    }


class _OnePayloadReceiver:
    def __init__(self, payload: dict[str, object], stop_event: threading.Event) -> None:
        self.payload = payload
        self.stop_event = stop_event
        self._pending: dict[object, object] = {}

    def receive(self):
        self.stop_event.set()
        return self.payload


class _ResultStore:
    def __init__(self) -> None:
        self.items: list[tuple[int, dict[str, object]]] = []

    def put(self, frame_id: int, payload: dict[str, object]) -> None:
        self.items.append((frame_id, payload))


class UEA2RuntimeIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._sys_path_before = list(sys.path)
        cls._modules_before = set(sys.modules)
        cls.runtime = _load_runtime()
        cls.profile = resolve_registered_profile(
            "ae32__u4__q0.9__zstd3__ckpt10cebbeede4d"
        )

    @classmethod
    def tearDownClass(cls) -> None:
        # The production runtime deliberately bootstraps its standalone import
        # paths.  Keep that process-local behavior from leaking into unrelated
        # unit-test modules in the same discovery interpreter.
        sys.path[:] = cls._sys_path_before
        for name in set(sys.modules) - cls._modules_before:
            module = sys.modules.get(name)
            module_file = str(getattr(module, "__file__", "") or "")
            if (
                name == "ue_a2_runtime_test_module"
                or "pole_lraspp_multimodal_fusion" in module_file
            ):
                sys.modules.pop(name, None)

    def _worker(self, payload: dict[str, object]):
        stop_event = threading.Event()
        receiver = _OnePayloadReceiver(payload, stop_event)
        worker = self.runtime.FusionRemoteInferenceWorker(
            model=None,
            receiver=receiver,
            sender=None,
            device=None,
            stop_event=stop_event,
            transport=None,
            score_threshold=0.2,
            nms_radius_px=2,
            topk=120,
            max_objects_drawn=120,
            receive_queue_size=1,
            registered_profile=self.profile,
        )
        return worker

    def test_runtime_resolves_current_abiodun_codec_with_uint6(self) -> None:
        codec_path = Path(self.runtime.od_collect.__file__).resolve()
        self.assertTrue(codec_path.is_relative_to(ROOT), str(codec_path))
        self.assertIn(
            "per_channel_uint6",
            self.runtime.od_collect.QUANT_MODE_CHOICES,
        )

    def test_edge_rejects_identity_before_queue_insertion(self) -> None:
        invalid = _valid_payload(self.profile)
        invalid_identity = dict(invalid["profile_identity"])
        invalid_identity["quantization_mode"] = "per_channel_uint8"
        invalid["profile_identity"] = invalid_identity
        worker = self._worker(invalid)

        worker._receive_loop()

        self.assertIsNotNone(worker._payload_queue)
        self.assertTrue(worker._payload_queue.empty())
        self.assertEqual(worker._profile_identity_accepted, 0)
        self.assertEqual(worker._profile_identity_rejected, 1)

    def test_exact_registered_payload_is_queued_once(self) -> None:
        worker = self._worker(_valid_payload(self.profile))

        worker._receive_loop()

        self.assertEqual(worker._payload_queue.qsize(), 1)
        queued, _recv_perf, _recv_wall = worker._payload_queue.get_nowait()
        self.assertEqual(
            queued["profile_identity"],
            self.profile.wire_identity,
        )
        self.assertEqual(worker._profile_identity_accepted, 1)
        self.assertEqual(worker._profile_identity_rejected, 0)

    def test_map_packet_preserves_exact_profile_identity(self) -> None:
        publisher = self.runtime.SpatialMapResultPublisher.__new__(
            self.runtime.SpatialMapResultPublisher
        )
        publisher.stream_id = "ue-a2-test"
        publisher.traffic_light_id = ""
        publisher.traffic_light_actor_id = -1
        publisher.traffic_light_opendrive_id = ""
        publisher.camera_width = 1280
        publisher.camera_height = 720
        publisher.camera_fov = 100.0
        captured: list[dict[str, object]] = []
        publisher._enqueue = lambda packet, _frame_id: captured.append(packet)
        source = {
            "frame_id": 17,
            "stream_id": "ue-a2-test",
            "carla_timestamp": 1.7,
            "camera_transform": {},
            "camera_matrix": np.eye(4),
            "profile_identity": dict(self.profile.wire_identity),
        }
        result = {
            "frame_id": 17,
            "objects": [
                {
                    "class_name": "person",
                    "score": 0.9,
                    "world_x": 1.0,
                    "world_y": 2.0,
                    "world_z": 0.0,
                    "size_x": 0.5,
                    "size_y": 0.5,
                    "size_z": 1.7,
                }
            ],
            "mask": None,
            "server_ms": 1.0,
        }

        publisher.publish_from_payload(source_payload=source, result=result, timing={})

        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["profile_identity"], self.profile.wire_identity)
        self.assertEqual(captured[0]["objects"][0]["type"], "Pedestrian")

    def test_identity_mutation_cannot_be_relabelled_by_result_path(self) -> None:
        payload = _valid_payload(self.profile)
        invalid = copy.deepcopy(payload)
        invalid["profile_identity"]["action_contract_sha256"] = "0" * 64
        worker = self._worker(invalid)

        self.assertFalse(worker._accept_profile_payload(invalid))
        self.assertEqual(worker._processed, 0)

    def test_result_receiver_rejects_cross_profile_ack_before_store(self) -> None:
        invalid_result = {
            "frame_id": 17,
            "ack": True,
            "profile_identity": dict(self.profile.wire_identity),
        }
        invalid_result["profile_identity"]["profile_id"] = "another-profile"
        stop_event = threading.Event()
        store = _ResultStore()
        receiver = self.runtime.CameraResultReceiver(
            receiver=_OnePayloadReceiver(invalid_result, stop_event),
            result_store=store,
            stop_event=stop_event,
            registered_profile=self.profile,
        )

        receiver.run()

        self.assertEqual(store.items, [])
        self.assertEqual(receiver.profile_identity_accepted, 0)
        self.assertEqual(receiver.profile_identity_rejected, 1)

    def test_result_receiver_accepts_exact_registered_ack(self) -> None:
        result = {
            "frame_id": 17,
            "ack": True,
            "profile_identity": dict(self.profile.wire_identity),
        }
        stop_event = threading.Event()
        store = _ResultStore()
        receiver = self.runtime.CameraResultReceiver(
            receiver=_OnePayloadReceiver(result, stop_event),
            result_store=store,
            stop_event=stop_event,
            registered_profile=self.profile,
        )

        receiver.run()

        self.assertEqual(len(store.items), 1)
        self.assertEqual(store.items[0][0], 17)
        self.assertEqual(
            store.items[0][1]["profile_identity"],
            self.profile.wire_identity,
        )
        self.assertEqual(receiver.profile_identity_accepted, 1)
        self.assertEqual(receiver.profile_identity_rejected, 0)

    def test_front_result_path_revalidates_identity_before_map_stats(self) -> None:
        result = {"profile_identity": dict(self.profile.wire_identity)}
        self.assertEqual(
            self.runtime._validated_result_profile_identity(result, self.profile),
            self.profile.wire_identity,
        )
        result["profile_identity"]["entropy_level"] = 2
        with self.assertRaises(self.runtime.SplitWireContractError):
            self.runtime._validated_result_profile_identity(result, self.profile)

    def test_edge_summary_exists_at_startup_and_refreshes_on_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            logger = self.runtime.EdgeUplinkMetricsLogger(
                Path(temporary) / "edge_metrics.csv"
            )
            worker = self._worker(_valid_payload(self.profile))
            worker.edge_metrics_logger = logger
            logger.write_summary(worker.summary())
            startup = json.loads(logger.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(startup["registered_profile_id"], self.profile.profile_id)
            self.assertEqual(
                startup["registered_profile_identity"],
                self.profile.wire_identity,
            )
            self.assertEqual(startup["profile_identity_rejected"], 0)

            invalid = _valid_payload(self.profile)
            invalid["profile_identity"] = {
                **invalid["profile_identity"],
                "checkpoint_sha256": "0" * 64,
            }
            self.assertFalse(worker._accept_profile_payload(invalid))
            updated = json.loads(logger.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(updated["profile_identity_rejected"], 1)
            self.assertEqual(
                updated["last_profile_rejection"]["code"],
                "WIRE_IDENTITY_VALUE_MISMATCH",
            )
            self.assertFalse(logger.summary_path.with_name(
                logger.summary_path.name + ".tmp"
            ).exists())
            logger.close()

    def test_profile_state_lock_serializes_acceptance_and_copies_summary(self) -> None:
        worker = self._worker(_valid_payload(self.profile))
        entered = threading.Event()
        completed = threading.Event()

        def accept_payload() -> None:
            entered.set()
            worker._accept_profile_payload(_valid_payload(self.profile))
            completed.set()

        worker._profile_state_lock.acquire()
        thread = threading.Thread(target=accept_payload)
        try:
            thread.start()
            self.assertTrue(entered.wait(1.0))
            self.assertFalse(completed.wait(0.05))
        finally:
            worker._profile_state_lock.release()
        thread.join(timeout=1.0)
        self.assertFalse(thread.is_alive())
        self.assertTrue(completed.is_set())
        self.assertEqual(worker.summary()["profile_identity_accepted"], 1)

        with worker._profile_state_lock:
            worker._last_profile_rejection = {"code": "ORIGINAL"}
        snapshot = worker.summary()
        with worker._profile_state_lock:
            worker._last_profile_rejection["code"] = "MUTATED"
        self.assertEqual(snapshot["last_profile_rejection"]["code"], "ORIGINAL")


if __name__ == "__main__":
    unittest.main()
