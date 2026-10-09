"""Offline tests for the presentation-quality viewer and its live metrics.

No CARLA, CUDA, OAI, Docker, socket or port. The Flask endpoints are exercised
through the in-process WSGI test client, and the static town geometry comes from
the recorded fixture rather than from a CARLA query.

The JavaScript half runs under node against the exact asset files the server
inlines, so a pass covers the shipped code rather than a reimplementation.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

REPO = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parents[1]
ASSETS = PACKAGE / "viewer_assets"
NODE_SUITE = PACKAGE / "tests" / "js" / "presentation_tests_v1.js"
STATIC_GEOMETRY = PACKAGE / "recordings" / "two_ego_live.jsonl.static.json"

SESSION_ID = "viewer-presentation-test-v1"
STREAM_A = "two-ue/ue-a"
STREAM_B = "two-ue/ue-b"


def node_available() -> bool:
    return shutil.which("node") is not None


class NodePresentationSuiteTests(unittest.TestCase):
    """Run the deterministic JavaScript suite for the shipped viewer assets."""

    @unittest.skipUnless(node_available(), "node is unavailable")
    def test_presentation_javascript_suite_passes(self):
        completed = subprocess.run(
            ["node", str(NODE_SUITE)],
            capture_output=True, text=True, timeout=300, cwd=str(REPO),
        )
        self.assertEqual(
            completed.returncode, 0,
            f"node suite failed:\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        report = json.loads(completed.stdout)
        self.assertEqual(report["status"], "PASS")
        # A guard against the suite silently degrading to a handful of checks.
        self.assertGreater(report["checks_passed"], 90)
        index = report["road_index"]
        self.assertGreater(index["segments"], 5000)
        self.assertGreater(index["on_road_hits"], 1000)
        self.assertGreater(index["speedup"], 4.0)

    @unittest.skipUnless(node_available(), "node is unavailable")
    def test_every_asset_parses_under_node(self):
        for name in (
            "presentation_predictor_v1.js",
            "presentation_inventory_v1.js",
            "viewer_app_v2.js",
        ):
            with self.subTest(asset=name):
                completed = subprocess.run(
                    ["node", "--check", str(ASSETS / name)],
                    capture_output=True, text=True, timeout=60,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)


class ViewerAssetIntegrityTests(unittest.TestCase):
    """The served HTML must contain the tested assets verbatim."""

    def setUp(self) -> None:
        sys.path.insert(0, str(REPO))
        from spatial_map_coop import spatial_map_server_moving_ego as server
        self.server = server

    def test_viewer_html_inlines_the_exact_asset_bytes(self):
        html = self.server._viewer_html()
        self.assertNotIn("/*INLINE:", html)
        for name in (
            "presentation_predictor_v1.js",
            "presentation_inventory_v1.js",
            "viewer_app_v2.js",
        ):
            with self.subTest(asset=name):
                body = (ASSETS / name).read_text()
                self.assertIn(body, html)

    def test_viewer_shell_declares_the_recording_layout_and_controls(self):
        html = self.server._viewer_html()
        self.assertIn("aspect-ratio: 16 / 9", html)
        self.assertIn("tabular-nums", html)
        for control in ("rawBtn", "assocBtn", "metricsBtn", "presentBtn"):
            self.assertIn(control, html)
        self.assertIn("R / 1", html)
        self.assertIn("F / 2", html)

    def test_no_asset_labels_a_residual_as_network_latency(self):
        from spatial_map_coop.coop_input_v1 import live_metrics_v1
        blob = self.server._viewer_html().lower()
        for banned in live_metrics_v1.FORBIDDEN_LABELS:
            with self.subTest(term=banned):
                self.assertNotIn(banned, blob)
        for banned in ("5g", "radio latency"):
            with self.subTest(term=banned):
                self.assertNotIn(banned, blob)

    def test_missing_inline_marker_is_a_hard_error(self):
        original = self.server._read_viewer_asset

        def broken(name):
            if name == "viewer_v2.html":
                return "<html>no markers here</html>"
            return original(name)

        self.server._read_viewer_asset = broken
        try:
            with self.assertRaisesRegex(RuntimeError, "inline marker"):
                self.server._viewer_html()
        finally:
            self.server._read_viewer_asset = original


def edge_result(stream_id: str, frame_id: int, capture_ns: int, world_x: float) -> dict:
    record = {
        "stream_id": stream_id,
        "capture_timestamp_ns": capture_ns,
        "frame_id": frame_id,
        "class_name": "vehicle",
        "score": 0.9,
        "world_x": world_x,
        "world_y": 28.0,
        "world_z": 0.5,
        "size_x": 4.6,
        "size_y": 2.0,
        "size_z": 1.5,
        "yaw_sin": 0.0,
        "yaw_cos": 1.0,
        "candidate_identity": f"{frame_id}:0:{int(world_x * 100)}",
    }
    return {
        "schema": "splitfusion_edge_result.v2",
        "stream_id": stream_id,
        "frame_id": frame_id,
        "capture_timestamp_ns": capture_ns,
        "action_id": 50,
        "object_map_update": {
            "schema": "splitfusion_object_map_update.v1",
            "stream_id": stream_id,
            "frame_id": frame_id,
            "capture_timestamp_ns": capture_ns,
            "action_id": 50,
            "records": [record],
        },
    }


def live_view_update(
    ue_id: str, stream_id: str, frame_id: int, capture_ns: int, world_x: float,
    ego_x: float,
) -> dict:
    return {
        "schema": "scenesense.multi_ue_live_view_update.v1",
        "ue_id": ue_id,
        "session_id": SESSION_ID,
        "clock_domain": "carla_simulation_ns",
        "horizontal_fov_deg": 120.0,
        "ego_pose": {
            "actor_id": 1,
            "location": {"x": ego_x, "y": 28.0, "z": 0.6},
            "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
        },
        "payload": edge_result(stream_id, frame_id, capture_ns, world_x),
    }


class ServerPresentationEndpointTests(unittest.TestCase):
    """Health, metrics and snapshot behaviour of the running map service."""

    def setUp(self) -> None:
        sys.path.insert(0, str(REPO))
        from spatial_map_coop import spatial_map_server_moving_ego as server
        self.server = server
        argv = [
            "spatial_map_server_moving_ego.py",
            "--multi-ue-v1",
            "--multi-ue-session-id", SESSION_ID,
            "--multi-ue-source", f"{STREAM_A}=ue-a",
            "--multi-ue-source", f"{STREAM_B}=ue-b",
            "--focus-follow-stream-id", STREAM_A,
            "--focus-radius-m", "40",
            "--focus-padding-m", "0",
            "--stream-stale-s", "10",
            "--output-dir", os.environ.get("TMPDIR", "/tmp") + "/viewer_presentation_test",
        ]
        original_argv = sys.argv
        sys.argv = argv
        try:
            server.CONFIG = server.parse_args()
        finally:
            sys.argv = original_argv

        # Reset every module-level accumulator so tests are independent.
        server.latest_streams.clear()
        server.fusion_tracks.clear()
        server.ingest_counters.clear()
        server.latest_service_diagnostic.clear()
        server.latest_service_diagnostic["status"] = "WAITING_FOR_REGISTERED_UE_UPDATES"
        server._initialize_multi_ue_service()
        server._initialize_presentation_contracts()

        # Pre-load the recorded static town geometry: no CARLA query is made.
        geometry = json.loads(STATIC_GEOMETRY.read_text())
        server.static_map_cache.update(
            {
                "loaded_at": 1.0,
                "map_name": geometry["map_name"],
                "roads": geometry["roads"],
                "buildings": [{"footprint": item} for item in geometry["buildings"]],
                "traffic_lights": [],
                "error": None,
            }
        )
        self.client = server.app.test_client()

    def post_pair(self, frame_a: int, frame_b: int, capture_a: int, capture_b: int,
                  x_a: float, x_b: float):
        first = self.client.post(
            "/api/multi_ue/v1/live_updates",
            json=live_view_update("ue-a", STREAM_A, frame_a, capture_a, x_a, 0.0),
        )
        second = self.client.post(
            "/api/multi_ue/v1/live_updates",
            json=live_view_update("ue-b", STREAM_B, frame_b, capture_b, x_b, -15.0),
        )
        return first, second

    # ---- health counters come from the multi-UE service -------------------
    def test_health_reports_multi_ue_service_state_not_fusion_tracks(self):
        # A non-empty historical structure must not be mistaken for map tracks.
        self.server.fusion_tracks["stale-historical"] = {"ignored": True}
        self.post_pair(10, 20, 1_000_000_000, 1_030_000_000, 11.0, 11.3)
        self.client.get("/api/spatial_map/latest")

        health = self.client.get("/healthz").get_json()
        service = health["multi_ue_service"]
        self.assertTrue(service["multi_ue_v1_enabled"])
        self.assertEqual(service["implementation"], "CONSERVATIVE_MULTI_UE_V1")
        self.assertEqual(service["status"], "ok")
        self.assertEqual(service["input_source_count"], 2)
        self.assertEqual(service["association_count"], 1)
        self.assertEqual(service["multi_source_association_count"], 1)
        self.assertEqual(service["current_association_track_count"], 1)
        self.assertEqual(service["buffered_source_count"], 2)
        # The headline count is the service's, not the historical dictionary's.
        self.assertEqual(health["map_tracks"], 1)
        self.assertNotIn("fused_tracks", health)
        self.assertEqual(health["ingest_counters"]["accepted"], 2)

    def test_live_updates_healthz_reports_service_state(self):
        self.post_pair(10, 20, 1_000_000_000, 1_030_000_000, 11.0, 11.3)
        self.client.get("/api/spatial_map/latest")
        document = self.client.get(
            "/api/multi_ue/v1/live_updates/healthz"
        ).get_json()
        self.assertEqual(document["status"], "ok")
        self.assertEqual(
            document["registered_sources"], {STREAM_A: "ue-a", STREAM_B: "ue-b"}
        )
        service = document["multi_ue_service"]
        self.assertEqual(service["association_count"], 1)
        self.assertEqual(service["accepted_observation_count"], 2)
        self.assertIn("rejection_reasons", service)
        self.assertEqual(document["ingest_counters"]["accepted"], 2)

    def test_health_without_the_service_reports_the_historical_path(self):
        self.server.multi_ue_service = None
        self.server.fusion_tracks["t"] = {"x": 1}
        health = self.client.get("/healthz").get_json()
        service = health["multi_ue_service"]
        self.assertFalse(service["multi_ue_v1_enabled"])
        self.assertEqual(service["historical_fusion_tracks"], 1)
        self.assertEqual(health["map_tracks"], 1)

    # ---- status endpoints must not mutate scientific state ---------------
    def test_presentation_endpoints_never_advance_the_track_store(self):
        self.post_pair(10, 20, 1_000_000_000, 1_030_000_000, 11.0, 11.3)
        self.client.get("/api/spatial_map/latest")

        calls = []
        original = self.server.multi_ue_service.snapshot_latest

        def spy(*args, **kwargs):
            calls.append(kwargs)
            return original(*args, **kwargs)

        self.server.multi_ue_service.snapshot_latest = spy
        try:
            for _ in range(8):
                self.client.get("/healthz")
                self.client.get("/api/spatial_map/metrics")
                self.client.get("/api/multi_ue/v1/live_updates/healthz")
        finally:
            self.server.multi_ue_service.snapshot_latest = original

        self.assertEqual(
            calls, [],
            "a status/metrics endpoint took a service snapshot, which advances "
            "the conservative track store",
        )

    def test_metrics_endpoint_leaves_the_association_snapshot_unchanged(self):
        self.post_pair(10, 20, 1_000_000_000, 1_030_000_000, 11.0, 11.3)
        before = self.client.get("/api/spatial_map/latest").get_json()
        for _ in range(5):
            self.client.get("/api/spatial_map/metrics")
            self.client.get("/healthz")
        after = self.client.get("/api/spatial_map/latest").get_json()

        def scientific(document):
            return {
                "objects": [
                    {
                        key: value for key, value in item.items()
                        if key not in {"age_ns"}
                    }
                    for item in document["spatial_map_objects"]
                ],
                "raw": document["raw_spatial_map_objects"],
                "associations": document["object_associations"],
            }

        self.assertEqual(scientific(before), scientific(after))
        self.assertEqual(
            [item["update_count"] for item in after["spatial_map_objects"]],
            [item["update_count"] for item in before["spatial_map_objects"]],
        )

    # ---- honest transport labelling --------------------------------------
    def test_cross_clock_wall_aoi_is_the_incompatible_string(self):
        from spatial_map_coop.coop_input_v1 import live_metrics_v1
        self.post_pair(10, 20, 1_000_000_000, 1_030_000_000, 11.0, 11.3)
        metrics = self.client.get("/api/spatial_map/metrics").get_json()
        transport = metrics["transport"]
        self.assertEqual(
            transport["cross_clock_wall_aoi"], live_metrics_v1.INCOMPATIBLE_CLOCKS
        )
        self.assertNotIsInstance(transport["cross_clock_wall_aoi"], (int, float))
        self.assertEqual(transport["capture_clock_domain"], "carla_simulation_ns")
        self.assertEqual(transport["receipt_clock_domain"], "unix_wall_ns")
        self.assertIn("demo-control delivery time", transport["http_post_duration"])
        self.assertEqual(transport["model_execution"], "localhost model path")
        blob = json.dumps(metrics).lower()
        for banned in live_metrics_v1.FORBIDDEN_LABELS:
            self.assertNotIn(banned, blob)

    def test_direct_latency_charts_stay_empty_on_this_transport(self):
        from spatial_map_coop.coop_input_v1 import live_metrics_v1
        self.post_pair(10, 20, 1_000_000_000, 1_030_000_000, 11.0, 11.3)
        metrics = self.client.get("/api/spatial_map/metrics").get_json()
        self.assertFalse(metrics["direct_edge_timing_available"])
        self.assertEqual(len(metrics["latency_windows"]), 3)
        for window in metrics["latency_windows"]:
            self.assertEqual(window["count"], 0)
            self.assertEqual(window["current"], live_metrics_v1.NOT_AVAILABLE)
            self.assertEqual(window["p50"], live_metrics_v1.NOT_AVAILABLE)
            self.assertEqual(window["p95"], live_metrics_v1.NOT_AVAILABLE)
            self.assertIn("->", window["label"])
        self.assertEqual(metrics["terminal_outcome_counts"], {})

    def test_service_windows_and_stream_rates_are_reported(self):
        from spatial_map_coop.coop_input_v1 import live_metrics_v1
        self.post_pair(10, 20, 1_000_000_000, 1_030_000_000, 11.0, 11.3)
        self.client.get("/api/spatial_map/latest")
        metrics = self.client.get("/api/spatial_map/metrics").get_json()
        labels = {window["label"] for window in metrics["service_windows"]}
        self.assertIn(live_metrics_v1.LABEL_SNAPSHOT_BUILD, labels)
        self.assertIn(live_metrics_v1.LABEL_DEMO_POST, labels)
        build = next(
            w for w in metrics["service_windows"]
            if w["label"] == live_metrics_v1.LABEL_SNAPSHOT_BUILD
        )
        self.assertGreaterEqual(build["count"], 1)
        self.assertEqual(set(metrics["streams"]), {STREAM_A, STREAM_B})
        for stream in metrics["streams"].values():
            self.assertEqual(stream["accepted_updates"], 1)
            # One sample cannot define a rate; it must be N/A, not zero.
            self.assertEqual(stream["update_rate_hz"], live_metrics_v1.NOT_AVAILABLE)

    def test_rejected_and_duplicate_updates_are_counted(self):
        self.post_pair(10, 20, 1_000_000_000, 1_030_000_000, 11.0, 11.3)
        # An unregistered stream must be refused and counted.
        bad = live_view_update("ue-a", "unregistered/stream", 11, 1_040_000_000, 1.0, 0.0)
        response = self.client.post("/api/multi_ue/v1/live_updates", json=bad)
        self.assertEqual(response.status_code, 422)
        counters = self.client.get("/healthz").get_json()["ingest_counters"]
        self.assertEqual(counters["accepted"], 2)
        self.assertGreaterEqual(counters["conflict"] + counters["rejected"], 1)
        for key in ("accepted", "duplicate_identical", "conflict", "rejected"):
            self.assertIn(key, counters)

    # ---- capture clock exposure for the causal predictor ------------------
    def test_snapshot_exposes_the_measured_capture_stamp_and_its_clock(self):
        self.post_pair(10, 20, 1_000_000_000, 1_030_000_000, 11.0, 11.3)
        snapshot = self.client.get("/api/spatial_map/latest").get_json()
        streams = {item["stream_id"]: item for item in snapshot["active_streams"]}
        self.assertEqual(set(streams), {STREAM_A, STREAM_B})
        self.assertEqual(streams[STREAM_A]["capture_timestamp_ns"], 1_000_000_000)
        self.assertEqual(streams[STREAM_B]["capture_timestamp_ns"], 1_030_000_000)
        for stream in streams.values():
            self.assertEqual(stream["capture_clock_domain"], "carla_simulation_ns")

    def test_absent_capture_stamp_is_none_rather_than_zero(self):
        self.assertIsNone(
            self.server._stream_capture_timestamp_ns(
                {"carla_timestamp": 0.0, "capture_timestamp_ns": 0}
            )
        )
        self.assertEqual(
            self.server._stream_capture_timestamp_ns({"carla_timestamp": 2.5}),
            2_500_000_000,
        )
        self.assertIsNone(self.server._stream_capture_timestamp_ns({}))

    # ---- inventory projection through the real endpoints -----------------
    def test_raw_and_associated_projections_agree_with_the_service(self):
        self.post_pair(10, 20, 1_000_000_000, 1_030_000_000, 11.0, 11.3)
        snapshot = self.client.get("/api/spatial_map/latest").get_json()
        # Two raw reports, one A+B association, one representation.
        self.assertEqual(len(snapshot["raw_spatial_map_objects"]), 2)
        self.assertEqual(len(snapshot["spatial_map_objects"]), 1)
        track = snapshot["spatial_map_objects"][0]
        self.assertEqual(sorted(track["source_stream_ids"]), [STREAM_A, STREAM_B])
        self.assertEqual(track["support_stream_count"], 2)
        self.assertEqual(
            track["fusion_method"], "SELECTED_FRESHEST_SOURCE_NO_POSITION_AVERAGING"
        )
        # Freshest-source selection, no averaging: the position is UE-B's exactly.
        self.assertEqual(track["location"]["x"], 11.3)
        projection = snapshot["metadata"]["multi_ue_v1"]["viewer_projection"]
        self.assertEqual(projection["policy"], "CURRENT_ASSOCIATION_TRACKS_ONLY")
        self.assertEqual(projection["historical_tracks_not_rendered"], 0)

    def test_historical_tracks_are_never_presented_as_current_associations(self):
        self.post_pair(10, 20, 1_000_000_000, 1_030_000_000, 11.0, 11.3)
        self.client.get("/api/spatial_map/latest")
        # A second, far-away pair creates a new association. The previous track
        # is retained in service state but must not be projected.
        self.post_pair(12, 22, 1_200_000_000, 1_230_000_000, 60.0, 60.3)
        snapshot = self.client.get("/api/spatial_map/latest").get_json()
        projection = snapshot["metadata"]["multi_ue_v1"]["viewer_projection"]
        self.assertGreaterEqual(projection["retained_service_track_count"], 1)
        self.assertEqual(
            projection["rendered_track_count"], len(snapshot["spatial_map_objects"])
        )
        rendered_ids = {
            item["association_id"] for item in snapshot["spatial_map_objects"]
        }
        current_ids = {
            item["association_id"] for item in snapshot["object_associations"]
        }
        self.assertTrue(rendered_ids <= current_ids)
        self.assertLessEqual(
            len(snapshot["spatial_map_objects"]),
            len(snapshot["raw_spatial_map_objects"]),
        )

    # ---- route contract surfaces honestly --------------------------------
    def test_stage2_default_never_advertises_route_b_or_an_overlay(self):
        geometry = self.client.get("/api/spatial_map/static_geometry").get_json()
        self.assertIsNone(geometry["route_overlay"])
        health = self.client.get("/healthz").get_json()
        contract = health["route_contract"]
        self.assertEqual(contract["mode"], "stage2")
        self.assertFalse(contract["is_route_b"])
        self.assertFalse(contract["route_overlay_available"])
        self.assertFalse(contract["live_route_b_run_authorized"])
        self.assertEqual(contract["qualification_status"], "STAGE2_LOOP_NOT_ROUTE_B")

    def test_route_b_overlay_is_served_only_with_a_verified_contract(self):
        self.server.CONFIG.route_mode = "route-b-passive"
        self.server.CONFIG.route_overlay = True
        self.server._initialize_presentation_contracts()
        geometry = self.client.get("/api/spatial_map/static_geometry").get_json()
        overlay = geometry["route_overlay"]
        self.assertIsNotNone(overlay)
        self.assertGreater(len(overlay["points"]), 10)
        self.assertEqual(len(overlay["route_sha256"]), 64)
        self.assertIn("display overlay only", overlay["note"])
        self.assertEqual(
            overlay["qualification_status"],
            "ROUTE_B_CONTRACT_BOUND_LIVE_RUN_UNAUTHORIZED",
        )

    def test_overlay_is_refused_when_the_route_hash_cannot_be_verified(self):
        self.server.CONFIG.route_mode = "route-b-passive"
        self.server.CONFIG.route_overlay = True
        from spatial_map_coop.coop_input_v1 import route_contract_v1
        original = route_contract_v1.CANONICAL_ROUTE_B_SHA256
        route_contract_v1.CANONICAL_ROUTE_B_SHA256 = "0" * 64
        try:
            self.server._initialize_presentation_contracts()
            geometry = self.client.get("/api/spatial_map/static_geometry").get_json()
            self.assertIsNone(geometry["route_overlay"])
            self.assertIn("hash mismatch", geometry["route_overlay_error"])
        finally:
            route_contract_v1.CANONICAL_ROUTE_B_SHA256 = original

    def test_overlay_is_refused_in_stage2_mode_even_if_requested(self):
        self.server.CONFIG.route_mode = "stage2"
        self.server.CONFIG.route_overlay = True
        self.server._initialize_presentation_contracts()
        geometry = self.client.get("/api/spatial_map/static_geometry").get_json()
        self.assertIsNone(geometry["route_overlay"])
        self.assertIn("route-b-passive", geometry["route_overlay_error"])

    def test_viewer_route_serves_the_presentation_shell(self):
        response = self.client.get("/api/spatial_map/viewer")
        self.assertEqual(response.status_code, 200)
        body = response.get_data(as_text=True)
        self.assertIn("ScenesenseMapPresentation", body)
        self.assertIn("ScenesenseMapInventory", body)
        self.assertIn("/api/spatial_map/metrics", body)


class HarnessRouteModeTests(unittest.TestCase):
    """The live harness must never label the Stage-2 loop as Route B."""

    def setUp(self) -> None:
        sys.path.insert(0, str(REPO))
        from spatial_map_coop.multi_ue_v1 import live_two_ue_action50_v1 as harness
        self.harness = harness

    def namespace(self, **overrides):
        from argparse import Namespace
        values = {
            "route_mode": "stage2",
            "fixed_route_spawn_indices": self.harness.STAGE2_DEFAULT_SPAWN_INDICES,
            "route_b_path": "",
            "require_route_progress": False,
        }
        values.update(overrides)
        return Namespace(**values)

    def test_stage2_is_the_default_and_is_not_route_b(self):
        contract = self.harness._require_route_contract(self.namespace())
        self.assertFalse(contract.is_route_b)
        self.assertEqual(contract.stage2_spawn_indices, (80, 85, 91, 94, 99, 80))
        self.assertEqual(contract.qualification_status, "STAGE2_LOOP_NOT_ROUTE_B")

    def test_route_b_passive_binds_the_hash_verified_canonical_route(self):
        contract = self.harness._require_route_contract(
            self.namespace(route_mode="route-b-passive")
        )
        self.assertTrue(contract.is_route_b)
        self.assertEqual(contract.route.waypoint_count, 589)
        self.assertEqual(
            contract.registered_stream_ids, self.harness.REGISTERED_STREAM_IDS
        )
        self.assertEqual(len(contract.registered_stream_ids), 2)

    def test_route_b_passive_refuses_a_missing_route_file(self):
        with self.assertRaisesRegex(self.harness.LiveTwoUEError, "route contract refused"):
            self.harness._require_route_contract(
                self.namespace(route_mode="route-b-passive", route_b_path="/nope.json")
            )

    def test_two_registered_identities_are_declared(self):
        self.assertEqual(len(self.harness.REGISTERED_STREAM_IDS), 2)
        self.assertEqual(len(self.harness.REGISTERED_UE_IDS), 2)
        self.assertEqual(
            len(set(self.harness.REGISTERED_STREAM_IDS)),
            len(self.harness.REGISTERED_STREAM_IDS),
        )


if __name__ == "__main__":
    unittest.main()
