"""Phase-6 edge ready-record producer/consumer contract parity (addendum 4).

The producer is ``phase6_edge_runtime_v2.ready_document``, the exact builder the
live write site uses. The consumer is the **unchanged** ready predicate inside
``adapter_direct_v1.start_direct_live_edge``, executed for real. A fake launcher
writes the produced document where the container would, so no Docker, CUDA or
container runs, and the predicate is neither copied nor weakened.
"""

from __future__ import annotations

import ast
import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from rl_agent import ue_route_b_split_cell_adapter_v1 as pinned
from rl_agent.splitfusion_direct_edge_map_v1 import adapter_direct_v1 as D
from rl_agent.splitfusion_quality_feedback_probe_v1 import live_probe as LP

from . import phase6_edge_launch_v2 as EL
from . import phase6_edge_runtime_v2 as ER
from . import phase6_live_child_v2 as C
from . import phase6_live_runner_v2 as RUN
from . import run4_live_wire_v2 as W

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "rl_agent/splitfusion_hybrid_sac_live_route_b_v2"
REPAIR_BASE = "c5a6d0418c0d1c732edef07a9decb5dcdcd867e8"
MAP_HOST = "192.168.70.129"
SPEC_SHA = "d5d1e0d2d435076dd53c740f8b0e632620144194c32baf7d24db6d5043fc74d9"
EXPECTED_KEYS = {
    "schema", "run4_edge", "action_id", "direct_map_host", "direct_map_port",
    "ue_control_host", "ue_control_port", "tail_device", "architecture",
    "quality_spec_sha256", "dense_label_map_on_radio", "object_records_on_radio",
    "evaluation_evidence_dir",
}
RESULT_TERMS = ("q_perc", "reward", "latency", "success", "iou", "recall", "tp", "fp",
                "fn", "error", "score", "quality_value", "measurement")
FROZEN_SCIENTIFIC = (
    "phase6_live_child_v2.py", "phase6_decision_engine_v2.py", "reward_hold_controller_v2.py",
    "live_state_v2.py", "run4_live_wire_v2.py", "run4_map_protocol_v2.py",
    "run4_ue_ledger_v2.py", "continuous_execution_v2.py", "frozen_actor_v2.py",
    "phase6_ue_runtime_v2.py", "phase6_map_server_v2.py", "phase6_result_reporting_v2.py",
    "phase6_prospective_addendum_2.json", "live_qualification_300_v2.json",
    "ACTOR_BINDING_V2.json", "phase6_edge_launch_v2.py", "phase6_live_child_nobuild_v2.py",
    "phase6_setup_repair_addendum_3.json",
)


def campaign_and_cell():
    config = json.loads(RUN.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    campaign = LP._probe_campaign(config, run_id="contract")
    campaign["campaign_id"] = "splitfusion_run4_phase6_v2/contract"
    cell = {"cell_id": "run4p6_contract_a71__favorable_stable", "action_id": 71}
    return campaign, cell


def produced(campaign, cell, *, extra_args: dict) -> dict:
    """The real producer, fed exactly what the live edge would read."""
    config = C.edge_config(campaign, cell, pinned.EDGE_EVIDENCE_LEAF)
    return ER.ready_document(
        action_id=int(extra_args["--action-id"]),
        direct_map_host=extra_args["--direct-map-host"],
        direct_map_port=int(extra_args["--direct-map-port"]),
        ue_control_host=extra_args["--ue-control-host"],
        ue_control_port=int(extra_args["--ue-control-port"]),
        tail_device="cuda:0", quality_spec_sha256=SPEC_SHA,
        evidence_dir=Path(config["evidence_dir"]))


class _FakeSubprocess:
    """Stands in for ``subprocess`` inside the adapter; runs nothing."""

    def __init__(self, write_ready) -> None:
        self._write_ready = write_ready
        self.CompletedProcess = subprocess.CompletedProcess
        self.PIPE, self.STDOUT, self.DEVNULL = subprocess.PIPE, subprocess.STDOUT, subprocess.DEVNULL

    def run(self, args, *pos, **kwargs):
        if [str(a) for a in args] == [str(EL.LEGACY_LAUNCHER)]:
            self._write_ready(kwargs["env"])
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")


class ConsumerParityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.saved = {name: getattr(pinned, name) for name in
                      ("tail_running", "seed_cell_edge_state")}
        self.saved_subprocess = D.subprocess
        self.saved_endpoint = dict(D._ENDPOINT)
        self.campaign, self.cell = campaign_and_cell()
        D._ENDPOINT.update({"endpoint": SimpleNamespace(host=MAP_HOST),
                            "config_relpath": self.campaign["runtime"][
                                "direct_edge_config_relpath"]})
        self.launched = 0
        pinned.tail_running = lambda: self.launched > 0
        pinned.seed_cell_edge_state = lambda campaign, scratch: None
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        for name, value in self.saved.items():
            setattr(pinned, name, value)
        D.subprocess = self.saved_subprocess
        D._ENDPOINT.clear()
        D._ENDPOINT.update(self.saved_endpoint)
        self.tmp.cleanup()

    def start(self, mutate=None):
        def write_ready(env):
            self.launched += 1
            words = env["FUSION_BACK_EXTRA_ARGS"].split()
            extra = {words[i]: words[i + 1] for i in range(len(words) - 1)
                     if words[i].startswith("--") and not words[i + 1].startswith("--")}
            document = produced(self.campaign, self.cell, extra_args=extra)
            if mutate is not None:
                document = mutate(copy.deepcopy(document))
            self.document = document
            host_ready = Path(env["SPLITFUSION_EDGE_STATE_ROOT"]) / Path(
                extra["--ready-file"]).relative_to("/work/torch_cache")
            host_ready.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")

        D.subprocess = _FakeSubprocess(write_ready)
        return D.start_direct_live_edge(self.campaign, self.cell, Path(self.tmp.name))

    def test_real_ready_document_passes_unchanged_consumer(self) -> None:
        scratch = self.start()
        self.assertTrue(Path(scratch).is_dir())
        expected_dir = str(Path("/work/torch_cache") / pinned.EDGE_EVIDENCE_LEAF)
        self.assertEqual(self.document["evaluation_evidence_dir"], expected_dir)
        self.assertIs(self.document["dense_label_map_on_radio"], False)
        self.assertIs(self.document["object_records_on_radio"], False)
        self.assertEqual(self.document["direct_map_host"], MAP_HOST)
        self.assertEqual(self.document["direct_map_port"],
                         int(self.campaign["runtime"]["direct_map_ingest_port"]))

    def test_every_consumer_condition_rejects_a_violation(self) -> None:
        def drop(key):
            return lambda d: {k: v for k, v in d.items() if k != key}

        def put(key, value):
            return lambda d: {**d, key: value}

        cases = {
            "schema absent": drop("schema"),
            "schema wrong": put("schema", "splitfusion_direct_live_edge_ready.v0"),
            "architecture absent": drop("architecture"),
            "architecture wrong": put("architecture", "UE_FORWARDED"),
            "tail_device absent": drop("tail_device"),
            "tail_device cpu": put("tail_device", "cpu"),
            "dense absent": drop("dense_label_map_on_radio"),
            "dense null": put("dense_label_map_on_radio", None),
            "dense truthy": put("dense_label_map_on_radio", True),
            "dense zero": put("dense_label_map_on_radio", 0),
            "objects absent": drop("object_records_on_radio"),
            "objects null": put("object_records_on_radio", None),
            "objects truthy": put("object_records_on_radio", True),
            "map host drift": put("direct_map_host", "10.0.0.2"),
            "map port drift": put("direct_map_port", 39321),
            "evidence absent": drop("evaluation_evidence_dir"),
            "evidence null": put("evaluation_evidence_dir", None),
            "evidence other path": put("evaluation_evidence_dir",
                                       "/work/torch_cache/other_evidence"),
        }
        for name, mutate in cases.items():
            with self.subTest(case=name):
                self.tearDown()
                self.setUp()
                with self.assertRaisesRegex(pinned.AdapterError,
                                            "ready record identity/endpoint drift"):
                    self.start(mutate)


class ProducerContentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.campaign, self.cell = campaign_and_cell()
        runtime = self.campaign["runtime"]
        self.document = produced(self.campaign, self.cell, extra_args={
            "--action-id": "71", "--direct-map-host": MAP_HOST,
            "--direct-map-port": str(runtime["direct_map_ingest_port"]),
            "--ue-control-host": str(runtime["ue_bind_host"]),
            "--ue-control-port": str(runtime["ue_control_port"])})

    def test_identity_fields_present_and_endpoints_distinct(self) -> None:
        self.assertEqual(set(self.document), EXPECTED_KEYS)
        self.assertIs(self.document["run4_edge"], True)
        self.assertEqual(self.document["action_id"], 71)
        self.assertEqual(self.document["quality_spec_sha256"], SPEC_SHA)
        self.assertEqual(W.load_run4_quality_spec(ROOT).canonical_sha256(), SPEC_SHA)
        self.assertNotEqual(
            (self.document["direct_map_host"], self.document["direct_map_port"]),
            (self.document["ue_control_host"], self.document["ue_control_port"]))
        self.assertNotEqual(self.document["direct_map_host"], self.document["ue_control_host"])

    def test_no_scientific_result_or_measurement(self) -> None:
        for key in self.document:
            for term in RESULT_TERMS:
                self.assertFalse(key == term or key.startswith(term + "_")
                                 or key.endswith("_" + term), (key, term))
        self.assertTrue(all(isinstance(v, (str, int, bool)) for v in self.document.values()))


class BoundedDiffTest(unittest.TestCase):
    """The runtime changed only by the pure builder and its use at the write site."""

    @staticmethod
    def _tree(source: str) -> ast.Module:
        return ast.parse(source)

    @staticmethod
    def _normalize_ready_write(function: ast.AST) -> str:
        for node in ast.walk(function):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "dump" and len(node.args) == 2
                    and isinstance(node.args[1], ast.Name) and node.args[1].id == "handle"):
                node.args[0] = ast.Name(id="READY_PAYLOAD", ctx=ast.Load())
        return ast.dump(function)

    def test_only_ready_payload_changed(self) -> None:
        old = subprocess.run(["git", "show", f"{REPAIR_BASE}:{PACKAGE}/phase6_edge_runtime_v2.py"],
                             cwd=ROOT, capture_output=True, check=True, text=True).stdout
        new = (ROOT / PACKAGE / "phase6_edge_runtime_v2.py").read_text(encoding="utf-8")
        old_nodes = list(self._tree(old).body)
        new_nodes = [n for n in self._tree(new).body
                     if getattr(n, "name", None) != "ready_document"]
        self.assertEqual(len(new_nodes), len(self._tree(new).body) - 1)   # one addition
        self.assertEqual(len(old_nodes), len(new_nodes))
        for before, after in zip(old_nodes, new_nodes):
            name = getattr(before, "name", type(before).__name__)
            self.assertEqual(getattr(after, "name", type(after).__name__), name)
            if name == "run_run4_edge_service":
                self.assertEqual(self._normalize_ready_write(before),
                                 self._normalize_ready_write(after))
            else:
                self.assertEqual(ast.dump(before), ast.dump(after), name)

    def test_write_site_uses_builder_with_config_evidence_dir(self) -> None:
        tree = self._tree((ROOT / PACKAGE / "phase6_edge_runtime_v2.py").read_text())
        service = next(n for n in tree.body if getattr(n, "name", "") == "run_run4_edge_service")
        calls = [n for n in ast.walk(service) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == "ready_document"]
        self.assertEqual(len(calls), 1)
        keywords = {k.arg: k.value for k in calls[0].keywords}
        self.assertIsInstance(keywords["evidence_dir"], ast.Name)
        self.assertEqual(keywords["evidence_dir"].id, "evidence_dir")
        assigns = [n for n in ast.walk(service) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "evidence_dir" for t in n.targets)]
        self.assertEqual(len(assigns), 1)
        self.assertIn("config['evidence_dir']", ast.unparse(assigns[0].value).replace('"', "'"))
        self.assertNotIn("edge_segmentation_evidence_dir", ast.unparse(service))

    def test_scientific_files_unchanged_since_repair_base(self) -> None:
        for name in FROZEN_SCIENTIFIC:
            committed = subprocess.run(["git", "show", f"{REPAIR_BASE}:{PACKAGE}/{name}"],
                                       cwd=ROOT, capture_output=True, check=True).stdout
            self.assertEqual((ROOT / PACKAGE / name).read_bytes(), committed, name)
        for path in ("rl_agent/splitfusion_direct_edge_map_v1/adapter_direct_v1.py",
                     "rl_agent/ue_route_b_split_cell_adapter_v1.py",
                     "scripts/receiver_container_fusion_back_up.sh",
                     "receiver_container/docker-compose.yaml",
                     "receiver_container/docker-compose.fusion-back.yaml",
                     "receiver_container/Dockerfile"):
            committed = subprocess.run(["git", "show", f"{REPAIR_BASE}:{path}"], cwd=ROOT,
                                       capture_output=True, check=True).stdout
            self.assertEqual((ROOT / path).read_bytes(), committed, path)


class AddendumTest(unittest.TestCase):
    def test_addendum_binds_base_and_both_failed_attempts(self) -> None:
        document = json.loads((ROOT / PACKAGE / "phase6_setup_repair_addendum_4.json")
                              .read_text(encoding="utf-8"))
        self.assertEqual(document["base_commit"], REPAIR_BASE)
        self.assertIs(document["scientific_protocol_changed"], False)
        self.assertEqual(sorted(document["added_ready_fields"]),
                         ["dense_label_map_on_radio", "evaluation_evidence_dir",
                          "object_records_on_radio"])
        attempts = document["preserved_failed_attempts"]
        self.assertEqual(len(attempts), 2)
        for attempt in attempts.values():
            for path, digest in attempt["sha256"].items():
                target = ROOT / path
                if not target.exists():
                    self.skipTest("failed-attempt evidence not present on this host")
                self.assertEqual(EL.sha256_file(target), digest, path)


if __name__ == "__main__":
    unittest.main()
