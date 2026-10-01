"""Boundary, parity and determinism tests for the operational-latency provider.

CPU-only; reads only the pinned retained evidence (copied to a temporary
tree for mutation tests).  No CARLA, OAI, Docker, CUDA or network use.
"""

from __future__ import annotations

import csv
import inspect
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rl_agent.splitfusion_operational_latency_v1 import provider as P


def _copy_sources(target: Path) -> dict[str, Path]:
    paths = {}
    for name, (relpath, _) in P.SOURCES.items():
        destination = target / relpath
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(P.ROOT / relpath, destination)
        paths[name] = destination
    return paths


def _rewrite_csv(path: Path, mutate) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    changed = 0
    for index, row in enumerate(rows):
        changed += mutate(index, row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return changed


GT_PROBE_COLUMNS = ("evaluation_enqueued_to_ue_receive_ms",
                    "evaluation_started_to_ue_receive_ms")
MAP_INGEST_COLUMNS = ("map_ingest_at", "map_install_at", "install_timestamp",
                      "map_worker_start_at", "map_lock_request_at",
                      "map_lock_acquired_at", "map_lock_released_at",
                      "association_start_at", "association_end_at",
                      "install_latency_from_publish_ms",
                      "install_latency_from_tail_ms", "map_age_at_install_ms")


def _shift(row, columns, delta):
    count = 0
    for column in columns:
        try:
            row[column] = repr(float(row[column]) + delta)
            count += 1
        except (KeyError, ValueError):
            pass
    return count


def _mutate_gt_and_map(paths: dict[str, Path]) -> int:
    changed = 0
    for name in ("probe_favorable", "probe_adverse"):
        changed += _rewrite_csv(
            paths[name], lambda i, row: _shift(row, GT_PROBE_COLUMNS, 37.0 + i))
    changed += _rewrite_csv(
        paths["direct_map_ingest"],
        lambda i, row: _shift(row, MAP_INGEST_COLUMNS, 0.25 + i * 1e-3))
    evidence = json.loads(paths["phase6_ue_evidence"].read_text())
    for key in list(evidence):
        if key.startswith("gt_"):
            evidence[key] = {"mutated": True}
            changed += 1
    for row in evidence["feedback_rows"]:
        row["q_perc"] = 0.123
        changed += 1
    for row in evidence["resolutions"]:
        row["q_perc"] = 0.456
        row["latency_ms"] = 999.0
        changed += 1
    paths["phase6_ue_evidence"].write_text(json.dumps(evidence))
    return changed


def _trajectory(provider: P.OperationalLatencyProviderV1, seed: int, n: int):
    streams = provider.streams(seed)
    rows = []
    for index in range(n):
        draw = provider.draw(streams)
        outcome = provider.resolve(
            draw=draw, wire_bytes=10_000 + 997 * (index % 300),
            pre_enqueue_backlog_bytes=0, prior_ul_mcs=20 + index % 8,
            success_uniform=(index * 0.6180339887) % 1.0)
        rows.append((draw.to_dict(), outcome.to_dict()))
    return rows


class ProviderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.provider = P.OperationalLatencyProviderV1.load()

    def test_pinned_counts_hashes_and_label(self) -> None:
        for name, (count, digest) in P.EXPECTED_POOLS.items():
            self.assertEqual(len(self.provider.pool(name).values_ns), count)
            self.assertEqual(self.provider.pool(name).sha256, digest)
        binding = self.provider.binding_document()
        self.assertEqual(binding["label"],
                         "EXPLORATORY_POOLED_FAMILY_TRANSFER_ASSUMPTION")
        self.assertEqual(binding["family_conditioning"], "NONE_POOLED_A_AND_E")
        for name, (relpath, digest) in P.SOURCES.items():
            self.assertEqual(binding["sources"][name],
                             {"relpath": relpath, "sha256": digest})
        for excluded in ("GT_WAIT", "GT_SCORING", "QPERC_COMPUTATION",
                         "MAP_INSTALLATION", "OLD_ACTOR_RESERVE"):
            self.assertIn(excluded, binding["excluded_components"])
        self.assertFalse(binding["pools"]["E_HOLD_SENSITIVITY"]["used_by_compose"])
        self.assertEqual(P.OperationalLatencyProviderV1.load().binding_sha256,
                         self.provider.binding_sha256)

    def test_all_pool_values_nonnegative_ints(self) -> None:
        for name in P.EXPECTED_POOLS:
            values = self.provider.pool(name).values_ns
            self.assertTrue(all(type(v) is int and v >= 0 for v in values))

    def test_gt_and_map_mutation_changes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _copy_sources(Path(tmp))
            self.assertGreater(_mutate_gt_and_map(paths), 1000)
            mutated = P.OperationalLatencyProviderV1.unpinned_for_tests(paths)
            for name in P.EXPECTED_POOLS:
                self.assertEqual(mutated.pool(name).values_ns,
                                 self.provider.pool(name).values_ns)
            self.assertEqual(mutated.binding_document()["pools"],
                             self.provider.binding_document()["pools"])
            self.assertEqual(_trajectory(mutated, 17, 400),
                             _trajectory(self.provider, 17, 400))
            # The pinned loader still refuses the mutated files.
            with self.assertRaisesRegex(P.ProviderError, "source hash drift"):
                P.OperationalLatencyProviderV1.load(Path(tmp))

    def test_in_boundary_mutation_is_detected(self) -> None:
        """Non-vacuity control: the extraction does read A/E/D columns."""
        with tempfile.TemporaryDirectory() as tmp:
            paths = _copy_sources(Path(tmp))
            _rewrite_csv(paths["direct_map_ingest"], lambda i, row: _shift(
                row, ("edge_publish_start_wall_s",), 0.001))
            _rewrite_csv(paths["probe_favorable"], lambda i, row: _shift(
                row, (P.D_COLUMN,), 1.0))
            evidence = json.loads(paths["phase6_ue_evidence"].read_text())
            for row in evidence["decisions"]:
                stages = row.get("stages") or {}
                if "first_packet_send_raw_ns" in stages:
                    stages["first_packet_send_raw_ns"] += 1000
            paths["phase6_ue_evidence"].write_text(json.dumps(evidence))
            mutated = P.OperationalLatencyProviderV1.unpinned_for_tests(paths)
            for name in ("A", "E", "D"):
                self.assertNotEqual(mutated.pool(name).sha256,
                                    self.provider.pool(name).sha256, name)

    def test_each_component_exactly_once(self) -> None:
        parts = {"a_ns": 1_000_003, "s_ns": 20_011, "t_ns": 300_000_007,
                 "e_ns": 4_000_037, "d_ns": 50_000_017}
        self.assertEqual(P.compose_total_ns(**parts), sum(parts.values()))
        for name, value in parts.items():
            dropped = dict(parts, **{name: 0})
            self.assertEqual(P.compose_total_ns(**parts)
                             - P.compose_total_ns(**dropped), value)
        draw = self.provider.draw(self.provider.streams(29))
        outcome = self.provider.resolve(
            draw=draw, wire_bytes=50_000, pre_enqueue_backlog_bytes=0,
            prior_ul_mcs=24, success_uniform=0.0)
        self.assertEqual(outcome.total_ns, outcome.a_ns + outcome.s_ns
                         + outcome.t_ns + outcome.e_ns + outcome.d_ns)
        self.assertEqual((outcome.a_ns, outcome.e_ns, outcome.d_ns),
                         (draw.a_ns, draw.e_ns, draw.d_ns))
        self.assertEqual(outcome.s_ns, round(0.513047 * 50_000))
        prediction = self.provider.transport_model.predict(
            pre_enqueue_backlog_bytes=0.0, wire_bytes=50_000, prior_ul_mcs=24)
        self.assertEqual(outcome.t_ns,
                         int(round(prediction.conditional_latency_ms * 1e6)))

    def test_inclusive_deadline(self) -> None:
        self.assertTrue(P.is_timely(transport_success=True,
                                    total_ns=170_000_000))
        self.assertFalse(P.is_timely(transport_success=True,
                                     total_ns=170_000_001))
        self.assertFalse(P.is_timely(transport_success=False, total_ns=1))

    def test_deterministic_streams_and_exact_resume(self) -> None:
        self.assertEqual(_trajectory(self.provider, 43, 300),
                         _trajectory(self.provider, 43, 300))
        self.assertNotEqual(_trajectory(self.provider, 43, 50),
                            _trajectory(self.provider, 17, 50))
        streams = self.provider.streams(43)
        for _ in range(123):
            self.provider.draw(streams)
        state = json.loads(json.dumps(streams.get_state()))
        expected = [self.provider.draw(streams).to_dict() for _ in range(200)]
        restored = self.provider.streams(43)
        restored.set_state(state)
        self.assertEqual([self.provider.draw(restored).to_dict()
                          for _ in range(200)], expected)
        seeds = {P.stream_seed(43, label) for label in P.STREAM_LABELS}
        self.assertEqual(len(seeds), 3)

    def test_streams_are_independent(self) -> None:
        reference = self.provider.streams(17)
        perturbed = self.provider.streams(17)
        perturbed._rngs["A"].random()  # advance only stream A
        a_ref, e_ref, d_ref = zip(*[reference.draw_indices() for _ in range(64)])
        a_new, e_new, d_new = zip(*[perturbed.draw_indices() for _ in range(64)])
        self.assertNotEqual(a_ref, a_new)
        self.assertEqual((e_ref, d_ref), (e_new, d_new))

    def test_resolve_takes_no_family_mode_or_quality(self) -> None:
        parameters = set(inspect.signature(
            P.OperationalLatencyProviderV1.resolve).parameters)
        self.assertEqual(parameters, {"self", "draw", "wire_bytes",
                                      "pre_enqueue_backlog_bytes",
                                      "prior_ul_mcs", "success_uniform"})

    def test_import_is_side_effect_free(self) -> None:
        code = (
            "import sys\n"
            "opened=[]\n"
            "sys.addaudithook(lambda e,a: opened.append(str(a[0])) "
            "if e=='open' else None)\n"
            "import rl_agent.splitfusion_operational_latency_v1.provider\n"
            "print([o for o in opened if 'experiments' in o])\n")
        result = subprocess.run([sys.executable, "-c", code], cwd=P.ROOT,
                                capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), "[]")


if __name__ == "__main__":
    unittest.main()
