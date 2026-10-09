"""CPU-only E3 scope evidence checks; no process or profiler is launched."""
import json
from pathlib import Path
import tempfile
import unittest

from . import live_process_scope as S


SOURCES = {"producer.py": "a" * 64, "scorer.py": "b" * 64}
BASELINE = {"OMP_NUM_THREADS": "24", "MKL_NUM_THREADS": "24"}
STAMP = 123_456_789_000


def mapping(host, parent, ns, parent_ns, kind):
    return dict(schema=S.MAPPING_SCHEMA, host_pid=host, host_parent_pid=parent,
        nspid=ns, parent_nspid=parent_ns, proc_start_ticks=123456,
        observed_raw_ns=STAMP, candidate_kind=kind, status_sha256="c" * 64,
        stat_sha256="d" * 64, cmdline_sha256="e" * 64,
        inventory_sampled=True, inventory_t_raw_ns=STAMP, error=None)


def row(host, kind, *, container=False, variables=None):
    return dict(pid=host, role=kind, in_container=container, t_raw_ns=STAMP,
        variables={"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
                  if variables is None else variables)


def fixture():
    records = []
    for role, ns, parent in (("producer", 101, 51), ("scorer", 201, 71)):
        ready = dict(status="READY", role=role, pid=ns, parent_pid=parent,
            authentication="INHERITED_SOCKET_CAPABILITY_CREATOR_UID_PID_AND_NONCE",
            source_files=len(SOURCES), cuda_initialized=False)
        close = dict(closed=True, valid=True, retained_jobs=0, failure=None)
        records.append(dict(role=role, ready=ready, close=close, exitcode=0,
                            source_sha256=SOURCES))
    ready = dict(status="READY", context=dict(pid=202, parent_pid=71,
        parent_death_signal="SIGTERM", numerical_imports=[]),
        config=dict(binding=dict(numeric_worker_source_digest=S._digest(S._canonical(SOURCES)))))
    close = dict(child_pid=202, child_returncode=0, failed=False, fault=None,
                 outstanding_jobs=0)
    records.append(dict(role="archive", ready=ready, close=close, exitcode=0,
                        source_sha256=SOURCES))
    mappings = [mapping(1101, 1051, [1101, 101], [1051, 51], S.PROCESS_CANDIDATE),
                mapping(1201, 1071, [1201, 201], [1071, 71], S.PROCESS_CANDIDATE),
                mapping(1202, 1071, [1202, 202], [1071, 71], S.ARCHIVE_CANDIDATE)]
    inventory = [row(1101, S.PROCESS_CANDIDATE, container=True),
                 row(1201, S.PROCESS_CANDIDATE, container=True),
                 row(1202, S.ARCHIVE_CANDIDATE, container=True),
                 row(100, "ue"),
                 row(300, "oai", variables=BASELINE)]
    return inventory, mappings, records


class LiveProcessScopeTests(unittest.TestCase):
    def check(self, inventory, mappings, records):
        return S.verify_scope(inventory_rows=inventory, mappings=mappings,
            worker_scope_records=records, baseline=BASELINE, expected_sources=SOURCES)

    def test_exact_source_bound_workers_only_are_exempt(self):
        result = self.check(*fixture())
        self.assertTrue(result["scope_ok"], result)
        self.assertEqual(result["exempt_host_pids"], {1101: "producer", 1201: "scorer",
                                                       1202: "archive"})
        self.assertEqual(result["sampled_ue_processes"], 1)

    def test_unbound_candidate_and_other_baseline_violation_remain_failures(self):
        inventory, mappings, records = fixture()
        inventory.append(row(1300, S.PROCESS_CANDIDATE))
        inventory.append(row(1400, "oai", variables={"OMP_NUM_THREADS": "1",
                                                       "MKL_NUM_THREADS": "24"}))
        result = self.check(inventory, mappings, records)
        self.assertFalse(result["scope_ok"])
        self.assertTrue(any(item.get("reason") == "UNBOUND_NUMERIC_CANDIDATE"
                            for item in result["scope_violations"]))
        self.assertTrue(any(item.get("pid") == 1400 and item.get("variable") == "OMP_NUM_THREADS"
                            for item in result["scope_violations"]))

    def test_wrong_sources_pid_identity_and_unsampled_worker_fail_closed(self):
        inventory, mappings, records = fixture()
        records[0]["source_sha256"] = {"producer.py": "0" * 64}
        with self.assertRaisesRegex(S.ProcessScopeError, "foreign source"):
            self.check(inventory, mappings, records)
        inventory, mappings, records = fixture()
        mappings[0]["proc_start_ticks"] = 0
        self.assertFalse(self.check(inventory, mappings, records)["scope_ok"])
        inventory, mappings, records = fixture()
        mappings[1]["inventory_sampled"] = False
        self.assertFalse(self.check(inventory, mappings, records)["scope_ok"])
        inventory, mappings, records = fixture()
        mappings[2]["nspid"][-1] = 999
        self.assertFalse(self.check(inventory, mappings, records)["scope_ok"])

    def test_wrong_close_or_duplicate_inventory_is_rejected(self):
        inventory, mappings, records = fixture()
        records[2]["close"]["child_returncode"] = 1
        self.assertFalse(self.check(inventory, mappings, records)["scope_ok"])
        inventory, mappings, records = fixture()
        inventory.append(dict(inventory[0]))
        with self.assertRaisesRegex(S.ProcessScopeError, "duplicate host PID"):
            self.check(inventory, mappings, records)

    def test_exact_module_candidate_only(self):
        self.assertEqual(S.candidate_kind("python -m " + S.PROCESS_MODULE),
                         S.PROCESS_CANDIDATE)
        self.assertEqual(S.candidate_kind("/usr/bin/python -m " + S.ARCHIVE_MODULE + " --child"),
                         S.ARCHIVE_CANDIDATE)
        self.assertIsNone(S.candidate_kind("python -m " + S.PROCESS_MODULE + "_lookalike"))
        self.assertIsNone(S.candidate_kind("python " + S.PROCESS_MODULE))

    def test_proc_observation_binds_namespace_starttime_and_parent(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "1101").mkdir()
            (root / "1051").mkdir()
            def stat(pid, parent, start):
                tail = ["S", str(parent)] + ["0"] * 17 + [str(start), "0"]
                return (f"{pid} (python) " + " ".join(tail)).encode()
            (root / "1101" / "stat").write_bytes(stat(1101, 1051, 123456))
            (root / "1101" / "status").write_text("Name: python\nNSpid:\t1101\t101\n")
            (root / "1101" / "cmdline").write_bytes(("python\0-m\0" + S.PROCESS_MODULE + "\0").encode())
            (root / "1051" / "status").write_text("Name: parent\nNSpid:\t1051\t51\n")
            observed = S.observe_pid_mapping(1101, proc_root=root)
            self.assertEqual(observed["nspid"], [1101, 101])
            self.assertEqual(observed["parent_nspid"], [1051, 51])
            self.assertEqual(observed["proc_start_ticks"], 123456)
            self.assertEqual(observed["candidate_kind"], S.PROCESS_CANDIDATE)

    def test_exact_capture_file_loader_and_inventory_footer(self):
        inventory, mappings, records = fixture()
        with tempfile.TemporaryDirectory() as temp:
            capture = Path(temp) / "capture"
            cell = capture / "cells" / "a71__favorable_stable"
            evidence = capture / "threadcap"
            edge = cell / "profiler" / "edge"
            evidence.mkdir(parents=True)
            edge.mkdir(parents=True)
            (evidence / "PROCESS_ENV_INVENTORY.jsonl").write_text("".join(
                json.dumps(item) + "\n" for item in
                [*inventory, {"summary": {"rows": len(inventory), "errors": 0}}]))
            (evidence / S.MAPPING_FILENAME).write_text("".join(
                json.dumps(item) + "\n" for item in mappings))
            (cell / "numeric_ue_scope.json").write_text(json.dumps(dict(
                schema=S.WORKER_SCOPE_SCHEMA, workers=[records[0]])))
            (edge / "numeric_edge_scope.json").write_text(json.dumps(dict(
                schema=S.WORKER_SCOPE_SCHEMA, workers=records[1:])))
            result = S.verify_scope_files(capture_root=capture, cell_root=cell,
                baseline=BASELINE, expected_sources=SOURCES)
            self.assertTrue(result["scope_ok"])
            self.assertEqual(len(result["files"]), 4)
            bad = evidence / "PROCESS_ENV_INVENTORY.jsonl"
            bad.write_text(bad.read_text().replace('"errors": 0', '"errors": 1'))
            with self.assertRaisesRegex(S.ProcessScopeError, "failed"):
                S.verify_scope_files(capture_root=capture, cell_root=cell,
                    baseline=BASELINE, expected_sources=SOURCES)


if __name__ == "__main__":
    unittest.main()
