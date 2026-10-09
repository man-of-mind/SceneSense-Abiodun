"""CPU-only receipt contract tests with a modeled ArchiveClient completion."""
import copy
import json
import os
from pathlib import Path
import select
import tempfile
import time
from types import SimpleNamespace
import unittest

from rl_agent.splitfusion_hybrid_sac_v1.offline_quality_grid import quality
from rl_agent.splitfusion_live_gt_handoff_v1.archive import ArchiveClient
from rl_agent.splitfusion_minimal_quality_scorer_v1.scorer import NumericalScorer
from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
from rl_agent.splitfusion_numeric_gt_integration_v2 import scoring as Q
from rl_agent.splitfusion_numeric_gt_integration_v2.test_scoring import fixture
from rl_agent.splitfusion_numeric_gt_producer_v1.replay_v11 import quality_bridge

from . import live_archive as A, live_clock as CLOCK, live_receipt as R
from .test_live_archive import clock_for, source_sync


PREFIX = "frame_10/live/"


def scored_archive():
    scorer = NumericalScorer()
    case = fixture(scorer)
    target = case["target"]
    frame = Q.join_admitted_inputs(scorer, target=target, prediction=case["prediction"])
    result = scorer.score(frame)
    spec, _ = quality_bridge()
    measured = quality.evaluate_exact_quality(spec, result.measurement)
    q_perc = None if measured is None else float(measured.q_perc)
    # Other modules test the source and scorer observations. Keep those modeled
    # observations in the past so the receipt observations below are real RAW.
    sync = source_sync(target, time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW) - 500_000_000)
    ledger = clock_for(target, sync)
    assert all(event["raw_ns"] <= time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
               for event in ledger._events.values())
    archive = A.build_live_archive(ready_target=target, score_result=result, q_perc=q_perc,
        quality_spec=spec, input_metadata=case["source_kwargs"]["input_metadata"],
        source_synchronization=sync, numerical_avo=C.thaw(target.metadata["avo"]),
        clock_ledger=ledger)
    ledger.record(role="scorer", name="archive_build_end",
        raw_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW), pid=os.getpid(),
        boot_id=CLOCK.read_boot_id(), ready_identity_bytes=archive.ready_identity_bytes,
        semantic_sha256=archive.semantic_sha256)
    reply = dict(files=dict(archive.files),
        sha256={name: C.sha256(raw) for name, raw in archive.files.items()},
        archive_manifest_sha256=archive.manifest_sha256,
        ready_semantic_sha256=archive.semantic_sha256,
        identity_sha256=C.sha256(archive.ready_identity_bytes),
        terminal=ledger.deadline()["terminal"],
        clock_descriptor_bytes=archive.clock_descriptor_bytes, live_qualified=False)
    rebuilt = R.archive_from_scorer_reply(archive.ready_identity_bytes, reply)
    return rebuilt, ledger


def scorer_completion(archive, ledger, receipt, submitted, completed):
    actual = ledger.record(role=submitted["role"], name=submitted["name"],
        raw_ns=submitted["raw_ns"], pid=submitted["pid"], boot_id=submitted["boot_id"],
        ready_identity_bytes=archive.ready_identity_bytes,
        semantic_sha256=archive.semantic_sha256)
    assert actual == submitted
    raw = ledger.durability_record(descriptor_bytes=archive.clock_descriptor_bytes,
        receipt=receipt, raw_ns=completed["raw_ns"], pid=completed["pid"],
        boot_id=completed["boot_id"], ready_identity_bytes=archive.ready_identity_bytes,
        semantic_sha256=archive.semantic_sha256)
    return dict(status="COMPLETION_SIDECAR_PENDING_DURABILITY",
        terminal=ledger.deadline()["terminal"],
        archive_manifest_sha256=archive.manifest_sha256,
        clock_completion_bytes=raw, clock_completion_sha256=C.sha256(raw),
        live_qualified=False)


def modeled_completion(root, archive):
    """Model only the already-tested ArchiveClient state and exact durable files."""
    capture = C._parse(archive.ready_identity_bytes)["capture_identity"]
    directory = root / PREFIX
    directory.mkdir(parents=True)
    rows = []
    for name, raw in archive.files.items():
        (directory / name).write_bytes(raw)
        rows.append(dict(root=0, relative=PREFIX + name, sha256=C.sha256(raw),
                         bytes=len(raw), ready=False))
    receipt = dict(status="COMPLETE", durable=True, error=None, job_id=1,
                   identity=capture, pid=424242, written=rows, finalized_ready=[])
    client = object.__new__(ArchiveClient)
    client.config = {"roots": [{"path": str(root)}]}
    client.failed = False
    client.process = SimpleNamespace(pid=receipt["pid"])
    client.pending = {1: {}}
    client.submissions = [dict(job_id=1, ownership="SEALED_KERNEL_TRANSFER_NOT_DURABLE",
                               identity=capture, files=rows)]
    client.completions = []
    return client, receipt


class LiveReceiptTests(unittest.TestCase):
    def test_actual_archive_child_complete_receipt_and_sidecar(self):
        archive, ledger = scored_archive()
        with tempfile.TemporaryDirectory(prefix="live-receipt-child-", dir="/tmp") as temporary:
            root = Path(temporary) / "archive"
            sidecars = Path(temporary) / "sidecars"
            root.mkdir()
            sidecars.mkdir()
            client = ArchiveClient(roots=[root], max_jobs=2, max_bytes=128 * 1024 * 1024,
                                   max_frames=1, binding={"run_binding": "test-live-receipt-child"})
            try:
                capture = C._parse(archive.ready_identity_bytes)["capture_identity"]
                files = R.archive_files_for_submit(client, archive,
                    relative_prefix=PREFIX, root_index=0)
                job = client.submit(capture, files)
                submitted = R.observe_archive_submission(client, job, archive,
                    relative_prefix=PREFIX, root_index=0)
                deadline = time.monotonic() + 30
                while True:
                    client.poll()
                    matches = [row for row in client.completions if row["job_id"] == job]
                    if matches:
                        self.assertEqual(len(matches), 1)
                        receipt = matches[0]
                        break
                    self.assertLess(time.monotonic(), deadline, "archive child completion timeout")
                    select.select([client.socket], [], [], .1)
                self.assertEqual(receipt["status"], "COMPLETE")
                completed = R.observe_durable_completion(client, receipt, archive,
                    relative_prefix=PREFIX, root_index=0)
                confirmation = scorer_completion(archive, ledger, receipt, submitted, completed)
                proof = R.persist_live_completion(client, receipt, archive, confirmation,
                    submitted, completed, relative_prefix=PREFIX, root_index=0,
                    sidecar_directory=sidecars)
                self.assertEqual((sidecars / R.SIDECAR_NAME).read_bytes(), proof["readback_bytes"])
                self.assertEqual(proof["receipt_sha256"], C.sha256(C.canonical(receipt)))
                with self.assertRaises(ValueError):
                    R.persist_live_completion(client, receipt, archive, confirmation,
                        submitted, completed, relative_prefix=PREFIX, root_index=0,
                        sidecar_directory=sidecars)
                client.close()
            finally:
                if not client.closed:
                    client.abort()

    def test_exact_complete_receipt_and_create_only_postreceipt_sidecar(self):
        archive, ledger = scored_archive()
        with tempfile.TemporaryDirectory(prefix="live-receipt-", dir="/tmp") as temporary:
            root = Path(temporary) / "archive"
            sidecars = Path(temporary) / "sidecars"
            root.mkdir()
            sidecars.mkdir()
            client, receipt = modeled_completion(root, archive)
            files = R.archive_files_for_submit(client, archive, relative_prefix=PREFIX, root_index=0)
            self.assertEqual(len(files), len(archive.files))
            submitted = R.observe_archive_submission(client, 1, archive,
                relative_prefix=PREFIX, root_index=0)
            self.assertEqual(submitted["pid"], os.getpid())
            client.pending.clear()
            client.completions.append(receipt)
            verified = R.verify_live_receipt(archive, receipt,
                relative_prefix=PREFIX, root_index=0, archive_root=root)
            self.assertEqual(set(verified["files"]), {PREFIX + name for name in archive.files})
            self.assertEqual(verified["archive_manifest_sha256"], archive.manifest_sha256)
            completed = R.observe_durable_completion(client, receipt, archive,
                relative_prefix=PREFIX, root_index=0)
            confirmation = scorer_completion(archive, ledger, receipt, submitted, completed)
            written = R.persist_live_completion(client, receipt, archive, confirmation,
                submitted, completed, relative_prefix=PREFIX, root_index=0,
                sidecar_directory=sidecars)
            sidecar = sidecars / R.SIDECAR_NAME
            raw = sidecar.read_bytes()
            self.assertEqual(len(raw), written["bytes"])
            self.assertEqual(C.sha256(raw), written["sha256"])
            self.assertEqual(raw, written["readback_bytes"])
            self.assertEqual(written["schema"], R.PROOF_SCHEMA)
            self.assertEqual(written["clock_completion_bytes"],
                             len(confirmation["clock_completion_bytes"]))
            self.assertTrue(all(written[name] for name in
                ("created_exclusive", "file_fsynced", "directory_fsynced", "readback_verified")))
            document = json.loads(raw)
            self.assertEqual(document["schema"], R.SCHEMA)
            self.assertEqual(document["archive_receipt"], receipt)
            self.assertEqual(document["clock_completion"]["schema"], CLOCK.COMPLETION_SCHEMA)
            self.assertEqual(document["clock_completion"]["archive_scope"], "SCORER_NUMERIC_ARCHIVE")
            self.assertFalse(any(item["ready"] for item in receipt["written"]))
            with self.assertRaises(ValueError):
                R.persist_live_completion(client, receipt, archive, confirmation,
                    submitted, completed, relative_prefix=PREFIX, root_index=0,
                    sidecar_directory=sidecars)
            self.assertEqual(sidecar.read_bytes(), raw)

    def test_tampering_traversal_symlinks_and_foreign_capture_fail_closed(self):
        archive, ledger = scored_archive()
        with tempfile.TemporaryDirectory(prefix="live-receipt-negative-", dir="/tmp") as temporary:
            root = Path(temporary) / "archive"
            sidecars = Path(temporary) / "sidecars"
            root.mkdir()
            sidecars.mkdir()
            client, receipt = modeled_completion(root, archive)
            with self.assertRaises(ValueError):
                R.observe_archive_submission(client, 1, archive,
                    relative_prefix="../live/", root_index=0)
            submitted = R.observe_archive_submission(client, 1, archive,
                relative_prefix=PREFIX, root_index=0)
            client.pending.clear()
            client.completions.append(receipt)
            completed = R.observe_durable_completion(client, receipt, archive,
                relative_prefix=PREFIX, root_index=0)
            confirmation = scorer_completion(archive, ledger, receipt, submitted, completed)

            variants = {}
            foreign = copy.deepcopy(receipt)
            foreign["identity"]["frame_id"] += 1
            variants["foreign capture"] = foreign
            missing = copy.deepcopy(receipt)
            missing["written"].pop()
            variants["missing file"] = missing
            extra = copy.deepcopy(receipt)
            extra["written"].append(dict(extra["written"][0], relative=PREFIX + "extra.json"))
            variants["extra file"] = extra
            duplicate = copy.deepcopy(receipt)
            duplicate["written"].append(dict(duplicate["written"][0]))
            variants["duplicate file"] = duplicate
            wrong_hash = copy.deepcopy(receipt)
            wrong_hash["written"][0]["sha256"] = "0" * 64
            variants["wrong hash"] = wrong_hash
            wrong_size = copy.deepcopy(receipt)
            wrong_size["written"][0]["bytes"] += 1
            variants["wrong size"] = wrong_size
            wrong_root = copy.deepcopy(receipt)
            wrong_root["written"][0]["root"] = 1
            variants["wrong root"] = wrong_root
            ready = copy.deepcopy(receipt)
            ready["written"][0]["ready"] = True
            variants["false READY"] = ready
            traversal = copy.deepcopy(receipt)
            traversal["written"][0]["relative"] = PREFIX + "../escape.json"
            variants["path traversal"] = traversal
            for name, changed in variants.items():
                with self.subTest(name=name), self.assertRaises(ValueError):
                    R.verify_live_receipt(archive, changed, relative_prefix=PREFIX,
                                          root_index=0, archive_root=root)

            alias = root / "alias"
            alias.symlink_to(root / "frame_10", target_is_directory=True)
            aliased = copy.deepcopy(receipt)
            for row in aliased["written"]:
                row["relative"] = row["relative"].replace("frame_10/", "alias/", 1)
            with self.assertRaises(OSError):
                R.verify_live_receipt(archive, aliased, relative_prefix="alias/live/",
                                      root_index=0, archive_root=root)
            linked_sidecar_dir = Path(temporary) / "linked_sidecars"
            linked_sidecar_dir.symlink_to(sidecars, target_is_directory=True)
            with self.assertRaises(OSError):
                R.persist_live_completion(client, receipt, archive, confirmation,
                    submitted, completed, relative_prefix=PREFIX, root_index=0,
                    sidecar_directory=linked_sidecar_dir)
            self.assertFalse((sidecars / R.SIDECAR_NAME).exists())


if __name__ == "__main__":
    unittest.main()
