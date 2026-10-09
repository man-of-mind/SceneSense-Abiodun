"""CPU-only V1/V2 completion receipt version and handoff checks."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import time
import unittest

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C

from . import live_archive as A, live_clock as CLOCK, live_receipt as R
from . import test_live_archive as ARCHIVE_TEST
from .test_live_receipt import (PREFIX, modeled_completion, scored_archive,
                                scorer_completion)


def async_scored_archive():
    ARCHIVE_TEST.LiveArchiveTests.setUpClass()
    _, kwargs = ARCHIVE_TEST.LiveArchiveTests().values()
    target = kwargs["ready_target"]
    sync = ARCHIVE_TEST.source_sync(target,
        time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW) - 500_000_000)
    start = sync["observed_raw_ns"] - 2_000_000
    ledger = CLOCK.LiveClockLedger(target.identity_bytes, target.semantic_sha256,
                                   handoff_id=CLOCK.ASYNC_HANDOFF_ID)
    boot, pid = CLOCK.read_boot_id(), os.getpid()
    events = (("collector", "owned_tick_snapshot", 0),
              ("ue", "source_owned", 100_000),
              ("ue", "action_open", 600_000),
              ("ue", "last_datagram_sent", 800_000),
              ("producer", "source_stage_received", 900_000),
              ("producer", "prepare_start", 1_000_000),
              ("producer", "source_alignment_observed", 2_000_000),
              ("producer", "numeric_ready", 3_000_000),
              ("producer", "ready_published", 4_000_000),
              ("scorer", "target_admitted", 5_000_000),
              ("scorer", "prediction_admitted", 6_000_000),
              ("scorer", "score_start", 7_000_000),
              ("scorer", "score_end", 8_000_000),
              ("coordinator", "feedback_emitted", 9_000_000),
              ("ue", "feedback_received", 10_000_000),
              ("scorer", "archive_build_start", 11_000_000))
    for role, name, offset in events:
        ledger.record(role=role, name=name, raw_ns=start + offset, pid=pid,
                      boot_id=boot, ready_identity_bytes=target.identity_bytes,
                      semantic_sha256=target.semantic_sha256)
    archive = A.build_live_archive(**dict(kwargs, source_synchronization=sync,
                                         clock_ledger=ledger))
    ledger.record(role="scorer", name="archive_build_end",
                  raw_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW), pid=pid,
                  boot_id=boot, ready_identity_bytes=target.identity_bytes,
                  semantic_sha256=target.semantic_sha256)
    return archive, ledger


def changed_confirmation(confirmation, *, schema=None, handoff_id="UNCHANGED"):
    changed = dict(confirmation)
    document = C._parse(changed["clock_completion_bytes"])
    if schema is not None:
        document["schema"] = schema
    if handoff_id == "DELETE":
        document.pop("handoff_id", None)
    elif handoff_id != "UNCHANGED":
        document["handoff_id"] = handoff_id
    changed["clock_completion_bytes"] = C.canonical(document)
    changed["clock_completion_sha256"] = C.sha256(changed["clock_completion_bytes"])
    return changed


class AsyncLiveReceiptTests(unittest.TestCase):
    def exercise(self, archive, ledger, *, expected_schema, expected_handoff):
        with tempfile.TemporaryDirectory(prefix="async-receipt-", dir="/tmp") as temporary:
            root, sidecars = Path(temporary) / "archive", Path(temporary) / "sidecars"
            root.mkdir()
            sidecars.mkdir()
            client, receipt = modeled_completion(root, archive)
            submitted = R.observe_archive_submission(client, 1, archive,
                relative_prefix=PREFIX, root_index=0)
            client.pending.clear()
            client.completions.append(receipt)
            completed = R.observe_durable_completion(client, receipt, archive,
                relative_prefix=PREFIX, root_index=0)
            confirmation = scorer_completion(archive, ledger, receipt, submitted, completed)
            self.assertEqual(C._parse(confirmation["clock_completion_bytes"])["schema"],
                             expected_schema)
            opposite = (CLOCK.COMPLETION_SCHEMA if expected_handoff else
                        CLOCK.ASYNC_COMPLETION_SCHEMA)
            variants = {
                "crossed completion schema": changed_confirmation(confirmation, schema=opposite),
                "foreign handoff": changed_confirmation(confirmation,
                    handoff_id="FOREIGN_ASYNC_HANDOFF_V2"),
            }
            if expected_handoff:
                variants["missing V2 handoff"] = changed_confirmation(confirmation,
                    handoff_id="DELETE")
            else:
                variants["V1 explicit null handoff"] = changed_confirmation(confirmation,
                    handoff_id=None)
            for name, changed in variants.items():
                with self.subTest(name=name), self.assertRaises(R.LiveReceiptError):
                    R.persist_live_completion(client, receipt, archive, changed,
                        submitted, completed, relative_prefix=PREFIX, root_index=0,
                        sidecar_directory=sidecars)
                self.assertFalse((sidecars / R.SIDECAR_NAME).exists())
            proof = R.persist_live_completion(client, receipt, archive, confirmation,
                submitted, completed, relative_prefix=PREFIX, root_index=0,
                sidecar_directory=sidecars)
            self.assertTrue(proof["readback_verified"])
            document = C._parse(proof["readback_bytes"])["clock_completion"]
            self.assertEqual(document["schema"], expected_schema)
            self.assertEqual(document.get("handoff_id"), expected_handoff)

    def test_v1_completion_remains_strict_and_durable(self):
        archive, ledger = scored_archive()
        self.exercise(archive, ledger, expected_schema=CLOCK.COMPLETION_SCHEMA,
                      expected_handoff=None)

    def test_v2_completion_requires_exact_handoff_and_is_durable(self):
        archive, ledger = async_scored_archive()
        self.exercise(archive, ledger, expected_schema=CLOCK.ASYNC_COMPLETION_SCHEMA,
                      expected_handoff=CLOCK.ASYNC_HANDOFF_ID)

    def test_descriptor_crossed_schema_and_handoff_rejected(self):
        archive, _ = async_scored_archive()
        descriptor = C._parse(archive.clock_descriptor_bytes)
        for changed in (dict(descriptor, schema=CLOCK.SCHEMA),
                        dict(descriptor, handoff_id="wrong"),
                        {key: value for key, value in descriptor.items() if key != "handoff_id"}):
            with self.subTest(changed=changed.get("handoff_id")), self.assertRaises(R.LiveReceiptError):
                R._completion_version(type("Descriptor", (),
                    {"clock_descriptor_bytes": C.canonical(changed)})())
        old, _ = scored_archive()
        old_descriptor = C._parse(old.clock_descriptor_bytes)
        for changed in (dict(old_descriptor, schema=CLOCK.ASYNC_SCHEMA),
                        dict(old_descriptor, handoff_id=None),
                        dict(old_descriptor, handoff_id=CLOCK.ASYNC_HANDOFF_ID)):
            with self.subTest(old_handoff=changed.get("handoff_id")), self.assertRaises(R.LiveReceiptError):
                R._completion_version(type("Descriptor", (),
                    {"clock_descriptor_bytes": C.canonical(changed)})())


if __name__ == "__main__":
    unittest.main()
