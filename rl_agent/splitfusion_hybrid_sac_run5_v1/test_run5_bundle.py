"""Fault-injection tests for crash-consistent Run-5 bundles (tiny payloads, CPU only)."""

from __future__ import annotations

import errno
import json
import os
import tempfile
import unittest
from pathlib import Path

import torch

from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_bundle as B

PAYLOADS = {
    "event.json": B.canonical_bytes({"schema": "event", "update_count": 5}),
    "training_state.pt": B.torch_bytes({"online_critics": {"w": torch.ones(3)},
                                        "target_critics": {"w": torch.zeros(3)},
                                        "actor_optimizer": {"state": {}},
                                        "critic_optimizer": {"state": {}},
                                        "generators": {"replay": torch.Generator().get_state()}}),
    "actor_state_dict.pt": B.torch_bytes({"encoder.0.weight": torch.arange(6.0)}),
    "channel_state.json": B.canonical_bytes({"tick": 12}),
}


class _Crash(Exception):
    pass


class FaultIO(B.BundleIO):
    """Raise at the n-th occurrence of one operation (optionally as ENOSPC)."""

    def __init__(self, operation: str, occurrence: int, enospc: bool = False) -> None:
        self.operation, self.occurrence, self.enospc = operation, occurrence, enospc
        self.seen: dict[str, int] = {}

    def _tick(self, name: str) -> None:
        self.seen[name] = self.seen.get(name, 0) + 1
        if name == self.operation and self.seen[name] == self.occurrence:
            if self.enospc:
                raise OSError(errno.ENOSPC, "No space left on device (simulated)")
            raise _Crash(f"{name}#{self.occurrence}")

    def mkdir(self, path):
        self._tick("mkdir")
        super().mkdir(path)

    def write(self, stream, data):
        self._tick("write")
        super().write(stream, data)

    def fsync(self, descriptor):
        self._tick("fsync")
        super().fsync(descriptor)

    def rename(self, source, target):
        self._tick("rename")
        super().rename(source, target)

    def replace(self, source, target):
        self._tick("replace")
        super().replace(source, target)


class CountIO(B.BundleIO):
    def __init__(self):
        self.seen: dict[str, int] = {}

    def __getattribute__(self, name):
        attribute = super().__getattribute__(name)
        if name in ("mkdir", "write", "fsync", "rename", "replace") and callable(attribute):
            def counted(*args, **kwargs):
                self.seen[name] = self.seen.get(name, 0) + 1
                return attribute(*args, **kwargs)
            return counted
        return attribute


def manifest(update=5):
    return {"seed": 17, "update_count": update}


class BundleTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def publish(self, name="checkpoint_000005", **kwargs):
        return B.publish_bundle(self.root, name, PAYLOADS, manifest(), **kwargs)

    def test_publication_commits_all_files_marker_and_latest(self) -> None:
        digest = self.publish()
        bundle = B.verify_bundle(self.root / "checkpoint_000005")
        self.assertEqual(bundle.manifest_sha256, digest)
        self.assertEqual(set(bundle.manifest["files"]), set(PAYLOADS))
        self.assertEqual((self.root / "checkpoint_000005" / B.COMMITTED).read_text().strip(), digest)
        pointer = json.loads((self.root / B.LATEST).read_text())
        self.assertEqual(pointer, {"name": "checkpoint_000005", "manifest_sha256": digest})
        selection = B.select_resume(self.root)
        self.assertEqual((selection.bundle.name, selection.latest_status),
                         ("checkpoint_000005", "LATEST_CURRENT"))
        with self.assertRaises(B.BundleError):
            self.publish()
        self.assertEqual(B.torch_from_bytes(bundle.payload("actor_state_dict.pt"))
                         ["encoder.0.weight"].tolist(), list(range(6)))

    def test_fault_before_and_after_every_write_fsync_and_rename(self) -> None:
        counter = CountIO()
        B.publish_bundle(self.root, "checkpoint_000001",
                         PAYLOADS, manifest(1), bundle_io=counter)
        totals = dict(counter.seen)
        self.assertGreaterEqual(totals["write"], 6)
        self.assertGreaterEqual(totals["fsync"], 8)
        cases = [(op, n, enospc) for op, count in totals.items() for n in range(1, count + 1)
                 for enospc in ((False, True) if op == "write" else (False,))]
        for operation, occurrence, enospc in cases:
            with self.subTest(operation=operation, occurrence=occurrence, enospc=enospc):
                name = "checkpoint_000009"
                faults = FaultIO(operation, occurrence, enospc)
                with self.assertRaises((_Crash, OSError)):
                    B.publish_bundle(self.root, name, PAYLOADS, manifest(9), bundle_io=faults)
                final = self.root / name
                pointer = json.loads((self.root / B.LATEST).read_text())
                if final.exists():
                    # Only possible after the atomic rename: it must be complete.
                    B.verify_bundle(final)
                    selection = B.select_resume(self.root)
                    self.assertEqual(selection.bundle.name, name)
                    self.assertIn(pointer["name"], ("checkpoint_000001", name))
                    import shutil
                    shutil.rmtree(final)
                    B.write_latest(self.root, "checkpoint_000001",
                                   B.verify_bundle(self.root / "checkpoint_000001").manifest_sha256)
                else:
                    self.assertEqual(pointer["name"], "checkpoint_000001")
                    self.assertEqual(B.select_resume(self.root).bundle.name, "checkpoint_000001")
                self.assertFalse(any(p.name.startswith(B.STAGING_PREFIX)
                                     for p in self.root.iterdir()))

    def test_crash_between_event_and_weights_leaves_no_final_bundle(self) -> None:
        # payloads are written in sorted order: actor, channel, event, training_state.
        faults = FaultIO("write", 4)
        with self.assertRaises(_Crash):
            self.publish(bundle_io=faults)
        self.assertFalse((self.root / "checkpoint_000005").exists())
        self.assertIsNone(B.select_resume(self.root).bundle)

    def test_incomplete_staging_directory_is_ignored(self) -> None:
        self.publish()
        staging = self.root / f"{B.STAGING_PREFIX}checkpoint_000010-1-abc"
        staging.mkdir()
        (staging / "event.json").write_bytes(b"{}")
        selection = B.select_resume(self.root)
        self.assertEqual(selection.bundle.name, "checkpoint_000005")
        self.assertIn(staging.name, selection.ignored_staging)

    def test_missing_or_corrupt_members_are_refused(self) -> None:
        for member in (*PAYLOADS, B.MANIFEST, B.COMMITTED):
            for damage in ("delete", "flip"):
                with self.subTest(member=member, damage=damage):
                    with tempfile.TemporaryDirectory() as tmp:
                        root = Path(tmp)
                        B.publish_bundle(root, "checkpoint_000005", PAYLOADS, manifest())
                        path = root / "checkpoint_000005" / member
                        if damage == "delete":
                            path.unlink()
                        else:
                            data = bytearray(path.read_bytes())
                            data[len(data) // 2] ^= 0x01
                            path.write_bytes(bytes(data))
                        with self.assertRaises(B.BundleCorrupt):
                            B.verify_bundle(root / "checkpoint_000005")
                        with self.assertRaises(B.BundleCorrupt):
                            B.select_resume(root)

    def test_extra_member_is_refused(self) -> None:
        self.publish()
        (self.root / "checkpoint_000005" / "stray.pt").write_bytes(b"x")
        with self.assertRaises(B.BundleCorrupt):
            B.select_resume(self.root)

    def test_stale_latest_pointer_policies(self) -> None:
        first = self.publish("checkpoint_000005")
        B.publish_bundle(self.root, "checkpoint_000007", PAYLOADS, manifest(7),
                         update_latest=False)            # crash before pointer update
        selection = B.select_resume(self.root)
        self.assertEqual(selection.bundle.name, "checkpoint_000007")
        self.assertTrue(selection.latest_status.startswith("LATEST_LAGGED_REPAIRED"))
        self.assertEqual(json.loads((self.root / B.LATEST).read_text())["name"],
                         "checkpoint_000007")
        B.write_latest(self.root, "checkpoint_000005", first)
        with self.assertRaises(B.BundleCorrupt):
            B.select_resume(self.root, repair_lag=False)
        for pointer in ({"name": "checkpoint_000099", "manifest_sha256": first},
                        {"name": "checkpoint_000005", "manifest_sha256": "0" * 64}):
            (self.root / B.LATEST).write_bytes(B.canonical_bytes(pointer))
            with self.assertRaises(B.BundleCorrupt):
                B.select_resume(self.root)
        (self.root / B.LATEST).write_bytes(b"not json")
        with self.assertRaises(B.BundleCorrupt):
            B.select_resume(self.root)
        (self.root / B.LATEST).unlink()
        with self.assertRaises(B.BundleCorrupt):
            B.select_resume(self.root)

    def test_unregistered_names_and_symlinks_are_refused(self) -> None:
        with self.assertRaises(B.BundleError):
            B.publish_bundle(self.root, "checkpoint_5", PAYLOADS, manifest())
        self.publish()
        os.symlink(self.root / "checkpoint_000005", self.root / "checkpoint_000006")
        with self.assertRaises(B.BundleCorrupt):
            B.select_resume(self.root)

    def test_append_only_ledger_is_resume_safe(self) -> None:
        path = self.root / "metrics.jsonl"
        ledger = B.AppendOnlyJsonl(path)
        for index in range(5):
            ledger.record(index, {"loss": index * 0.5})
        prefix = ledger.prefix(3)
        with open(path, "ab") as stream:
            stream.write(b'{"index":5,"lo')                # torn write at crash
        resumed = B.AppendOnlyJsonl(path)
        self.assertEqual(len(resumed), 5)
        resumed.require_prefix(prefix)
        resumed.record(3, {"loss": 1.5})                   # recomputed, verified, not duplicated
        with self.assertRaises(B.BundleCorrupt):
            resumed.record(4, {"loss": 99.0})
        with self.assertRaises(B.BundleError):
            resumed.record(7, {"loss": 1.0})
        resumed.record(5, {"loss": 2.5})
        self.assertEqual(len(path.read_bytes().splitlines()), 6)
        tampered = bytearray(path.read_bytes())
        tampered[5] ^= 1
        path.write_bytes(bytes(tampered))
        with self.assertRaises(B.BundleCorrupt):
            B.AppendOnlyJsonl(path).require_prefix(prefix)


if __name__ == "__main__":
    unittest.main()
