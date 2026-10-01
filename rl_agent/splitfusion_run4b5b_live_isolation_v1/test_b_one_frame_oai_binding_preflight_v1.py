"""Focused mocked tests for the one-frame local OAI read-only-bind preflight.

``RealProductionOpsV1._require_local_oai_binding`` is exercised against a
temporary authority/repository pair.  Kernel mount state, inode identity and
git objects are substituted; the real source-pin hash verifier runs over
substituted pins.  No mount, service, network, CUDA or OAI operation occurs.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from rl_agent import splitfusion_phase14a_100mhz_calibration_v1 as PHASE14A
from rl_agent.splitfusion_run4_split_host_l10319_v1 import local_ran_lifecycle_v1 as LR
from rl_agent.splitfusion_run4b5b_live_isolation_v1 import (
    b_one_frame_production_factory_v1 as FACTORY,
)

OPS = FACTORY.RealProductionOpsV1
COMMIT = FACTORY.OAI_GITLINK_COMMIT
EXECUTABLES = ("nr_softmodem", "nr_uesoftmodem", "tracer_multi", "tracer_record")
REAL_STAT = Path.stat
REAL_READ_TEXT = Path.read_text


class OaiBindingPreflightTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        base = Path(temporary.name).resolve()
        self.authority = base / "abiodun" / "OAI" / "openairinterface5g"
        self.authority.mkdir(parents=True)
        self.repository = base / "abiodun_run4b5b_runtime_v1"
        self.target = self.repository / "OAI" / "openairinterface5g"
        self.target.mkdir(parents=True)
        self.radio_base = base / "radio_state"
        self.radio_base.mkdir()
        self.config = SimpleNamespace(local_repository=self.repository,
                                      run_id="run4b_oneframe_eng_test")
        pins = []
        for pin in LR.SOURCE_PINS:
            path = self.repository / pin.relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"fixture {pin.name}".encode())
            if pin.name in EXECUTABLES:
                path.chmod(0o755)
            pins.append(replace(pin, sha256=hashlib.sha256(
                path.read_bytes()).hexdigest()))
        self.pins = tuple(pins)
        self.inode = {self.authority: (7, 11), self.target: (7, 11)}
        self.options = "ro,relatime"
        self.source = str(self.authority)
        self.mount_targets = [str(self.target)]
        self.objects = {(self.authority, "HEAD"): COMMIT,
                        (self.repository, "HEAD:OAI/openairinterface5g"): COMMIT}
        self.reconcile = {"status": "PHASE14A_CPU_RECONCILIATION_PASSED"}
        self.reconcile_calls = []

    # -- substituted kernel/git state ----------------------------------
    @staticmethod
    def _unbound(function):
        """Plain function so Path binds the instance as the first argument."""
        def method(path, *args, **kwargs):
            return function(path, *args, **kwargs)
        return method

    def _mountinfo(self) -> str:
        lines = ["21 1 259:2 / / rw,relatime shared:1 - ext4 /dev/root rw"]
        for target in self.mount_targets:
            lines.append(f"53 30 259:2 {self.source} {target} {self.options} "
                         "shared:1 - ext4 /dev/nvme0n1p2 rw,errors=remount-ro")
        return "\n".join(lines) + "\n"

    def _stat(self, path, *args, **kwargs):
        value = REAL_STAT(path, *args, **kwargs)
        key = Path(path)
        if key in self.inode:
            dev, ino = self.inode[key]
            return SimpleNamespace(st_dev=dev, st_ino=ino,
                                   st_mode=value.st_mode)
        return value

    def _read_text(self, path, *args, **kwargs):
        if str(path) == "/proc/self/mountinfo":
            return self._mountinfo()
        return REAL_READ_TEXT(path, *args, **kwargs)

    def _git(self, repository, name):
        try:
            return self.objects[(Path(repository), name)]
        except KeyError:
            raise FACTORY.ProductionOneFrameError(
                f"cannot resolve git object {name} in {repository}") from None

    def _reconcile_contract(self, config_path, binding_path):
        self.reconcile_calls.append((Path(config_path), Path(binding_path)))
        return dict(self.reconcile)

    def run_preflight(self) -> None:
        real_verify = LR.verify_source_pins
        with mock.patch.object(FACTORY, "OAI_AUTHORITY_ROOT", self.authority), \
                mock.patch.object(FACTORY.OLD, "LOCAL_RADIO_STATE_BASE",
                                  self.radio_base), \
                mock.patch.object(LR, "SOURCE_PINS", self.pins), \
                mock.patch.object(LR, "verify_source_pins",
                                  lambda root: real_verify(root, pins=self.pins)), \
                mock.patch.object(OPS, "_git_object",
                                  staticmethod(self._git)), \
                mock.patch.object(PHASE14A, "reconcile_contract",
                                  self._reconcile_contract), \
                mock.patch.object(Path, "stat", self._unbound(self._stat)), \
                mock.patch.object(Path, "read_text", self._unbound(self._read_text)):
            OPS._require_local_oai_binding(self.config)

    def refused(self, pattern: str) -> None:
        with self.assertRaisesRegex(FACTORY.ProductionOneFrameError, pattern):
            self.run_preflight()

    # -- acceptance ------------------------------------------------------
    def test_exact_read_only_bind_with_all_pins_passes(self) -> None:
        self.run_preflight()
        self.assertEqual(self.reconcile_calls, [(
            self.repository /
            "rl_agent/configs/splitfusion_phase14a_100mhz_calibration_v1.json",
            self.repository /
            "rl_agent/configs/splitfusion_phase14a_campaign_binding_v1.json")])

    def test_escaped_mountinfo_paths_are_decoded(self) -> None:
        self.assertEqual(OPS._mountinfo_path(r"/a\040b\011c\134d"), "/a b\tc\\d")

    # -- mount target / source / options --------------------------------
    def test_missing_mount_record_for_exact_target_is_refused(self) -> None:
        self.mount_targets = [str(self.target) + "_other"]
        self.refused("not one exact mountpoint")

    def test_stacked_duplicate_mount_records_are_refused(self) -> None:
        self.mount_targets = [str(self.target), str(self.target)]
        self.refused("not one exact mountpoint")

    def test_foreign_bind_source_is_refused(self) -> None:
        self.source = str(self.authority.parent / "other_oai")
        self.refused("bind source differs from authority")

    def test_writable_bind_is_refused(self) -> None:
        self.options = "rw,relatime"
        self.refused("not read-only")

    def test_option_substring_is_not_accepted_as_read_only(self) -> None:
        self.options = "rw,relatime,errors=remount-ro"
        self.refused("not read-only")

    # -- identity -------------------------------------------------------
    def test_target_with_foreign_inode_is_refused(self) -> None:
        self.inode[self.target] = (7, 12)
        self.refused("not the authority inode")

    def test_target_on_foreign_device_is_refused(self) -> None:
        self.inode[self.target] = (8, 11)
        self.refused("not the authority inode")

    def test_symlink_only_binding_is_refused(self) -> None:
        shutil.rmtree(self.target)
        self.target.symlink_to(self.authority, target_is_directory=True)
        self.refused("resolves away from the moved repository")

    def test_missing_target_is_refused(self) -> None:
        shutil.rmtree(self.target)
        with self.assertRaises(FileNotFoundError):
            self.run_preflight()

    def test_missing_authority_is_refused(self) -> None:
        self.authority.rmdir()
        with self.assertRaises(FileNotFoundError):
            self.run_preflight()

    # -- git pins -------------------------------------------------------
    def test_oai_head_drift_is_refused(self) -> None:
        self.objects[(self.authority, "HEAD")] = "0" * 40
        self.refused("OAI authority commit differs")

    def test_parent_gitlink_drift_is_refused(self) -> None:
        self.objects[(self.repository, "HEAD:OAI/openairinterface5g")] = "1" * 40
        self.refused("moved-worktree OAI gitlink differs")

    def test_unresolvable_gitlink_is_refused(self) -> None:
        del self.objects[(self.repository, "HEAD:OAI/openairinterface5g")]
        self.refused("cannot resolve git object")

    # -- source pins ----------------------------------------------------
    def _pin_path(self, name: str) -> Path:
        pin = next(item for item in self.pins if item.name == name)
        return self.repository / pin.relative_path

    def test_every_missing_pin_is_refused(self) -> None:
        for pin in self.pins:
            with self.subTest(pin=pin.name):
                path = self.repository / pin.relative_path
                data, mode = path.read_bytes(), path.stat().st_mode
                path.unlink()
                try:
                    with self.assertRaisesRegex(
                            LR.LocalRanLifecycleError,
                            f"source authority is missing: {pin.name}"):
                        self.run_preflight()
                finally:
                    path.write_bytes(data)
                    path.chmod(mode)

    def test_every_drifted_pin_is_refused(self) -> None:
        for pin in self.pins:
            with self.subTest(pin=pin.name):
                path = self.repository / pin.relative_path
                data = path.read_bytes()
                path.write_bytes(data + b"drift")
                try:
                    with self.assertRaisesRegex(
                            LR.LocalRanLifecycleError,
                            f"source authority drift: {pin.name}"):
                        self.run_preflight()
                finally:
                    path.write_bytes(data)

    def test_named_pins_cover_softmodems_tracers_config_and_messages(self) -> None:
        names = {pin.name for pin in LR.SOURCE_PINS}
        self.assertTrue({"nr_softmodem", "nr_uesoftmodem", "tracer_multi",
                         "tracer_record", "t_messages", "phase14a_config",
                         "phase14a_binding"} <= names)

    def test_non_executable_softmodem_or_tracer_is_refused(self) -> None:
        for name in EXECUTABLES:
            with self.subTest(pin=name):
                path = self._pin_path(name)
                path.chmod(0o644)
                try:
                    self.refused(f"not executable: {name}")
                finally:
                    path.chmod(0o755)

    def test_incomplete_pin_verification_is_refused(self) -> None:
        real_verify = LR.verify_source_pins
        short = self.pins[:-1]
        with mock.patch.object(LR, "verify_source_pins",
                               lambda root: real_verify(root, pins=short)):
            with mock.patch.object(FACTORY, "OAI_AUTHORITY_ROOT", self.authority), \
                    mock.patch.object(FACTORY.OLD, "LOCAL_RADIO_STATE_BASE",
                                      self.radio_base), \
                    mock.patch.object(LR, "SOURCE_PINS", self.pins), \
                    mock.patch.object(OPS, "_git_object", staticmethod(self._git)), \
                    mock.patch.object(Path, "stat", self._unbound(self._stat)), \
                    mock.patch.object(Path, "read_text", self._unbound(self._read_text)):
                with self.assertRaisesRegex(FACTORY.ProductionOneFrameError,
                                            "source-pin verification is incomplete"):
                    OPS._require_local_oai_binding(self.config)

    # -- Phase14a and create-only radio state ---------------------------
    def test_phase14a_reconcile_failure_is_refused(self) -> None:
        self.reconcile = {"status": "PHASE14A_CPU_RECONCILIATION_FAILED"}
        self.refused("Phase14a CPU reconciliation did not pass")

    def test_existing_radio_state_root_is_refused(self) -> None:
        (self.radio_base / f"split_host_{self.config.run_id}").mkdir()
        self.refused("local radio state is not create-only")

    def test_absent_radio_state_root_is_not_created(self) -> None:
        self.run_preflight()
        self.assertFalse((self.radio_base /
                          f"split_host_{self.config.run_id}").exists())

    def test_preflight_is_wired_before_remote_checks(self) -> None:
        source = Path(FACTORY.__file__).read_text(encoding="utf-8")
        body = source[source.index("    def preflight(self, config"):]
        self.assertLess(body.index("self._require_local_oai_binding(config)"),
                        body.index('self._checked_ssh(("hostname"'))


if __name__ == "__main__":
    unittest.main()
