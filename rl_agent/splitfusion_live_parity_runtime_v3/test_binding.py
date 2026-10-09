"""New entry/contract wiring and unchanged scientific settings."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import contract, container_check, entries, startup
from rl_agent.splitfusion_live_parity_runtime_v2 import contract as previous


class BindingTests(unittest.TestCase):
    def test_frozen_scientific_contract_identical(self):
        self.assertEqual(contract.metrics(), previous.metrics())
        self.assertEqual(contract.BUDGET, 100)
        self.assertNotEqual(contract.ID, previous.ID)
        self.assertNotEqual(contract.EDGE, previous.EDGE)
        self.assertNotEqual(contract.CHILD, previous.CHILD)

    def test_bootstrap_dispatches_v3_not_v2(self):
        args = ["--ready-file", "/work/torch_cache/ready.json"]
        writes = []
        with patch.dict(startup.edge_entry.__globals__, write_x=lambda p, v: writes.append((p, v))), \
                patch.object(entries, "edge", return_value=0) as entry:
            self.assertEqual(startup.edge_entry(args), 0)
        entry.assert_called_once_with(args)
        self.assertEqual([p.name for p, _ in writes], ["STARTED.json", "EXITED.json"])

    def test_bootstrap_preserves_exact_failure(self):
        failure = ValueError("actual v3 failure")
        writes = []
        with patch.dict(startup.edge_entry.__globals__, write_x=lambda p, v: writes.append((p, v))), \
                patch.object(entries, "edge", side_effect=failure):
            with self.assertRaises(ValueError) as caught:
                startup.edge_entry(["--ready-file", "/work/torch_cache/ready.json"])
        self.assertIs(caught.exception, failure)
        self.assertEqual(writes[-1][0].name, "FAILED.json")

    def test_registration_is_diagnostic_only_and_v3(self):
        record = container_check.synthetic_registration()
        self.assertEqual(record["edge_module"], contract.EDGE)
        self.assertEqual(record["id"], contract.ID)
        self.assertTrue(record["diagnostic_only"])
        self.assertFalse(record["live_authorization"])

    def test_actual_native_fixture_not_model_grid(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            ticket, expected = container_check._ticket_and_target(Path(directory))
        self.assertEqual(ticket.predicted_mask.shape, (720, 1280))
        self.assertEqual(expected.measurement["loc_vehicle_tp"], 1)
        self.assertEqual(expected.measurement["loc_person_tp"], 1)
        self.assertFalse(torch.cuda.is_initialized())

    def test_container_command_is_nonlive_no_gpu_and_owned(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            for name in ("repo", "binding", "code", "output"):
                (root / name).mkdir()
            cmd = container_check.docker_command(repo_root=root / "repo",
                binding_directory=root / "binding", partition_code=root / "code", scratch_directory=root / "output")
        self.assertEqual(cmd[cmd.index("--network") + 1], "none")
        self.assertIn("--user", cmd)
        self.assertNotIn("--gpus", cmd)
        self.assertIn("rl_agent.splitfusion_live_parity_runtime_v3.container_check", cmd)


if __name__ == "__main__":
    unittest.main()
