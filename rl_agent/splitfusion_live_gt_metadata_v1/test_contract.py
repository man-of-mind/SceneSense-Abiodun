"""Reject foreign intervention bindings before any worker/source is admitted."""
import unittest
from unittest.mock import patch
from . import contract as C
from .client import Client
from rl_agent.splitfusion_live_gt_optimization_v4.worker import WorkerError
from rl_agent.splitfusion_live_gt_placement_trial_v1.client import Client as TrialClient


class ContractTests(unittest.TestCase):
    def context(self):
        return dict(bootstrap_sources={"source": "digest"}, environment=C.ENVIRONMENT,
            torch_threads=1, cuda_initialized=False, pid=1234,
            runtime_metadata=dict(schema=C.METADATA_SCHEMA, mode="fast", pid=1234,
                                  validators_removed=False, numerical_changes=False))

    def test_both_explicit_metadata_modes(self):
        for mode in C.METADATA_MODES:
            value = self.context()
            value["runtime_metadata"]["mode"] = mode
            self.assertIs(C.validate_context(value, {"source": "digest"}), value)

    def test_foreign_or_missing_scope_refused(self):
        for key, altered in (("schema", "other"), ("mode", "other"), ("pid", 5678),
                             ("validators_removed", True), ("numerical_changes", True)):
            value = self.context()
            value["runtime_metadata"][key] = altered
            with self.assertRaises(ValueError):
                C.validate_context(value, {"source": "digest"})
        value = self.context()
        del value["runtime_metadata"]
        with self.assertRaises(ValueError):
            C.validate_context(value, {"source": "digest"})

    def test_original_source_and_cuda_guards_retained(self):
        value = self.context()
        with self.assertRaises(ValueError):
            C.validate_context(value, {"other": "digest"})
        value["cuda_initialized"] = True
        with self.assertRaises(ValueError):
            C.validate_context(value, {"source": "digest"})

    def test_client_refuses_modes_before_start(self):
        with patch.object(TrialClient, "__init__") as start:
            for options in (dict(metadata_mode="unknown", mode="p4"),
                            dict(metadata_mode="fast", mode="control"),
                            dict(metadata_mode=None, mode="p4")):
                with self.assertRaises(WorkerError):
                    Client(**options)
            start.assert_not_called()

    def test_source_scope_is_additive(self):
        from rl_agent.splitfusion_live_gt_placement_trial_v1 import contract as P
        current, original = C.sources(), P.sources()
        self.assertTrue(original.items() <= current.items())
        extras = set(current) - set(original)
        self.assertTrue(extras)
        self.assertTrue(all(name.startswith("rl_agent/splitfusion_live_gt_metadata_v1/")
                            and not name.rsplit("/", 1)[-1].startswith("test_")
                            for name in extras))


if __name__ == "__main__":
    unittest.main()
