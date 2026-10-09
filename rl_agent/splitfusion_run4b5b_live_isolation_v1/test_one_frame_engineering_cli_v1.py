"""Regression test for the canonical one-frame package entrypoint."""

from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from . import one_frame_engineering_cli_v1 as CLI
from . import one_frame_engineering_v1 as O
from . import final_actor_gate_v2 as F


class CanonicalEntrypointTest(unittest.TestCase):
    def test_preflight_factory_receives_canonical_config_type(self) -> None:
        observed = []

        class Lifecycle:
            def preflight(self, config, actor):
                observed.append((type(config), actor))
            def start(self, config, actor): raise AssertionError("start")
            def execute(self, config, actor): raise AssertionError("execute")
            def stop(self, config): raise AssertionError("stop")

        actor = SimpleNamespace(identity=SimpleNamespace(
            actor_state_dict_sha256="a" * 64))
        canonical = object.__new__(O.OneFrameConfigV1)
        object.__setattr__(canonical, "variant", F.RUN4B_VARIANT)
        # The wrapper must delegate the canonical implementation function,
        # never execute a second copy of the schema module as __main__.
        self.assertIs(CLI.main, O.main)
        with mock.patch.object(O, "load_config", return_value=canonical), \
                mock.patch.object(O.OneFrameConfigV1, "binding_sha256",
                                  return_value="b" * 64), \
                mock.patch.object(O, "load_production_lifecycle",
                                  side_effect=lambda cfg: (
                                      self.assertIs(type(cfg), O.OneFrameConfigV1)
                                      or Lifecycle())), \
                mock.patch.object(O, "preflight", return_value=actor):
            self.assertEqual(CLI.main([
                "--config", "/not-read.json", "--preflight"]), 0)
        # load_production_lifecycle is the factory seam; its assertion above
        # proves exact canonical class identity at that boundary.


if __name__ == "__main__":
    unittest.main()
