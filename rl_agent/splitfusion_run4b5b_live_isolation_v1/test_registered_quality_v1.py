"""Offline tests for the hash-pinned registered Q_perc loader."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from rl_agent.splitfusion_hybrid_sac_run4_v1.scientific_basis import (
    QUALITY_SOURCE_FILE_SHA256,
    QUALITY_SOURCE_RELATIVE_PATH,
)
from rl_agent.splitfusion_run4b5b_live_isolation_v1 import (
    registered_quality_v1 as R,
)
from rl_agent.splitfusion_run4b5b_live_isolation_v1.test_postrun_evaluator_v1 import (
    reward_spec,
)


class RegisteredQualityLoaderTest(unittest.TestCase):
    def test_relative_root_is_refused_without_io(self) -> None:
        with self.assertRaisesRegex(R.RegisteredQualityError, "absolute"):
            R.load_registered_quality_spec(Path("relative"))

    def test_exact_pinned_source_and_real_type_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            expected = reward_spec()
            with mock.patch.object(
                    R, "verify_quality_source",
                    return_value=QUALITY_SOURCE_FILE_SHA256) as verify, \
                 mock.patch.object(
                    R, "load_reward_spec", return_value=expected) as load:
                self.assertIs(R.load_registered_quality_spec(root), expected)
                verify.assert_called_once_with(root)
                load.assert_called_once_with(
                    root / QUALITY_SOURCE_RELATIVE_PATH,
                    QUALITY_SOURCE_FILE_SHA256,
                )
            with mock.patch.object(
                    R, "verify_quality_source",
                    return_value=QUALITY_SOURCE_FILE_SHA256), \
                 mock.patch.object(R, "load_reward_spec", return_value=object()):
                with self.assertRaisesRegex(R.RegisteredQualityError, "foreign"):
                    R.load_registered_quality_spec(root)


if __name__ == "__main__":
    unittest.main()
