"""CPU-only runtime, v2 regression, and reused durable-archive test record."""
import argparse
import io
import json
from pathlib import Path
import unittest

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
from rl_agent.splitfusion_numeric_gt_integration_v2 import replay as H
from rl_agent.splitfusion_numeric_gt_producer_v1 import originals as O
from .replay import inventory, OUTPUT_ROOT, PACKAGE


def run(output):
    sources = inventory(dict(startup=O.startup_binding()))
    output = Path(output).absolute()
    C.require(not output.exists() and output.resolve().is_relative_to(OUTPUT_ROOT), "fresh test-result directory")
    output.mkdir(parents=True, exist_ok=False)
    H.write(output / "PLAN.json", dict(source_sha256=sources, live_launch=False,
        suites=[str(PACKAGE), "splitfusion_numeric_gt_integration_v2", "splitfusion_live_gt_handoff_v1.test_archive"]))
    suite = unittest.TestSuite()
    for package in (PACKAGE, H.PACKAGE):
        suite.addTests(unittest.defaultTestLoader.discover(str(package), pattern="test_*.py", top_level_dir=str(H.ROOT)))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromName("rl_agent.splitfusion_live_gt_handoff_v1.test_archive"))
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    (output / "TEST_LOG.txt").write_text(log.getvalue())
    unchanged = inventory(dict(startup=O.startup_binding())) == sources
    H.write(output / "RESULT.json", dict(status="PASS" if result.wasSuccessful() and unchanged else "FAIL",
        tests=result.testsRun, failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
        source_files_unchanged=unchanged, live_services_started=False))
    H.write(output / "OUTPUT_SHA256.json", {p.name: H.sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    print(json.dumps(dict(tests=result.testsRun, successful=result.wasSuccessful(), sources_unchanged=unchanged)))
    C.require(result.wasSuccessful() and unchanged, "runtime correctness gate failed; see preserved test log")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
