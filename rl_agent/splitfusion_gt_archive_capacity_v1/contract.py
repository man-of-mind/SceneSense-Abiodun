"""Identity of the GT-archive capacity repair v1 (narrow, in place, registered).

Defect: ``splitfusion_live_gt_handoff_v1/archive.py`` capped ``max_frames`` at 1,000, while the exact-GT v2 service
and scorer size the archive at 4 x the registered decision budget (one producer and one scorer job per decision,
with 2x headroom). A 1,000-frame registration therefore failed at service READY (``archive frame bound differs``).

Repair: ``max_frames`` is documented and enforced as the cumulative submission (job) bound of one client / attempt,
finite at ``MAX_CUMULATIVE_JOBS`` = 4,000 = 4 x the largest registered budget. Outstanding job credits (2..8),
outstanding byte credits, back-pressure, duplicate-identity/path refusal, durable completion and teardown are
unchanged. Cumulative bookkeeping (retained submission metadata + completion records) is accounted separately
(``bookkeeping_bytes``) and bounded by ``MAX_BOOKKEEPING_BYTES``. The module name is unchanged on purpose: the
retained process-scope observers identify the archive child by exactly that module.

Changed files: exactly ``ARCHIVE`` and ``TEST_ARCHIVE`` (the old 1,001-refusal case); originals preserved
byte-exact under ``EVIDENCE``/original.
"""
import hashlib
from pathlib import Path

ARCHIVE = "rl_agent/splitfusion_live_gt_handoff_v1/archive.py"
TEST_ARCHIVE = "rl_agent/splitfusion_live_gt_handoff_v1/test_archive.py"
ORIGINAL = {ARCHIVE: "81abd840647f28b06e43d5b9459456b0614c27fe250f827efe76ab81475446dc",
            TEST_ARCHIVE: "3723493f40dd44db9482fcdc861146f6597a69e9ad44363fbfb8979bcdc62127"}
REPAIRED = {ARCHIVE: "98308f84c9154fbecb5bfba65dac53a26df56a455c6d3a1bdcf3919dbba9efa5", TEST_ARCHIVE: "cc9a34c316c736c4cd0f1ccef4bca8ff9777c3547b1c36cd3acc8c9190bab8d5"}
MAX_CUMULATIVE_JOBS = 4000
LARGEST_REGISTERED_BUDGET = 1000
JOBS_PER_DECISION_SIZING = 4
EVIDENCE = "experiments/splitfusion_gt_archive_capacity_v1/20261006_v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def repaired(root):
    """Current digests of the two changed files, refusing anything but the registered repair."""
    root = Path(root).resolve(strict=True)
    for relative, digest in REPAIRED.items():
        path = root / relative
        if not path.is_file() or path.is_symlink() or sha(path) != digest:
            raise ValueError("archive repair source differs from the registered repair: " + relative)
    return dict(REPAIRED)


def substitution_ok(proof_sources, current_sources, root):
    """True iff a retained proof binds exactly the current sources except the registered original->repaired files."""
    if proof_sources == current_sources:
        return True
    expected = repaired(root)
    if set(proof_sources) != set(current_sources):
        return False
    differing = {k for k in current_sources if proof_sources[k] != current_sources[k]}
    return (differing == set(ORIGINAL) and all(proof_sources[k] == ORIGINAL[k] for k in ORIGINAL)
            and all(current_sources[k] == expected[k] for k in ORIGINAL))
