#!/usr/bin/env python3
"""UE child entry point for the thread-cap experiment (both arms).

Runs the unchanged profiled child (``child_v1.main``) with the same argv.
Before it, using only the standard library:

1. verifies the parent handed this process the registered arm and that the
   arm's variables are present at ``exec`` (so numerical libraries read them at
   initialization);
2. installs the ``subprocess.Popen`` restore so every process this child
   launches gets the baseline values of the cap variables;
3. records the pre-import settings snapshot.

After the child returns (or raises), it records the post-run snapshot with the
effective PyTorch/OpenMP/MKL/OpenCV thread counts and the loaded pools.  Both
snapshots are written create-only under ``<attempt-dir>/threadcap/`` and are
identical in structure in both arms.  The child's return code is unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence

from . import threadcap_v1 as TC


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - live
    args = list(sys.argv[1:] if argv is None else argv)
    arm, restore = TC.child_arm()
    TC.install_child_environment_restore(restore)
    from . import child_v1 as CH             # stdlib-only module; numerical imports happen later

    attempt_dir = CH.attempt_dir_from_argv(args)
    TC.require(attempt_dir is not None, "--attempt-dir is required")
    evidence = Path(attempt_dir) / TC.EVIDENCE_LEAF
    TC.write_snapshot(evidence, TC.settings_snapshot("pre", arm=arm))
    try:
        return CH.main(args)
    finally:
        TC.write_snapshot(evidence, TC.settings_snapshot("post", arm=arm))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
