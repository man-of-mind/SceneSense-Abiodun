"""Canonical package entrypoint for the split-host 300-frame B run.

Like ``one_frame_engineering_cli_v1``: running the implementation module
directly would load it as ``__main__`` and duplicate its classes, so this
wrapper imports it normally and delegates only the CLI arguments.
"""

from __future__ import annotations

from .split_host_300_v1 import main


if __name__ == "__main__":  # pragma: no cover - exercised by subprocess
    raise SystemExit(main())
