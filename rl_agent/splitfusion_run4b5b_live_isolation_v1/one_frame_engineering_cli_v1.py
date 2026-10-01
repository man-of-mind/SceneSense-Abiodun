"""Canonical package entrypoint for one-frame B engineering.

Running ``one_frame_engineering_v1`` directly with ``python -m`` gives that
module the name ``__main__``.  Exact-type safety checks in the separately
imported production factory must not mistake those duplicate class objects
for the canonical package classes.  This wrapper imports the implementation
normally and delegates only its CLI arguments; it changes no validation or
execution semantics.
"""

from __future__ import annotations

from .one_frame_engineering_v1 import main


if __name__ == "__main__":  # pragma: no cover - exercised by subprocess
    raise SystemExit(main())
