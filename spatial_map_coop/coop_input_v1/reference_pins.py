"""Pinned, read-only references to the frozen production direct-map modules.

This branch (``spatial-map-coop-v1``) predates
``rl_agent/splitfusion_direct_edge_map_v1/`` and therefore **cannot import** the
production protocol module. That is a deliberate constraint, not an oversight:

* duplicating ``protocol.validate_object_map_update`` here would create a second,
  divergent copy of a frozen contract, and
* weakening it into a permissive parser would let a malformed production document
  into a cooperative map that has no business accepting one.

So this module does the only honest thing available offline: it records the exact
file paths and SHA-256 digests of the production reference implementation as it
stood when this seam was written, and the adapters in this package validate
*canonical fixture documents* shaped by those pinned files.

``verify_reference_pins()`` is advisory. It reports agreement when the master
tree happens to be visible from this checkout and reports ``UNAVAILABLE`` when it
is not. It never imports production code and never fails a cooperative-map
operation, because the cooperative map in this branch never touches the
production wire path.

**Final runtime integration must happen later in a master-based bridge worktree**
that imports ``rl_agent.splitfusion_direct_edge_map_v1.protocol`` directly and
calls ``validate_object_map_update`` before this package's
``from_direct_object_map_update`` runs. See ``PRODUCTION_COMPATIBILITY_V1.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path


# Path of the production package relative to the abiodun/ repository root.
PRODUCTION_PACKAGE_RELPATH = "rl_agent/splitfusion_direct_edge_map_v1"

# Reference digests captured 2026-09-21 from the master worktree at
# .../neu_collab/abiodun (branch master, HEAD 18b1af3). Read-only observation.
REFERENCE_PINS: dict[str, str] = {
    "protocol.py": "158ecab3a8b3f2bd7efab2862a9ac5e12343fab38dad4760f295d28dff5df0da",
    "map_ingest.py": "dc0830c179f51874fb8d6038cd583e703649c394c9ff1d6385f757883462cb1c",
    "edge_publisher.py": "8e54f896e49b0656c02f6667b5ac91dbe8eb82338d562caa468622a95f7584c7",
}

# Production framing observed in phase2_map_sharing/transport.py at the same
# time. Recorded so the compatibility table is checkable rather than asserted.
PRODUCTION_CHUNK_HEADER_STRUCT = "!IHH"
PRODUCTION_CHUNK_HEADER_BYTES = 8
PRODUCTION_DEFAULT_CHUNK_BYTES = 12500
PRODUCTION_PAYLOAD_COMPRESSION = "zlib.compress(level=1)"
PRODUCTION_PAYLOAD_ENCODING = "json.dumps(allow_nan=False, sort_keys=True, separators=(',',':'))"

# Schema identifiers of the frozen production documents. Declared as constants so
# the adapters can reject schema drift without importing production code.
DIRECT_OBJECT_MAP_UPDATE_SCHEMA = "splitfusion_direct_object_map_update.v1"
DIRECT_MAP_FEEDBACK_SCHEMA = "splitfusion_direct_map_feedback.v1"
EDGE_TERMINAL_CONTROL_SCHEMA = "splitfusion_edge_terminal_control.v1"
DIRECT_PROTOCOL_VERSION = 1

# Mirrors protocol.REQUIRED_UPDATE_IDENTITY at the pinned digest. The adapter
# requires every one of these; it does not re-implement the rest of the
# production validator.
DIRECT_REQUIRED_IDENTITY = (
    "run_id",
    "cell_id",
    "stream_id",
    "frame_id",
    "action_id",
    "profile_id",
    "capture_timestamp_ns",
)

# Mirrors protocol.TERMINAL_OUTCOMES / AGENT_CREDITS at the pinned digest.
DIRECT_TERMINAL_OUTCOMES = (
    "RESULT_INSTALLED",
    "SUPERSEDED_PENDING",
    "STALE_BEFORE_EDGE",
    "STALE_BEFORE_MAP",
    "TRANSPORT_INCOMPLETE",
    "MAP_REJECTED",
    "FEEDBACK_TIMEOUT",
)
DIRECT_AGENT_CREDITS = (
    "CREDIT_INSTALLED_ON_TIME",
    "CREDIT_INSTALLED_LATE",
    "CREDIT_SUPERSEDED_BY_FRESHER",
    "CREDIT_STALE_AT_SOURCE",
    "CREDIT_STALE_AT_MAP",
    "CREDIT_NETWORK_INCOMPLETE",
    "CREDIT_REJECTED",
    "CREDIT_NO_FEEDBACK",
)

INTEGRATION_STATUS = "REQUIRES_MASTER_BASED_BRIDGE_WORKTREE"


@dataclass(frozen=True)
class PinStatus:
    filename: str
    expected_sha256: str
    observed_sha256: str | None
    status: str  # "MATCH" | "MISMATCH" | "UNAVAILABLE"


def _repository_root() -> Path:
    # .../abiodun/spatial_map_coop/coop_input_v1/reference_pins.py -> .../abiodun
    return Path(__file__).resolve().parents[2]


def verify_reference_pins(root: Path | str | None = None) -> tuple[PinStatus, ...]:
    """Advisory comparison of the pinned digests against a visible master tree.

    Returns ``UNAVAILABLE`` per file when the production package is not present,
    which is the expected result inside this branch.
    """

    base = Path(root) if root is not None else _repository_root()
    package = base / PRODUCTION_PACKAGE_RELPATH
    results: list[PinStatus] = []
    for filename in sorted(REFERENCE_PINS):
        expected = REFERENCE_PINS[filename]
        candidate = package / filename
        if not candidate.is_file():
            results.append(PinStatus(filename, expected, None, "UNAVAILABLE"))
            continue
        digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
        results.append(
            PinStatus(
                filename,
                expected,
                digest,
                "MATCH" if digest == expected else "MISMATCH",
            )
        )
    return tuple(results)


def reference_pin_report(root: Path | str | None = None) -> dict[str, object]:
    statuses = verify_reference_pins(root)
    return {
        "production_package_relpath": PRODUCTION_PACKAGE_RELPATH,
        "integration_status": INTEGRATION_STATUS,
        "production_module_importable_in_this_branch": False,
        "files": [
            {
                "filename": item.filename,
                "expected_sha256": item.expected_sha256,
                "observed_sha256": item.observed_sha256,
                "status": item.status,
            }
            for item in statuses
        ],
    }
