"""Independent identity for a prospective change to evidence durability timing."""
import hashlib
import json
from pathlib import Path
from rl_agent.splitfusion_live_gt_optimization_v6 import contract as V6

ENVIRONMENT = dict(V6.ENVIRONMENT)
BOOTSTRAP = "rl_agent.splitfusion_live_gt_handoff_v1.bootstrap"
validate_context = V6.validate_context
SCHEMA = "splitfusion.gt_scoring_before_archive.v1"
SOCKET_NAME = "gt_packet.sock"
MAX_JOBS = 8
MAX_BYTES = 192 * 1024 * 1024


def sources():
    root = Path(__file__).resolve().parents[2]
    result = V6.sources()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        if not path.name.startswith("test_"):
            if path.is_symlink() or not path.is_file():
                raise ValueError("regular handoff source required")
            result[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(result.items()))


def binding():
    return hashlib.sha256(json.dumps({"schema": SCHEMA, "sources": sources()},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
