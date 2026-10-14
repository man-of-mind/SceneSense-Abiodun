"""Verify packaged file hashes; no access to the original experiment paths."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


if __name__ == "__main__":
    manifest = json.loads((ROOT / "FILE_SHA256.json").read_text())
    failed = []
    for name, digest in manifest.items():
        p = ROOT / name
        if not p.is_file() or not p.resolve().is_relative_to(ROOT) or sha(p) != digest:
            failed.append(name)
    print(json.dumps({"status": "FAIL" if failed else "PASS", "files": len(manifest), "mismatches": failed}, indent=2))
    raise SystemExit(bool(failed))
