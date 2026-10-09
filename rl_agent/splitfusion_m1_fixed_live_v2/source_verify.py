"""Standard-library-only verification of sealed retained and launcher sources."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

from .source_manifest import (EXPECTED_PATHS_SHA256, EXPECTED_SOURCE_COUNT,
    LAUNCHER_SOURCES, REQUIRED_SOURCES, SCHEMA)

_SHA256 = re.compile(r"[0-9a-f]{64}").fullmatch


def paths_sha256(paths):
    raw = json.dumps(sorted(paths), separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_files(sources, root):
    if not isinstance(sources, dict):
        raise ValueError("M1 fixed exact-source inventory must be an object")
    root = Path(root).resolve(strict=True)
    for relative, expected in sources.items():
        if not isinstance(relative, str) or not relative or relative.startswith("/"):
            raise ValueError("unsafe registered source path: " + repr(relative))
        parts = relative.split("/")
        if any(part in ("", ".", "..") for part in parts):
            raise ValueError("unsafe registered source path: " + repr(relative))
        pure = PurePosixPath(relative)
        if pure.is_absolute() or not isinstance(expected, str) or _SHA256(expected) is None:
            raise ValueError("invalid registered source identity: " + relative)
        candidate = root.joinpath(*pure.parts)
        if (not candidate.is_file() or candidate.is_symlink()
                or not candidate.resolve(strict=True).is_relative_to(root)):
            raise ValueError("registered source is not a safe in-root regular file: " + relative)
        if file_sha256(candidate) != expected:
            raise ValueError("registered source drift: " + relative)
    return len(sources)


def verify_registered_sources(sources, root):
    keys = tuple(sources) if isinstance(sources, dict) else ()
    fingerprint = paths_sha256(keys)
    if len(keys) != EXPECTED_SOURCE_COUNT or fingerprint != EXPECTED_PATHS_SHA256:
        raise ValueError(f"M1 fixed exact-source inventory incomplete or foreign: "
                         f"count={len(keys)} paths_sha256={fingerprint}")
    missing = sorted(set(REQUIRED_SOURCES).difference(keys))
    if missing:
        raise ValueError("M1 fixed inventory omits required sources: " + ", ".join(missing))
    return dict(schema=SCHEMA, checked=_verify_files(sources, root), paths_sha256=fingerprint)


def verify_launcher_sources(sources, root):
    if not isinstance(sources, dict) or set(sources) != set(LAUNCHER_SOURCES):
        raise ValueError("M1 fixed launcher-source inventory incomplete or foreign")
    return dict(checked=_verify_files(sources, root), paths_sha256=paths_sha256(sources))
