"""Verify source bytes once, compile those bytes, recheck at run postflight.

No import side effects, execution, global patching or filesystem cache. A caller
must use the returned verified bytes/code, not reopen a path for execution.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from .contract import ContractError, canonical, is_digest, require, sha256


@dataclass(frozen=True, slots=True)
class SourceEntry:
    name: str
    relative_path: str
    sha256: str
    raw: bytes


@dataclass(frozen=True, slots=True, init=False)
class VerifiedSources:
    root: Path
    entries: tuple[SourceEntry, ...]
    binding_sha256: str

    def __init__(self, *args, **kwargs):
        raise ContractError("use verify_sources_once")

    def source(self, name):
        for entry in self.entries:
            if entry.name == name:
                return entry
        raise ContractError("source role not verified")

    def compile(self, name, *, functions=None):
        entry = self.source(name)
        filename = str(self.root / entry.relative_path)
        if functions is None:
            return compile(entry.raw, filename, "exec", dont_inherit=True)
        require(type(functions) is tuple and functions and len(set(functions)) == len(functions)
                and all(type(x) is str for x in functions), "unique selected function names required")
        parsed = ast.parse(entry.raw, filename=filename)
        nodes = [node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name in functions]
        require(len(nodes) == len(functions) and {n.name for n in nodes} == set(functions),
                "verified function inventory changed")
        futures = [node for node in parsed.body if isinstance(node, ast.ImportFrom)
                   and node.module == "__future__"]
        return compile(ast.Module(body=futures + nodes, type_ignores=[]), filename, "exec", dont_inherit=True)

    def verify_unchanged(self):
        pins = {entry.name: (entry.relative_path, entry.sha256) for entry in self.entries}
        checked = verify_sources_once(self.root, pins)
        require(checked.binding_sha256 == self.binding_sha256, "source inventory changed")
        return MappingProxyType({e.relative_path: e.sha256 for e in checked.entries})


def verify_sources_once(root, pins):
    root = Path(root).resolve(strict=True)
    require(root.is_dir() and type(pins) is dict and pins, "explicit root/source inventory required")
    entries, seen = [], set()
    for name, spec in sorted(pins.items()):
        require(type(name) is str and name and type(spec) in (list, tuple) and len(spec) == 2,
                "invalid source role")
        relative, expected = spec
        require(type(relative) is str and is_digest(expected), "explicit source path/digest required")
        p = Path(relative)
        require(not p.is_absolute() and ".." not in p.parts and p.as_posix() == relative,
                "source path must be canonical and relative")
        path = root / p
        require(not path.is_symlink(), "symlink source refused")
        resolved = path.resolve(strict=True)
        require(resolved.is_relative_to(root) and resolved.is_file(), "source escapes root or is not regular")
        require(relative not in seen, "duplicate source path")
        seen.add(relative)
        raw = path.read_bytes()
        require(sha256(raw) == expected, "source pin differs: " + relative)
        entries.append(SourceEntry(name, relative, expected, raw))
    bundle = object.__new__(VerifiedSources)
    identity = {e.name: dict(path=e.relative_path, sha256=e.sha256) for e in entries}
    for key, value in dict(root=root, entries=tuple(entries), binding_sha256=sha256(canonical(identity))).items():
        object.__setattr__(bundle, key, value)
    return bundle
