from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

from .contracts import ROOT, load_recovery_config, resolve_repo_path, sha256

MODULES = ("common", "data", "model", "losses", "train", "infer", "evaluate")


def load_base() -> SimpleNamespace:
    """Load the immutable original modules only when a gated runtime calls us."""
    config = load_recovery_config()
    package = resolve_repo_path(config["original"]["package"])
    expected = config["original"]["source_files_sha256"]
    for name, digest in expected.items():
        if sha256(package / name) != digest:
            raise RuntimeError(f"immutable original source drift: {name}")
    for name in MODULES:
        loaded = sys.modules.get(name)
        if loaded is not None and Path(loaded.__file__).resolve().parent != package:
            raise RuntimeError(f"unsafe top-level module collision: {name} from {loaded.__file__}")
    sys.path.insert(0, str(package))
    try:
        values = {name: importlib.import_module(name) for name in MODULES}
    finally:
        if sys.path[0] == str(package):
            sys.path.pop(0)
    for name, module in values.items():
        if Path(module.__file__).resolve().parent != package:
            raise RuntimeError(f"base module resolution drift: {name}")
    return SimpleNamespace(**values)


@contextmanager
def patched_base_loss(base: SimpleNamespace, replacement: object) -> Iterator[None]:
    original = base.losses.geometry_losses
    base.losses.geometry_losses = replacement
    try:
        yield
    finally:
        base.losses.geometry_losses = original
