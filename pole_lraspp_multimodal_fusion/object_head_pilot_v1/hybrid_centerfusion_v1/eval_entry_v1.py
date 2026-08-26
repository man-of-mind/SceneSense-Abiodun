#!/usr/bin/env python3
"""Evaluation entry point: install the hybrid builder, then run the production
evaluator unchanged. Every decoder argument is forwarded verbatim."""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent.parent), str(_HERE.parent.parent.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hybrid_model_v1 import install  # noqa: E402


def main() -> None:
    install()
    from pole_lraspp_multimodal_fusion import evaluate_fusion
    evaluate_fusion.main()


if __name__ == "__main__":
    main()
