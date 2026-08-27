#!/usr/bin/env python3
"""Install the clean CenterNet builder and run the existing Route B evaluator."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for path in (HERE, HERE.parent.parent, HERE.parent.parent.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from centernet_model_v1 import install  # noqa: E402


def main() -> None:
    install()
    from pole_lraspp_multimodal_fusion import evaluate_fusion
    evaluate_fusion.main()


if __name__ == "__main__":
    main()

