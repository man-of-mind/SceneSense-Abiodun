from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Iterable

from .contracts import PACKAGE, resolve_repo_path, load_recovery_config


CALL_NAMES = {"normalize", "norm", "vector_norm", "log", "log1p", "exp", "expm1", "sqrt", "softmax",
              "normalize_yaw_fp32", "exp_dimensions_fp64", "decode", "div", "groupnorm"}


def _call_name(node: ast.Call) -> str:
    try:
        return ast.unparse(node.func)
    except Exception:
        return "<call>"


def inventory(paths: Iterable[Path] | None = None) -> list[dict[str, Any]]:
    """Enumerate active divisions/projections and named sensitive operations."""
    if paths is None:
        original = resolve_repo_path(load_recovery_config()["original"]["package"])
        scoring = original.parent / "route_b_v3_1_native_grid_expanded_training_v2/scoring_v2.py"
        native = original.parent / "route_b_v3_1_native_grid_v1/evaluate_v1.py"
        scoring_contract = original.parent / "route_b_v3_1_clean_base_v1/score_contract_v1.py"
        audit = original.parent / "route_b_v3_1_targeted_refinement_v1/audit_v1.py"
        paths = (original / "model.py", original / "losses.py", original / "evaluate.py", scoring,
                 native, scoring_contract, audit,
                 PACKAGE / "safe_math.py", PACKAGE / "recovery_model.py", PACKAGE / "recovery_losses.py",
                 PACKAGE / "guards.py")
    rows = []
    for path in paths:
        source = path.read_text(encoding="utf-8"); tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            kind = None
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                # Pathlib joins also use `/`; a string/f-string operand cannot
                # be a numerical denominator and is excluded.
                path_literal = lambda value: isinstance(value, ast.JoinedStr) or (
                    isinstance(value, ast.Constant) and isinstance(value.value, (str, bytes)))
                expression = ast.get_source_segment(source, node) or ""
                path_tokens = ("Path(", "ROOT /", "PACKAGE /", "FUSION_ROOT /", "NATIVE_PACKAGE /",
                               "BASE_PKG /", "REFINE_PKG /", "experiment /", "prediction_root /",
                               "checkpoint_dir /", "baseline_experiment /")
                obvious_path_join = any(token in expression for token in path_tokens)
                if not path_literal(node.left) and not path_literal(node.right) and not obvious_path_join:
                    kind = "division"
            elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
                kind = "projection_matmul"
            elif isinstance(node, ast.Call):
                name = _call_name(node)
                terminal = name.lower().split(".")[-1]
                if terminal in CALL_NAMES or terminal == "lovasz_softmax":
                    kind = "box_decode" if name.endswith(".decode") else "sensitive_call"
            if kind is not None:
                rows.append({"file": str(path), "line": int(node.lineno), "kind": kind,
                             "expression": ast.get_source_segment(source, node)})
    return sorted(rows, key=lambda row: (row["file"], row["line"], row["expression"] or ""))
