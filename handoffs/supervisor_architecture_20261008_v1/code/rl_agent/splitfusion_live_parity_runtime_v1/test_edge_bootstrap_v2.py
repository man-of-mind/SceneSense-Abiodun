"""Source-parity regressions for the edge-only additive startup repair."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def function(path, name):
    return next(node for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == name)


def body_without_doc_and_imports(node):
    return [ast.dump(x, include_attributes=False) for x in node.body
            if not isinstance(x, (ast.Import, ast.ImportFrom))
            and not (isinstance(x, ast.Expr) and isinstance(x.value, ast.Constant)
                     and isinstance(x.value.value, str))]


class EdgeBootstrapV2Tests(unittest.TestCase):
    def test_dispatch_preserves_original_non_import_statements(self):
        old = function(ROOT / "rl_agent/splitfusion_live_parity_repair_v1/composition.py", "run_edge")
        new = function(HERE / "edge_bootstrap_v2.py", "run_edge_only")
        self.assertEqual(body_without_doc_and_imports(old), body_without_doc_and_imports(new))

    def test_entry_preserves_registration_and_scope_order(self):
        old = function(HERE / "entries.py", "edge")
        new = function(HERE / "edge_bootstrap_v2.py", "edge")

        class ReplaceDelegate(ast.NodeTransformer):
            def visit_Attribute(self, node):
                if isinstance(node.value, ast.Name) and node.value.id == "composition" and node.attr == "run_edge":
                    return ast.Name(id="run_edge_only", ctx=ast.Load())
                return self.generic_visit(node)

        old = ReplaceDelegate().visit(old)
        self.assertEqual(body_without_doc_and_imports(old), body_without_doc_and_imports(new))

    def test_no_ue_only_dependencies_imported(self):
        tree = ast.parse((HERE / "edge_bootstrap_v2.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [name.name for name in node.names]
                self.assertFalse(set(names) & {"composition", "gt_bridge", "source_geometry", "scipy"})


if __name__ == "__main__":
    unittest.main()
