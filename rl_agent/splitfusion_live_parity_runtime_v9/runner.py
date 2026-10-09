"""Retained runner/partition lifecycle with prospective registered E3 scope."""
from pathlib import Path
import sys
from rl_agent.splitfusion_live_parity_runtime_v8 import runner as V8
from . import contract as C
from .bindings import namespace
from .process_scope import registered_inventory_scope

_bound = namespace(V8, C=C, __file__=__file__)


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args.count("--registration") != 1 or args.index("--registration") + 1 >= len(args):
        raise ValueError("explicit sealed registration required")
    output = Path(args[args.index("--registration") + 1]).parent
    with registered_inventory_scope(output):
        return _bound["main"](args)


if __name__ == "__main__":
    raise SystemExit(main())
