"""Fresh control worker, with identical startup collection and observation."""
from .bootstrap import main
from . import control_contract as C


if __name__ == "__main__":
    raise SystemExit(main(mode="control", contract=C))
