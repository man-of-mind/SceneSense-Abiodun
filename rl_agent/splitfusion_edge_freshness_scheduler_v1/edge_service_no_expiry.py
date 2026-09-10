"""Live latest-only edge with no pre-compute expiry ceiling."""

from .live_edge_service import main as _main
from .pipeline import CandidatePolicy


def main(argv: list[str] | None = None) -> int:
    return _main(argv, forced_policy=CandidatePolicy.LATEST_ONLY_NO_EXPIRY)


if __name__ == "__main__":
    raise SystemExit(main())
