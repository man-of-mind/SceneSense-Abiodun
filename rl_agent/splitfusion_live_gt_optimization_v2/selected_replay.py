"""Repeat selected combined-vs-control parity/cost with final guarded source."""
from . import benchmark


def main():
    # Per-process invocation setting, not a change to the earlier sealed script.
    benchmark.ARMS = ("control", "combined")
    return benchmark.main()


if __name__ == "__main__":
    main()
