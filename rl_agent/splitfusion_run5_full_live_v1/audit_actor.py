"""Profiler audit module and G8 replay loader for the full Run-5 live actor (``python -m`` prints the audit JSON)."""
import json

from .actor import audit, load_registered_actor  # noqa: F401  (load_registered_actor is the G8 replay loader)


def main() -> int:
    result = audit()
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
