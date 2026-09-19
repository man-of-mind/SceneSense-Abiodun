"""Command line entry point; metadata commands are model/CUDA inert."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

from .contract import (
    EXPECTED_GRID_ROWS,
    OfflineGridContractError,
    repository_root,
    sha256_file,
)
from .executor import EXECUTE_TOKEN
from .manifest import (
    completion_manifest_document,
    run_manifest_document,
    row_schema_document,
    validate_completion_manifest,
)
from .preflight import run_metadata_preflight
from .quality import load_reward_spec
from .schema import expected_row_keys
from .selection import (
    build_selection_manifest,
    validate_selection_manifest,
    verify_selection_sources,
    write_json_create_only,
)
from .store import ExactRowStore


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OfflineGridContractError(f"cannot load {label}: {path}") from exc
    if type(value) is not dict:
        raise OfflineGridContractError(f"{label} must be a JSON object")
    return value


def _print(document: Mapping[str, Any]) -> None:
    print(json.dumps(document, indent=2, sort_keys=True, allow_nan=False))


def _write_bytes_create_only(path: Path, payload: bytes) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    if path.exists() or partial.exists():
        raise OfflineGridContractError(f"create-only output exists: {path}")
    with partial.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return sha256_file(path)


def _persist_or_verify_bytes(path: Path, payload: bytes, label: str) -> str:
    if path.exists():
        if path.read_bytes() != payload:
            raise OfflineGridContractError(f"durable {label} differs from current input")
        return sha256_file(path)
    return _write_bytes_create_only(path, payload)


def _persist_or_verify_json(
    path: Path, document: Mapping[str, Any], label: str
) -> str:
    if path.exists():
        if _load_json_object(path, f"durable {label}") != dict(document):
            raise OfflineGridContractError(f"durable {label} differs from current binding")
        return sha256_file(path)
    return write_json_create_only(path, document)


def _next_attempt_directory(output: Path) -> Path:
    parent = output / "attempts"
    parent.mkdir(parents=True, exist_ok=True)
    ordinals = []
    for child in parent.iterdir():
        if child.is_dir() and child.name.startswith("attempt_"):
            try:
                ordinals.append(int(child.name.removeprefix("attempt_")))
            except ValueError:
                raise OfflineGridContractError(f"foreign attempt directory: {child}")
    path = parent / f"attempt_{(max(ordinals, default=0) + 1):04d}"
    path.mkdir(exist_ok=False)
    return path


def _artifact_hashes(output: Path, paths: list[Path]) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            raise OfflineGridContractError(f"completion artifact is missing: {path}")
        relative = str(path.relative_to(output))
        result[relative] = sha256_file(path)
    return dict(sorted(result.items()))


def _manifest_inputs(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], Any, dict[str, Any], dict[str, Any]]:
    root = repository_root()
    preflight = run_metadata_preflight(root)
    selection = _load_json_object(args.selection, "selection manifest")
    validate_selection_manifest(selection)
    source_audit = verify_selection_sources(root, selection)
    reward_spec = load_reward_spec(args.reward_spec, args.reward_spec_sha256)
    manifest = run_manifest_document(
        selection_manifest=selection, preflight=preflight, reward_spec=reward_spec
    )
    return preflight, selection, reward_spec, manifest, source_audit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed exact offline SplitFusion quality-grid producer"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    preflight = sub.add_parser("preflight", help="metadata/hash checks only; no payload/model/CUDA")
    preflight.add_argument("--output", type=Path)

    schema = sub.add_parser("row-schema", help="print the strict durable row schema")
    schema.add_argument("--output", type=Path)

    select = sub.add_parser("select", help="generate the deterministic 512+256 selection")
    select.add_argument("--output", required=True, type=Path)
    select.add_argument("--progress-every", type=int, default=100)

    for name in ("dry-run", "execute"):
        command = sub.add_parser(name)
        command.add_argument("--selection", required=True, type=Path)
        command.add_argument("--reward-spec", required=True, type=Path)
        command.add_argument("--reward-spec-sha256", required=True)
        if name == "execute":
            command.add_argument("--output", required=True, type=Path)
            command.add_argument("--resume", action="store_true")
            command.add_argument("--execute-token", required=True)
            command.add_argument(
                "--max-new-rows",
                type=int,
                help="explicit bounded smoke execution; leaves a resumable incomplete store",
            )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "preflight":
            report = run_metadata_preflight()
            if args.output:
                write_json_create_only(args.output, report)
            _print(report)
            return 0
        if args.command == "row-schema":
            document = row_schema_document()
            if args.output:
                write_json_create_only(args.output, document)
            _print(document)
            return 0
        if args.command == "select":
            if args.progress_every < 0:
                raise OfflineGridContractError("--progress-every must be non-negative")
            selection = build_selection_manifest(progress_every=args.progress_every)
            output_sha256 = write_json_create_only(args.output, selection)
            _print(
                {
                    "status": "SELECTION_CREATED",
                    "path": str(args.output),
                    "file_sha256": output_sha256,
                    "selection_manifest_sha256": selection["selection_manifest_sha256"],
                    "selected_frames": selection["selected_frame_count"],
                    "test_episode_access": "NONE_ALLOWLIST_ONLY_05_06",
                }
            )
            return 0
        preflight, selection, reward_spec, manifest, source_audit = _manifest_inputs(args)
        if args.command == "dry-run":
            _print(
                {
                    "status": "PASS_DRY_RUN_NO_MODEL_OR_CUDA_INFERENCE",
                    "preflight_binding_sha256": preflight["preflight_binding_sha256"],
                    "selection_manifest_sha256": selection["selection_manifest_sha256"],
                    "run_manifest": manifest,
                    "selected_source_audit": source_audit,
                    "expected_rows": EXPECTED_GRID_ROWS,
                    "would_load_model": False,
                    "would_query_cuda": False,
                }
            )
            return 0
        if args.execute_token != EXECUTE_TOKEN:
            raise OfflineGridContractError(
                f"execution refused; --execute-token must equal {EXECUTE_TOKEN!r}"
            )
        if args.max_new_rows is not None and args.max_new_rows <= 0:
            raise OfflineGridContractError("--max-new-rows must be positive")
        output = args.output.resolve()
        if args.resume and not output.is_dir():
            raise OfflineGridContractError("--resume requires an existing output directory")
        if not args.resume and output.exists() and any(output.iterdir()):
            raise OfflineGridContractError("execute output is create-only and already non-empty")
        output.mkdir(parents=True, exist_ok=True)
        complete_terminal_path = output / "COMPLETE.json"
        final_manifest_path = output / "run_manifest.json"
        if complete_terminal_path.exists() or final_manifest_path.exists():
            raise OfflineGridContractError("completed output is immutable and cannot resume")
        initial_manifest_path = output / "run_manifest.initial.json"
        selection_copy_path = output / "selection_manifest.json"
        reward_copy_path = output / "reward_spec.json"
        preflight_path = output / "metadata_preflight.json"
        source_audit_path = output / "selection_source_audit.json"
        row_schema_path = output / "row_schema.json"
        store_path = output / "quality_rows.sqlite3"
        durable_manifest = manifest
        _persist_or_verify_bytes(
            selection_copy_path, args.selection.read_bytes(), "selection manifest"
        )
        _persist_or_verify_bytes(
            reward_copy_path, args.reward_spec.read_bytes(), "reward spec"
        )
        _persist_or_verify_json(preflight_path, preflight, "metadata preflight")
        _persist_or_verify_json(source_audit_path, source_audit, "selection source audit")
        _persist_or_verify_json(row_schema_path, row_schema_document(), "row schema")
        _persist_or_verify_json(
            initial_manifest_path, durable_manifest, "initial run manifest"
        )
        from .executor import execute_grid

        frozen_keys = expected_row_keys(selection)
        attempt = _next_attempt_directory(output)
        try:
            with ExactRowStore(
                store_path, durable_manifest,
                resume=bool(args.resume and store_path.exists()),
                expected_row_keys=frozen_keys,
            ) as store:
                result = execute_grid(
                    selection=selection,
                    manifest=durable_manifest,
                    reward_spec=reward_spec,
                    store=store,
                    max_new_rows=args.max_new_rows,
                )
        except Exception as exc:
            write_json_create_only(
                attempt / "failure.json",
                {
                    "schema": "splitfusion_exact_offline_quality_grid_failure_v1",
                    "status": "FAILED",
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "run_binding_sha256": durable_manifest["run_binding_sha256"],
                },
            )
            raise
        runtime_preflight_path = attempt / "runtime_preflight.json"
        runtime_environment_path = attempt / "runtime_environment.json"
        runtime_source_audit_path = attempt / "runtime_source_audit.json"
        equivalence_path = attempt / "shared_single_action_equivalence.json"
        execution_result_path = attempt / "execution_result.json"
        write_json_create_only(runtime_preflight_path, result["runtime_preflight"])
        write_json_create_only(
            runtime_environment_path, result["runtime_environment"]
        )
        write_json_create_only(
            runtime_source_audit_path, result["runtime_source_audit"]
        )
        write_json_create_only(
            equivalence_path,
            {
                "schema": "splitfusion_shared_single_action_equivalence_v1",
                "result": result["shared_single_action_equivalence"],
            },
        )
        write_json_create_only(execution_result_path, result)

        response: dict[str, Any] = dict(result)
        response["attempt_directory"] = str(attempt)
        if result["status"] == "COMPLETE":
            base_artifacts = sorted(
                path
                for path in output.rglob("*")
                if path.is_file()
                and path not in (final_manifest_path, complete_terminal_path)
            )
            partials = [path for path in base_artifacts if path.name.endswith(".partial")]
            if partials:
                raise OfflineGridContractError(
                    f"completion refused with partial artifacts: {partials}"
                )
            artifact_sha256 = _artifact_hashes(output, base_artifacts)
            final_manifest = completion_manifest_document(
                initial_manifest=durable_manifest,
                execution_result=result,
                artifact_sha256=artifact_sha256,
                store_audit=result["store_audit"],
            )
            validate_completion_manifest(final_manifest)
            final_manifest_file_sha256 = write_json_create_only(
                final_manifest_path, final_manifest
            )
            terminal = {
                "schema": "splitfusion_exact_offline_quality_grid_complete_v1",
                "status": "COMPLETE",
                "run_binding_sha256": durable_manifest["run_binding_sha256"],
                "run_manifest_sha256": final_manifest["run_manifest_sha256"],
                "run_manifest_file_sha256": final_manifest_file_sha256,
                "rows": int(result["store_audit"]["rows"]),
                "expected_rows": EXPECTED_GRID_ROWS,
            }
            terminal_file_sha256 = write_json_create_only(
                complete_terminal_path, terminal
            )
            response.update(
                final_run_manifest=str(final_manifest_path),
                final_run_manifest_file_sha256=final_manifest_file_sha256,
                completion_terminal=str(complete_terminal_path),
                completion_terminal_file_sha256=terminal_file_sha256,
            )
        _print(response)
        return 0
    except (OfflineGridContractError, OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
