#!/usr/bin/env python3
"""Dedicated live engine for the bounded UE-N3C -3.0 dB refinement.

The default configuration remains pending and therefore cannot execute OAI or
open the RFsim control socket.  Once a sealed N3B adjudication and a separately
reviewed explicit-live config are pinned, this engine runs exactly three fresh
cold starts using run-local configs.  The -3.0 dB candidate is baked before UE
launch and is never applied over Telnet; exactly one Telnet command restores
the UL model to -50 dB after candidate evidence.

The 6.5 +/- 0.5 dB achieved-PUSCH band is consistency-only provenance from a
sealed already-attached command-ladder rung.  It is not an actuator mapping or
a promoted numeric bound.  Post-restore recovery is required infrastructure
evidence: failure is fail-closed, never a valid candidate service failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import signal
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_agent import ue_n2_oai_ul_calibration_smoke as n2  # noqa: E402
from rl_agent import ue_n3_oai_ul_command_calibration_v1 as calibration  # noqa: E402
from rl_agent import ue_n3b_oai_ul_cold_attach_confirmation_v1 as n3b  # noqa: E402
from rl_agent import ue_n3c_oai_ul_cold_attach_refinement_v1 as plan  # noqa: E402


DEFAULT_CONFIG = (
    ROOT / "rl_agent/configs/ue_n3c_oai_ul_cold_attach_refinement_live_v1.json"
)
CANONICAL_PLAN_CONFIG = (
    ROOT / "rl_agent/configs/ue_n3c_oai_ul_cold_attach_refinement_v1.json"
)
CANONICAL_PLAN_CONFIG_SHA256 = (
    "d2719311feee728c3d33f2b4f46c81f5f3806addf91826c7e4f4bb3999e541b6"
)
PREPARE_ONLY = plan.PREPARE_ONLY
EXECUTE_LIVE = plan.EXECUTE_LIVE
SCHEMA = "scenesense.ue_n3c_oai_ul_cold_attach_refinement_live_config.v1"
PLAN_BLOCKED = "UE_N3C_LIVE_PLAN_FROZEN_PREREQUISITES_PENDING"
PLAN_READY = "UE_N3C_LIVE_PLAN_FROZEN_READY_FOR_EXPLICIT_EXECUTE_LIVE"
REP_PASSED = "UE_N3C_COLD_ATTACH_SERVICE_REPETITION_PASSED"
REP_ATTACH_FAILED = (
    "UE_N3C_COLD_ATTACH_REPETITION_VALID_OPERATIONAL_SCREEN_NONCONFIRMATION"
)
REP_SERVICE_FAILED = "UE_N3C_COLD_ATTACH_REPETITION_VALID_SERVICE_FAILURE"
REP_SNR_MISMATCH = "UE_N3C_COLD_ATTACH_REPETITION_VALID_SNR_CONSISTENCY_MISMATCH"
REP_UNCONFIRMED = "UE_N3C_COLD_ATTACH_REPETITION_EVIDENCE_UNCONFIRMED"
CAMPAIGN_PASSED = "UE_N3C_COLD_ATTACH_REFINEMENT_3_OF_3_PASSED_REVIEW_REQUIRED"
CAMPAIGN_NOT_3_OF_3 = "UE_N3C_COLD_ATTACH_REFINEMENT_NOT_3_OF_3_REVIEW_REQUIRED"
RESTORE_FAILED = "UE_N3C_FAILED_RESTORE"
FAILED = "FAILED"

EXPECTED_N3B_ADJUDICATION = {
    "directory": (
        "rl_agent/experiments/ue_n3b_cold_attach_outcome_review_v1/"
        "20260821_review_01"
    ),
    "manifest": "manifest.json",
    "manifest_sha256": (
        "bda88d7f89e41822bf07209a69f32a9a2f72cacabccc6112364cfa81b135398f"
    ),
    "terminal": (
        "UE_N3B_OUTCOME_ADJUDICATED_N3C_ELIGIBLE_REVIEW_REQUIRED.json"
    ),
    "terminal_sha256": (
        "55152131655d66c35c37b83167ac9174f9d95ac37e9419d0c1ab4e8bc313f11f"
    ),
    "resolved_config": "resolved_config.json",
    "resolved_config_sha256": (
        "b2c3bd8936b372f243839a16c6f33b0a36283cee1551f100ee9ec5fd880f71dc"
    ),
    "source_config_sha256": (
        "b1e702632bfc922f84e25a1619c75550b642cbb4b1188ca10ada1fa092330ce2"
    ),
    "source_runner_sha256": (
        "e6ef1bce54e29421ff36debef787daeac706acc3126643e9ab4f1cb6520a005d"
    ),
    "required_status": (
        "UE_N3B_OUTCOME_ADJUDICATED_N3C_ELIGIBLE_REVIEW_REQUIRED"
    ),
    "n3c_eligibility_status": (
        "UE_N3B_VALID_COLD_ATTACH_FAILURE_ACCEPTED_FOR_N3C"
    ),
    "n3c_selected_command_db": -3.0,
}

PROTECTED_SOURCE_DIRECTORIES = (
    plan.EXPECTED_N3B_LIVE["directory"],
    EXPECTED_N3B_ADJUDICATION["directory"],
    plan.EXPECTED_COMMAND_LADDER["directory"],
    "rl_agent/experiments/ue_n3c_oai_ul_cold_attach_refinement_v1",
)
EXPECTED_OUTPUT = {
    "resolved_config": "resolved_config.json",
    "campaign_plan": "campaign_plan.csv",
    "campaign_summary": "campaign_summary.json",
    "repetition_summary": "repetition_summary.json",
    "manifest": "manifest.json",
    "command_log": "command_log.csv",
    "raw_radio_events": "raw_radio_events.jsonl",
    "failure": "FAILED.json",
}


class LiveRefinementFailure(plan.ColdAttachRefinementFailure):
    """Fail-closed live authority, evidence, infrastructure, or lifecycle failure."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LiveRefinementFailure(message)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_output_contract(config: Mapping[str, Any]) -> None:
    """Reject output-path drift before any directory or evidence is written."""
    require(
        config.get("output") == EXPECTED_OUTPUT,
        "N3C live output leaf contract drift",
    )


def _scientific_projection(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return fields that must remain identical to the sealed PREPARE contract."""
    return {
        "claim_boundary": config["claim_boundary"],
        "campaign": config["campaign"],
        "startup_channel": config["startup_channel"],
        "rung": config["rung"],
        "traffic": config["traffic"],
        "transport_gates": config["transport_gates"],
        "preflight": config["preflight"],
        "radio": config["radio"],
        "actuator": config["actuator"],
        "telemetry": config["telemetry"],
        "analysis": config["analysis"],
        "output": config["output"],
        "paths": {
            key: value for key, value in config["paths"].items()
            if key != "output_root"
        },
        "ue_n1_bundle": config["predecessors"]["ue_n1_bundle"],
        "n3b_live_evidence": config["predecessors"]["n3b_live_evidence"],
        "command_ladder_expectation": config["predecessors"][
            "command_ladder_expectation"
        ],
    }


def _strict_manifest_inventory(
    directory: Path, manifest: Mapping[str, Any], *, expected_output_count: int,
) -> None:
    """Require both listed-file integrity and complete directory membership."""
    listed = plan._verify_manifest_inventory(directory, manifest)
    require(len(listed) == expected_output_count,
            f"strict manifest output-count mismatch: {directory}")
    status = str(manifest.get("status", ""))
    terminal = "FAILED.json" if status == FAILED else f"{status}.json"
    expected = set(listed) | {"manifest.json", terminal}
    actual = {
        str(path.relative_to(directory))
        for path in directory.rglob("*") if path.is_file()
    }
    require(
        actual == expected,
        f"strict manifest completeness mismatch at {directory}: "
        f"unexpected={sorted(actual - expected)} missing={sorted(expected - actual)}",
    )


def verify_strict_predecessor_completeness(config: Mapping[str, Any]) -> None:
    """Reject unlisted additions or omissions in every sealed predecessor tree."""
    n3b_source = config["predecessors"]["n3b_live_evidence"]
    n3b_dir = plan.resolve_repo_path(str(n3b_source["directory"]))
    n3b_manifest = load_json(n3b_dir / str(n3b_source["manifest"]))
    _strict_manifest_inventory(n3b_dir, n3b_manifest, expected_output_count=71)
    repetitions = sorted((n3b_dir / "repetitions").glob("rep_*_cold"))
    require(len(repetitions) == 3, "sealed N3B repetition directory count drift")
    for directory in repetitions:
        _strict_manifest_inventory(
            directory,
            load_json(directory / "manifest.json"),
            expected_output_count=20,
        )

    review_source = config["predecessors"]["n3b_adjudication"]
    require(not plan.pending_adjudication(review_source),
            "strict N3B adjudication completeness is pending")
    review_dir = plan.resolve_repo_path(str(review_source["directory"]))
    _strict_manifest_inventory(
        review_dir,
        load_json(review_dir / str(review_source["manifest"])),
        expected_output_count=6,
    )

    ladder_source = config["predecessors"]["command_ladder_expectation"]
    ladder_dir = plan.resolve_repo_path(str(ladder_source["directory"]))
    _strict_manifest_inventory(
        ladder_dir,
        load_json(ladder_dir / str(ladder_source["manifest"])),
        expected_output_count=529,
    )
    rung_dir = ladder_dir / str(ladder_source["rung_directory"])
    _strict_manifest_inventory(
        rung_dir,
        load_json(rung_dir / str(ladder_source["rung_manifest"])),
        expected_output_count=103,
    )


def validate_config(
    config: Mapping[str, Any], *, verify_hashes: bool = True,
    require_live: bool = False,
) -> None:
    require(config.get("schema") == SCHEMA, "unexpected N3C live config schema")
    require(
        config.get("experiment_id")
        == "ue_n3c_oai_ul_cold_attach_refinement_live_v1",
        "N3C live experiment identity drift",
    )
    require(
        config["paths"]["output_root"]
        == "rl_agent/experiments/ue_n3c_oai_ul_cold_attach_refinement_live_v1",
        "N3C live output root drift",
    )
    validate_output_contract(config)
    require(n2.sha256(CANONICAL_PLAN_CONFIG) == CANONICAL_PLAN_CONFIG_SHA256,
            "sealed N3C PREPARE config drift")
    canonical = load_json(CANONICAL_PLAN_CONFIG)
    require(_scientific_projection(config) == _scientific_projection(canonical),
            "N3C live scientific contract differs from sealed PREPARE contract")
    require(
        config["predecessors"].get("n3b_adjudication")
        == EXPECTED_N3B_ADJUDICATION,
        "sealed N3B adjudication evidence drift",
    )

    normalized = json.loads(json.dumps(config))
    normalized["schema"] = plan.SCHEMA
    normalized["experiment_id"] = "ue_n3c_oai_ul_cold_attach_refinement_v1"
    plan.validate_config(
        normalized, verify_hashes=verify_hashes, require_live=require_live,
    )
    required_live_path = "rl_agent/ue_n3c_oai_ul_cold_attach_refinement_live_v1.py"
    matches = [
        row for row in config["runtime_seals"] if row.get("path") == required_live_path
    ]
    require(len(matches) == 1, "N3C live runner seal is absent or duplicated")
    if verify_hashes:
        require(
            n2.sha256(plan.resolve_repo_path(required_live_path))
            == matches[0]["sha256"],
            "N3C live runner seal drift",
        )
        verify_strict_predecessor_completeness(config)


def campaign_plan_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    return plan.campaign_plan_rows(config)


def classify_repetition(summary: Mapping[str, Any]) -> dict[str, Any]:
    classified = dict(plan.classify_repetition(summary))
    attach_nonconfirmation = (
        classified.get("classified_outcome") == "COLD_ATTACH_FAILED"
    )
    joint_confirmation = (
        classified.get("joint_candidate_confirmation_pass") is True
    )
    classified.update({
        "post_restore_clean_reattach_evaluated": (
            False if attach_nonconfirmation else None
        ),
        "candidate_causal_attach_failure_confirmed": False,
        "attach_failure_evidence_role": (
            "VALID_OPERATIONAL_SCREEN_NONCONFIRMATION_NOT_CANDIDATE_CAUSAL_OR_"
            "PHYSICAL_BOUNDARY_PROOF"
            if attach_nonconfirmation else None
        ),
        "review_before_next_action_required": not joint_confirmation,
        "target_mapping_promoted": False,
        "numeric_bound_promoted": False,
        "connectivity_bound_promoted": False,
        "usable_service_bound_promoted": False,
        "operational_bound_promoted": False,
    })
    return classified


def aggregate_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(len(results) == 3, "N3C requires three completed repetition results")
    require(
        all(row.get("evidence_valid_for_aggregation") is True for row in results),
        "N3C contains invalid repetition evidence",
    )
    attach_passes = sum(bool(row.get("cold_attach_gate_pass")) for row in results)
    service_passes = sum(
        bool(row.get("authoritative_service_gate_pass")) for row in results
    )
    snr_passes = sum(bool(row.get("achieved_snr_gate_pass")) for row in results)
    joint_passes = sum(
        bool(row.get("joint_candidate_confirmation_pass")) for row in results
    )
    return {
        "status": CAMPAIGN_PASSED if joint_passes == 3 else CAMPAIGN_NOT_3_OF_3,
        "cold_attach_passes": attach_passes,
        "cold_attach_3_of_3_pass": attach_passes == 3,
        "authoritative_service_gate_passes": service_passes,
        "authoritative_service_gate_3_of_3_pass": service_passes == 3,
        "achieved_snr_consistency_passes": snr_passes,
        "achieved_snr_consistency_3_of_3_pass": snr_passes == 3,
        "joint_candidate_confirmation_passes": joint_passes,
        "joint_candidate_confirmation_3_of_3_pass": joint_passes == 3,
        "valid_nonconfirming_outcomes_retained": 3 - joint_passes,
        "operational_screen_attach_nonconfirmations": 3 - attach_passes,
        "candidate_causal_attach_failure_confirmed": False,
        "review_before_next_action_required": joint_passes < 3,
        "cold_attach_bound_evaluated": True,
        "target_mapping_promoted": False,
        "numeric_bound_promoted": False,
        "connectivity_bound_promoted": False,
        "usable_service_bound_promoted": False,
        "operational_bound_promoted": False,
    }


class ColdAttachLiveRepetition(n3b.ColdAttachRepetitionRunner):
    """One fresh-RAN -3.0 dB cold-attach/service repetition."""

    def __init__(
        self, config_path: Path, output_dir: Path, *, repetition_index: int,
        n3b_live_proof: Mapping[str, Any],
        n3b_adjudication_proof: Mapping[str, Any],
        expectation_proof: Mapping[str, Any],
    ) -> None:
        validate_output_contract(load_json(config_path.resolve()))
        calibration.RungRunner.__init__(
            self,
            config_path,
            output_dir,
            rung_index=0,
            command_db=plan.CANDIDATE_COMMAND_DB,
            clean_control_proof=None,
        )
        self.repetition_index = int(repetition_index)
        self.n3b_live_proof = dict(n3b_live_proof)
        self.n3b_adjudication_proof = dict(n3b_adjudication_proof)
        self.expectation_proof = dict(expectation_proof)
        self.application_count = 0
        self.restore_application_count = 0
        self.candidate_baked_config_verified = False
        self.startup_channel_runtime_verified = False
        self.source_hashes_before: dict[str, str] = {}
        self.ue_launch_monotonic_ns: int | None = None
        self.attach_gate: dict[str, Any] | None = None
        self.runtime_execution_attempted = False
        self.socket_execution_attempted = False

    def verify_dependencies(self) -> None:
        validate_config(self.config, verify_hashes=True, require_live=True)
        observed_n3b = plan.verify_n3b_live(
            self.config,
            proof_path=self.output_dir / "n3b_live_evidence_predecessor.json",
        )
        observed_review = plan.verify_n3b_adjudication(
            self.config,
            proof_path=self.output_dir / "n3b_adjudication_predecessor.json",
        )
        observed_ladder = plan.verify_command_ladder_expectation(
            self.config,
            proof_path=self.output_dir / "command_ladder_expectation_predecessor.json",
        )
        for observed, frozen, name in (
            (observed_n3b, self.n3b_live_proof, "N3B live evidence"),
            (observed_review, self.n3b_adjudication_proof, "N3B adjudication"),
            (observed_ladder, self.expectation_proof, "command-ladder expectation"),
        ):
            require(
                observed.get("manifest_sha256")
                == frozen.get("manifest_sha256")
                if "manifest_sha256" in observed
                else observed.get("campaign_manifest_sha256")
                == frozen.get("campaign_manifest_sha256"),
                f"{name} changed between repetitions",
            )
        n2.atomic_json(self.output_dir / "runtime_seals.json", {
            "status": "MATCHED",
            "observed_at": n2.utc_now(),
            "files": [
                {
                    "path": row["path"],
                    "expected_sha256": row["sha256"],
                    "observed_sha256": n2.sha256(self.path(row["path"])),
                }
                for row in self.config["runtime_seals"]
            ],
        })

    def materialize_configs(self) -> tuple[Path, Path]:
        paths = self.config["paths"]
        conf_root = self.path(paths["oai_ran_conf"])
        sources = {
            "gnb_base": conf_root / paths["gnb_base_config"],
            "ue_base": conf_root / paths["ue_base_config"],
            "channel": conf_root / paths["channel_config"],
        }
        self.source_hashes_before = {
            key: n2.sha256(path) for key, path in sources.items()
        }
        gnb_base = sources["gnb_base"].read_text(encoding="utf-8")
        ue_base = sources["ue_base"].read_text(encoding="utf-8")
        channel = n3b.rewrite_channel_noise(
            sources["channel"].read_text(encoding="utf-8"),
            self.config["startup_channel"],
        )
        require(
            n3b.configured_channel_values(channel) == self.config["startup_channel"],
            "run-local startup channel verification failed",
        )
        require("noise_power_dBFS" not in channel,
                "global noise_power_dBFS must remain unset")
        expected_imsi = str(self.config["radio"]["expected_imsi"])
        require(len(re.findall(r"(?m)^\s*uicc\d+\s*=\s*\{", ue_base)) == 1,
                "effective UE config is not single-UE")
        require(
            re.findall(r'(?m)^\s*imsi\s*=\s*"([0-9]+)"\s*;', ue_base)
            == [expected_imsi],
            "effective UE IMSI mismatch",
        )
        marker = '@include "channelmod_rfsimu_LEO_satellite.conf"'
        require(marker in ue_base, "UE base config lacks expected channel include")
        runtime = self.output_dir / "runtime"
        runtime.mkdir()
        channel_path = runtime / "effective_channel_cold_attach_minus3p0.conf"
        gnb_path = runtime / "effective_gnb_cold_attach_minus3p0.conf"
        ue_path = runtime / "effective_ue_cold_attach_minus3p0.conf"
        n2.atomic_text(channel_path, channel)
        n2.atomic_text(gnb_path, gnb_base + "\n\n" + channel + "\n")
        n2.atomic_text(ue_path, ue_base.replace(marker, channel))
        self.candidate_baked_config_verified = True
        n2.atomic_json(runtime / "config_hashes.json", {
            "source_before": self.source_hashes_before,
            "run_local": {
                "channel_sha256": n2.sha256(channel_path),
                "gnb_sha256": n2.sha256(gnb_path),
                "ue_sha256": n2.sha256(ue_path),
            },
            "startup_channel": n3b.configured_channel_values(channel),
            "candidate_baked_before_ue_launch": True,
            "source_paths_written": False,
        })
        return gnb_path, ue_path

    def verify_required_recovery(
        self, baseline_count: int, receiver_baseline_count: int,
    ) -> Mapping[str, Any]:
        return plan.verify_post_restore_recovery_fail_closed(
            self, baseline_count, receiver_baseline_count,
        )

    def write_manifest_terminal(self, status: str, summary: Mapping[str, Any]) -> None:
        classification = classify_repetition(summary)
        if classification["evidence_valid_for_aggregation"]:
            if classification["joint_candidate_confirmation_pass"]:
                final_status = REP_PASSED
            elif classification["classified_outcome"] == "COLD_ATTACH_FAILED":
                final_status = REP_ATTACH_FAILED
            elif classification["classified_outcome"] == (
                "ACHIEVED_SNR_OUTSIDE_CONSISTENCY_BAND"
            ):
                final_status = REP_SNR_MISMATCH
            else:
                final_status = REP_SERVICE_FAILED
        elif status == RESTORE_FAILED:
            final_status = RESTORE_FAILED
        elif status == FAILED:
            final_status = FAILED
        else:
            final_status = REP_UNCONFIRMED
        augmented = {
            **dict(summary),
            "status": final_status,
            "engine_status": status,
            "repetition_index": self.repetition_index,
            "commanded_noise_power_db": plan.CANDIDATE_COMMAND_DB,
            "outcome_classification": classification,
            **classification,
            "ran_epoch_id": self.ran_epoch_id,
            "control_session_id": self.control_session_id,
            "candidate_application_count": self.application_count,
            "restore_application_count": self.restore_application_count,
            "runtime_executed": self.runtime_execution_attempted,
            "socket_executed": self.socket_execution_attempted,
            "live_execution_attempted": (
                self.runtime_execution_attempted
                or self.socket_execution_attempted
            ),
            "review_required": True,
            "target_mapping_promoted": False,
            "numeric_bound_promoted": False,
            "connectivity_bound_promoted": False,
            "usable_service_bound_promoted": False,
            "operational_bound_promoted": False,
        }
        n2.atomic_json(
            self.output_dir / EXPECTED_OUTPUT["repetition_summary"], augmented,
        )
        excluded = {"manifest.json", "FAILED.json"}
        files = []
        for path in sorted(self.output_dir.rglob("*")):
            if (
                not path.is_file()
                or path.name in excluded
                or path.name.startswith("UE_N3C_")
            ):
                continue
            files.append({
                "path": str(path.relative_to(self.output_dir)),
                "bytes": path.stat().st_size,
                "sha256": n2.sha256(path),
            })
        manifest_path = self.output_dir / EXPECTED_OUTPUT["manifest"]
        n2.atomic_json(manifest_path, {
            "schema": "scenesense.ue_n3c_cold_attach_repetition_manifest.v1",
            "status": final_status,
            "repetition_index": self.repetition_index,
            "commanded_noise_power_db": plan.CANDIDATE_COMMAND_DB,
            "config_sha256": n2.sha256(self.config_path),
            "runner_sha256": n2.sha256(Path(__file__).resolve()),
            "engine_runner_sha256": n2.sha256(Path(calibration.__file__).resolve()),
            "ran_epoch_id": self.ran_epoch_id,
            "control_session_id": self.control_session_id,
            "candidate_application_count": self.application_count,
            "restore_application_count": self.restore_application_count,
            "n3b_live_manifest_sha256": self.n3b_live_proof["manifest_sha256"],
            "n3b_adjudication_manifest_sha256": self.n3b_adjudication_proof[
                "manifest_sha256"
            ],
            "command_ladder_manifest_sha256": self.expectation_proof[
                "campaign_manifest_sha256"
            ],
            "evidence_valid_for_aggregation": classification[
                "evidence_valid_for_aggregation"
            ],
            "joint_candidate_confirmation_pass": classification[
                "joint_candidate_confirmation_pass"
            ],
            "review_required": True,
            "operational_bound_promoted": False,
            "outputs": files,
        })
        terminal = {**augmented, "manifest_sha256": n2.sha256(manifest_path)}
        name = "FAILED.json" if final_status == FAILED else f"{final_status}.json"
        n2.atomic_json(self.output_dir / name, terminal)

    def run(self) -> int:
        n2.atomic_json(self.output_dir / "resolved_config.json", {
            **self.config,
            "resolved_repetition": {
                "repetition_index": self.repetition_index,
                "commanded_noise_power_db": plan.CANDIDATE_COMMAND_DB,
            },
        })
        previous_handlers: dict[signal.Signals, Any] = {}

        def terminate(signum: int, _frame: Any) -> None:
            raise LiveRefinementFailure(
                f"received termination signal {signal.Signals(signum).name}"
            )

        for caught in (signal.SIGTERM, signal.SIGHUP):
            previous_handlers[caught] = signal.getsignal(caught)
            signal.signal(caught, terminate)
        try:
            self.preflight()
            gnb_config, ue_config = self.materialize_configs()
            self.runtime_execution_attempted = True
            self.start_ran(gnb_config, ue_config)
            attached = self.wait_cold_attach()
            self.socket_execution_attempted = True
            n2.wait_tcp(int(self.config["actuator"]["telnet_port"]), 15)
            model_index = self.open_and_validate_candidate_telnet()
            if not attached:
                self.restore_clean_once(model_index)
                self.write_command_log()
                self.cleanup(strict=True)
                source = self.source_integrity()
                cleanup = load_json(self.output_dir / "cleanup_report.json")
                summary = {
                    "status": REP_ATTACH_FAILED,
                    "attach_gate": self.attach_gate,
                    "service_tail": None,
                    "service_window": None,
                    "transport": None,
                    "observed_cold_achieved_pusch_snr_db_p05": None,
                    "observed_cold_achieved_pusch_snr_db_p50": None,
                    "observed_cold_achieved_pusch_snr_db_p95": None,
                    "clean_recovery": {
                        "status": "NOT_APPLICABLE_NO_PDU_SESSION",
                        "passed": None,
                    },
                    "post_restore_clean_reattach_evaluated": False,
                    "candidate_causal_attach_failure_confirmed": False,
                    "attach_failure_evidence_role": (
                        "VALID_OPERATIONAL_SCREEN_NONCONFIRMATION_NOT_CANDIDATE_"
                        "CAUSAL_OR_PHYSICAL_BOUNDARY_PROOF"
                    ),
                    "review_before_next_action_required": True,
                    "hard_loss_reason": None,
                    "candidate_baked_config_verified": self.candidate_baked_config_verified,
                    "startup_channel_runtime_verified": self.startup_channel_runtime_verified,
                    "candidate_application_count": self.application_count,
                    "restore_application_count": self.restore_application_count,
                    "clean_restore_verified": self.restored,
                    "source_oai_configs_unchanged": source["unchanged"],
                    "cleanup_clean": cleanup.get("clean") is True,
                }
                self.write_manifest_terminal(REP_ATTACH_FAILED, summary)
                return 0 if classify_repetition(summary)[
                    "evidence_valid_for_aggregation"
                ] else 1

            self.start_telemetry()
            time.sleep(0.75)
            self.start_probe()
            self.establish_clean_lead()
            tail_start = time.monotonic_ns()
            tail_start_wall = time.time_ns()
            frozen_duration_ns = 60_000_000_000
            frozen_end = tail_start + frozen_duration_ns
            frozen_end_wall = tail_start_wall + frozen_duration_ns
            tail: dict[str, Any] | None = None
            service_loss_status: str | None = None
            try:
                self.wait_for(60.0, enforce_silence=True)
                tail = calibration.summarize_tail(
                    self.live_csv.snapshot() if self.live_csv else [],
                    self.live_mcs.snapshot() if self.live_mcs else [],
                    start_ns=tail_start,
                    end_ns=frozen_end,
                    expected_rnti=int(self.current_rnti),
                    minimum_pusch=int(
                        self.config["rung"]["minimum_service_pusch_samples"]
                    ),
                    minimum_mcs=int(
                        self.config["rung"]["minimum_service_mcs_samples"]
                    ),
                    required_mcs_table=int(
                        self.config["analysis"]["scheduler_required_mcs_table"]
                    ),
                    required_force_mcs=int(
                        self.config["analysis"]["scheduler_required_force_ul_mcs"]
                    ),
                )
                tail.update({
                    "start_wall_time_ns": tail_start_wall,
                    "end_wall_time_ns": frozen_end_wall,
                    "frozen_duration_ns": frozen_duration_ns,
                })
            except calibration.HardServiceLoss as exc:
                self.hard_loss_reason = str(exc)
                service_loss_status = calibration.classify_service_loss_reason(str(exc))

            self.restore_clean_once(model_index)
            pusch_baseline = self.live_csv.count() if self.live_csv else 0
            if self.event_tail is not None:
                self.event_tail.poll()
            receiver_baseline = self.event_tail.accepted_count if self.event_tail else 0
            recovery = self.verify_required_recovery(
                pusch_baseline, receiver_baseline,
            )
            transport = self.finish_probe(
                allow_partial_sender=service_loss_status is not None,
            )
            service = None
            if tail is not None:
                service = calibration.evaluate_tail_service(
                    self.output_dir / "traffic/sender.csv",
                    self.output_dir / "traffic/receiver_events.jsonl",
                    start_wall_ns=tail_start_wall,
                    end_wall_ns=frozen_end_wall,
                    fps=10.0,
                    expected_tail_frames=600,
                    expected_source_ip=self.config["transport_gates"][
                        "expected_source_ip"
                    ],
                    structural_integrity=bool(transport["integrity_gate"]),
                    gates=self.config["transport_gates"],
                )
                n2.atomic_json(
                    self.output_dir / "service_window_summary.json", service,
                )
            self.write_command_log()
            self.cleanup(strict=True)
            source = self.source_integrity()
            self.extract_ttracer()
            self.write_raw_limit_record()
            cleanup = load_json(self.output_dir / "cleanup_report.json")
            summary = {
                "status": service_loss_status or "COLD_ATTACH_SERVICE_WINDOW_CAPTURED",
                "attach_gate": self.attach_gate,
                "service_tail": tail,
                "service_window": service,
                "transport": transport,
                "clean_recovery": recovery,
                "hard_loss_reason": self.hard_loss_reason,
                "receiver_service_outage_detected": self.receiver_service_outage_detected,
                "candidate_baked_config_verified": self.candidate_baked_config_verified,
                "startup_channel_runtime_verified": self.startup_channel_runtime_verified,
                "candidate_application_count": self.application_count,
                "restore_application_count": self.restore_application_count,
                "clean_restore_verified": self.restored,
                "source_oai_configs_unchanged": source["unchanged"],
                "cleanup_clean": cleanup.get("clean") is True,
            }
            self.write_manifest_terminal(str(summary["status"]), summary)
            return 0 if classify_repetition(summary)[
                "evidence_valid_for_aggregation"
            ] else 1
        except (Exception, KeyboardInterrupt) as exc:
            self.best_effort_restore()
            try:
                self.write_command_log()
            except Exception:
                pass
            cleanup_errors = self.cleanup(strict=False)
            source = (
                self.source_integrity()
                if self.source_hashes_before else {"unchanged": False}
            )
            status = (
                RESTORE_FAILED
                if self.startup_channel_runtime_verified and not self.restored
                else FAILED
            )
            failure = {
                "status": status,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "attach_gate": self.attach_gate,
                "candidate_baked_config_verified": self.candidate_baked_config_verified,
                "startup_channel_runtime_verified": self.startup_channel_runtime_verified,
                "candidate_application_count": self.application_count,
                "restore_application_count": self.restore_application_count,
                "clean_restore_verified": self.restored,
                "source_oai_configs_unchanged": source.get("unchanged") is True,
                "cleanup_clean": not cleanup_errors,
                "cleanup_errors": cleanup_errors,
            }
            self.write_manifest_terminal(status, failure)
            return 1
        finally:
            for caught, previous in previous_handlers.items():
                signal.signal(caught, previous)


def verify_repetition_evidence(
    directory: Path, *, repetition_index: int, config_sha256: str,
    runner_sha256: str,
) -> dict[str, Any]:
    manifest_path = directory / "manifest.json"
    summary_path = directory / "repetition_summary.json"
    require(manifest_path.is_file() and summary_path.is_file(),
            f"repetition evidence is incomplete: {directory}")
    manifest, summary = load_json(manifest_path), load_json(summary_path)
    status = str(summary.get("status", ""))
    require(status in {
        REP_PASSED, REP_ATTACH_FAILED, REP_SERVICE_FAILED, REP_SNR_MISMATCH,
    }, f"repetition status is not valid evidence: {status}")
    terminal_path = directory / f"{status}.json"
    require(terminal_path.is_file(), "repetition terminal is absent")
    terminal = load_json(terminal_path)
    require(manifest.get("status") == terminal.get("status") == status,
            "repetition status binding mismatch")
    require(terminal.get("manifest_sha256") == n2.sha256(manifest_path),
            "repetition terminal/manifest hash mismatch")
    require(manifest.get("config_sha256") == config_sha256
            and manifest.get("runner_sha256") == runner_sha256,
            "repetition source seal mismatch")
    require(int(manifest.get("repetition_index", -1)) == repetition_index,
            "repetition identity mismatch")
    require(manifest.get("candidate_application_count") == 0
            and manifest.get("restore_application_count") == 1,
            "repetition command-count contract failed")
    for payload, label in (
        (manifest, "manifest"), (summary, "summary"), (terminal, "terminal"),
    ):
        require(
            math.isclose(
                float(payload.get("commanded_noise_power_db", math.nan)), -3.0,
            ),
            f"repetition {label} command identity mismatch",
        )
    require(terminal.get("evidence_valid_for_aggregation") is True,
            "repetition evidence is not valid for aggregation")
    plan._verify_manifest_inventory(directory, manifest)
    cleanup = load_json(directory / "cleanup_report.json")
    require(cleanup.get("clean") is True and not cleanup.get("errors"),
            "repetition cleanup is not clean")
    ran_epoch = str(manifest.get("ran_epoch_id", ""))
    control_session = str(manifest.get("control_session_id", ""))
    require(ran_epoch and control_session and ran_epoch != control_session,
            "repetition lacks distinct RAN/control identities")
    return {
        "status": "VERIFIED_N3C_REPETITION_EVIDENCE",
        "directory": str(directory),
        "repetition_index": repetition_index,
        "manifest_sha256": n2.sha256(manifest_path),
        "terminal_sha256": n2.sha256(terminal_path),
        "ran_epoch_id": ran_epoch,
        "control_session_id": control_session,
        "cold_attach_gate_pass": terminal["cold_attach_gate_pass"],
        "authoritative_service_gate_pass": terminal[
            "authoritative_service_gate_pass"
        ],
        "achieved_snr_gate_pass": terminal["achieved_snr_gate_pass"],
        "joint_candidate_confirmation_pass": terminal[
            "joint_candidate_confirmation_pass"
        ],
    }


class CampaignRunner:
    def __init__(self, config_path: Path, output_dir: Path) -> None:
        self.config_path = config_path.resolve()
        self.config = load_json(self.config_path)
        validate_output_contract(self.config)
        self.output_dir = output_dir.resolve()
        protected = [
            plan.resolve_repo_path(relative)
            for relative in PROTECTED_SOURCE_DIRECTORIES
        ]
        for source in protected:
            try:
                self.output_dir.relative_to(source)
            except ValueError:
                pass
            else:
                raise LiveRefinementFailure(
                    f"output directory is inside sealed source evidence: {source}"
                )
        if self.output_dir.exists():
            raise LiveRefinementFailure(
                f"create-only output already exists: {self.output_dir}"
            )
        self.output_dir.mkdir(parents=True)

    def write_plan(self, *, require_live: bool = False) -> list[dict[str, Any]]:
        validate_config(self.config, verify_hashes=True, require_live=require_live)
        n2.atomic_json(
            self.output_dir / EXPECTED_OUTPUT["resolved_config"], self.config,
        )
        rows = campaign_plan_rows(self.config)
        calibration.write_csv(
            self.output_dir / EXPECTED_OUTPUT["campaign_plan"], rows,
        )
        return rows

    def manifest_terminal(self, status: str, summary: Mapping[str, Any]) -> None:
        n2.atomic_json(
            self.output_dir / EXPECTED_OUTPUT["campaign_summary"], summary,
        )
        files = []
        for path in sorted(self.output_dir.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(self.output_dir)
            if relative == Path(EXPECTED_OUTPUT["manifest"]):
                continue
            if path.parent == self.output_dir and (
                path.name.startswith("UE_N3C_") or path.name == "FAILED.json"
            ):
                continue
            files.append({
                "path": str(relative),
                "bytes": path.stat().st_size,
                "sha256": n2.sha256(path),
            })
        manifest_path = self.output_dir / EXPECTED_OUTPUT["manifest"]
        n2.atomic_json(manifest_path, {
            "schema": "scenesense.ue_n3c_cold_attach_live_campaign_manifest.v1",
            "status": status,
            "config_sha256": n2.sha256(self.config_path),
            "runner_sha256": n2.sha256(Path(__file__).resolve()),
            "engine_runner_sha256": n2.sha256(Path(calibration.__file__).resolve()),
            "review_required": True,
            "target_mapping_promoted": False,
            "numeric_bound_promoted": False,
            "connectivity_bound_promoted": False,
            "usable_service_bound_promoted": False,
            "operational_bound_promoted": False,
            "outputs": files,
        })
        terminal = {
            **dict(summary),
            "status": status,
            "review_required": True,
            "target_mapping_promoted": False,
            "numeric_bound_promoted": False,
            "connectivity_bound_promoted": False,
            "usable_service_bound_promoted": False,
            "operational_bound_promoted": False,
            "manifest_sha256": n2.sha256(manifest_path),
        }
        name = EXPECTED_OUTPUT["failure"] if status == FAILED else f"{status}.json"
        n2.atomic_json(self.output_dir / name, terminal)

    def prepare(self) -> int:
        try:
            rows = self.write_plan(require_live=False)
            n3b_proof = plan.verify_n3b_live(
                self.config,
                proof_path=self.output_dir / "n3b_live_evidence_predecessor.json",
            )
            expectation = plan.verify_command_ladder_expectation(
                self.config,
                proof_path=self.output_dir / "command_ladder_expectation_predecessor.json",
            )
            pending = plan.pending_adjudication(
                self.config["predecessors"]["n3b_adjudication"]
            )
            review = None
            if not pending:
                review = plan.verify_n3b_adjudication(
                    self.config,
                    proof_path=self.output_dir / "n3b_adjudication_predecessor.json",
                )
            authority_ready = (
                self.config["authority"]["live_oai_run_authorized"] is True
                and self.config["authority"]["live_socket_execution_authorized"] is True
            )
            ready = not pending and authority_ready and review is not None
            status = PLAN_READY if ready else PLAN_BLOCKED
            summary = {
                "status": status,
                "runtime_executed": False,
                "socket_executed": False,
                "repetitions_planned": len(rows),
                "plan": rows,
                "n3b_live_predecessor": n3b_proof,
                "n3b_adjudication_predecessor": review,
                "command_ladder_expectation_predecessor": expectation,
                "n3b_adjudication_predecessor_pending": pending,
                "live_authority_ready": authority_ready,
                "live_execution_blocked": not ready,
                "cold_attach_bound_evaluated": False,
                "next": (
                    "EXPLICIT_EXECUTE_LIVE"
                    if ready
                    else "FINALIZE_ADJUDICATION_SEALS_AND_EXPLICIT_LIVE_AUTHORITY"
                ),
            }
            self.manifest_terminal(status, summary)
            return 0
        except (Exception, KeyboardInterrupt) as exc:
            self.manifest_terminal(FAILED, {
                "status": FAILED,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "runtime_executed": False,
                "socket_executed": False,
                "cold_attach_bound_evaluated": False,
            })
            return 1

    def execute(self) -> int:
        rows: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []
        proofs: list[dict[str, Any]] = []
        n3b_proof: dict[str, Any] | None = None
        review: dict[str, Any] | None = None
        expectation: dict[str, Any] | None = None
        ran_epochs: set[str] = set()
        control_sessions: set[str] = set()
        runtime_execution_attempted = False
        socket_execution_attempted = False
        repetitions_attempted = 0
        try:
            rows = self.write_plan(require_live=True)
            n3b_proof = plan.verify_n3b_live(
                self.config,
                proof_path=self.output_dir / "n3b_live_evidence_predecessor.json",
            )
            review = plan.verify_n3b_adjudication(
                self.config,
                proof_path=self.output_dir / "n3b_adjudication_predecessor.json",
            )
            expectation = plan.verify_command_ladder_expectation(
                self.config,
                proof_path=self.output_dir / "command_ladder_expectation_predecessor.json",
            )
            for row in rows:
                index = int(row["repetition_index"])
                directory = (
                    self.output_dir / "repetitions" / f"rep_{index:02d}_minus3p0_cold"
                )
                runner = ColdAttachLiveRepetition(
                    self.config_path,
                    directory,
                    repetition_index=index,
                    n3b_live_proof=n3b_proof,
                    n3b_adjudication_proof=review,
                    expectation_proof=expectation,
                )
                repetitions_attempted += 1
                try:
                    rc = runner.run()
                finally:
                    runtime_execution_attempted = (
                        runtime_execution_attempted
                        or bool(getattr(runner, "runtime_execution_attempted", False))
                    )
                    socket_execution_attempted = (
                        socket_execution_attempted
                        or bool(getattr(runner, "socket_execution_attempted", False))
                    )
                summary_path = directory / EXPECTED_OUTPUT["repetition_summary"]
                require(summary_path.is_file(),
                        f"repetition summary absent: {directory}")
                result = load_json(summary_path)
                results.append(result)
                require(rc == 0,
                        f"invalid repetition evidence at repetition {index}")
                proof = verify_repetition_evidence(
                    directory,
                    repetition_index=index,
                    config_sha256=n2.sha256(self.config_path),
                    runner_sha256=n2.sha256(Path(__file__).resolve()),
                )
                require(proof["ran_epoch_id"] not in ran_epochs,
                        "fresh RAN epoch identity was reused")
                require(proof["control_session_id"] not in control_sessions,
                        "control-session identity was reused")
                ran_epochs.add(proof["ran_epoch_id"])
                control_sessions.add(proof["control_session_id"])
                proofs.append(proof)
            aggregation = aggregate_results(results)
            status = str(aggregation["status"])
            summary = {
                "status": status,
                "runtime_executed": runtime_execution_attempted,
                "socket_executed": socket_execution_attempted,
                "live_execution_attempted": (
                    runtime_execution_attempted or socket_execution_attempted
                ),
                "repetitions_planned": 3,
                "repetitions_executed": len(results),
                "repetitions_attempted": repetitions_attempted,
                "fresh_ran_epoch_count": len(ran_epochs),
                "unique_control_session_count": len(control_sessions),
                **aggregation,
                "results": results,
                "repetition_evidence": proofs,
                "n3b_live_predecessor": n3b_proof,
                "n3b_adjudication_predecessor": review,
                "command_ladder_expectation_predecessor": expectation,
                "next": "REVIEW_N3C_EVIDENCE_NO_AUTOMATIC_BOUND_PROMOTION",
            }
            self.manifest_terminal(status, summary)
            return 0
        except (Exception, KeyboardInterrupt) as exc:
            self.manifest_terminal(FAILED, {
                "status": FAILED,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "repetitions_planned": len(rows),
                "repetitions_executed": len(results),
                "repetitions_attempted": repetitions_attempted,
                "results": results,
                "repetition_evidence": proofs,
                "n3b_live_predecessor": n3b_proof,
                "n3b_adjudication_predecessor": review,
                "command_ladder_expectation_predecessor": expectation,
                "cold_attach_bound_evaluated": False,
                "runtime_executed": runtime_execution_attempted,
                "socket_executed": socket_execution_attempted,
                "live_execution_attempted": (
                    runtime_execution_attempted or socket_execution_attempted
                ),
            })
            return 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--mode", choices=(PREPARE_ONLY, EXECUTE_LIVE), default=PREPARE_ONLY,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    runner = CampaignRunner(Path(args.config), Path(args.output_dir))
    return runner.prepare() if args.mode == PREPARE_ONLY else runner.execute()


if __name__ == "__main__":
    raise SystemExit(main())
