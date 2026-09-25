"""Live runner for the Run-4 near-capacity UE MCS/backlog sweep.

This is a thin, additive wrapper. Every radio-lifecycle behaviour that Run 3
already qualified -- preflight, cold-RAN assertion, config materialisation,
gNB/UE start, attach, radio-path proof, telemetry, telnet actuation, upper
anchor calibration, traffic launch, per-cell capture, T-tracer extraction,
teardown and the cold-host proof -- is **inherited unchanged** from
``ue_mcs_backlog_calibration_v1.runner.Runner``.  Only the campaign plan, the
contract identity, the create-only guarantees and the protected-evidence guard
are new.

Nothing here writes inside the protected Run-3 campaign, and no file in the
Run-3 package is edited; its digests are recorded in this run's manifest.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from rl_agent.ue_mcs_backlog_calibration_v1 import runner as V3R
from rl_agent.ue_mcs_backlog_calibration_v1.runner import RunFailure, require, utc_now
from rl_agent.ue_mcs_backlog_near_capacity_v1 import contract as C
from rl_agent.ue_mcs_backlog_near_capacity_v1 import protected_evidence as PE

ROOT = V3R.ROOT
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config_v1.json"

STATUS_OK = "UE_MCS_BACKLOG_NEAR_CAPACITY_CAPTURED"
STATUS_PARTIAL = "UE_MCS_BACKLOG_NEAR_CAPACITY_PARTIAL"
STATUS_FAILED = "UE_MCS_BACKLOG_NEAR_CAPACITY_FAILED"

#: The upper-anchor calibration deliberately reuses the Run-3 *calibration*
#: tier (action 71, 6,229 B, 0.50 Mbps) through the inherited
#: ``calibrate_upper_anchor``. It exists only to put enough PUSCH on the air to
#: measure the noise-power -> SNR mapping without flooding the link, it emits no
#: tagged decision, and it enters no analysis. Reusing it keeps the anchor
#: procedure byte-identical to the one Run 3 qualified (-12.5 dB -> 25.0 dB
#: median PUSCH SNR over 412 samples).
CALIBRATION_TIER_NOTE = (
    "upper-anchor calibration reuses the Run-3 calibration tier (action 71, "
    "6,229 B); it produces no tagged decision and enters no analysis")


class Runner(V3R.Runner):
    """Run-3 radio lifecycle, Run-4 campaign plan."""

    def manifest(self, status: str, extra: Mapping[str, Any]) -> None:
        """Seal the run under the Run-4 contract identity.

        Deliberately not ``super().manifest``: that would stamp the Run-3
        contract id onto Run-4 evidence.
        """
        files = []
        for path in sorted(self.output_dir.rglob("*")):
            if path.is_file() and path.name != "manifest.json":
                files.append({"relative_path": str(path.relative_to(self.output_dir)),
                              "size_bytes": path.stat().st_size,
                              "sha256": V3R.n2.sha256(path)})
        self.out("manifest.json").write_text(json.dumps({
            "schema": self.config["schema"],
            "contract_id": C.CONTRACT_ID,
            "contract_version": C.CONTRACT_VERSION,
            "claim_boundary": C.CLAIM_BOUNDARY,
            "status": status, "utc": utc_now(),
            "config_sha256": V3R.n2.sha256(self.config_path),
            "source_hashes": C.resolved_source_hashes(ROOT),
            "calibration_tier_note": CALIBRATION_TIER_NOTE,
            **dict(extra), "files": files,
        }, indent=2) + "\n")

    def run(self) -> int:  # noqa: C901 - mirrors the qualified Run-3 shape
        import signal

        status = STATUS_FAILED
        failure: str | None = None

        def terminate(signum: int, _frame: Any) -> None:
            self.aborted = True
            raise RunFailure(f"received signal {signum}")

        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, terminate)

        protected_before = PE.require_unchanged("before_campaign", ROOT)
        PE.assert_outside_protected_run(self.output_dir, ROOT)

        tiers = C.resolve_load_tiers(ROOT)
        profiles = {p.profile_id: p for p in
                    C.resolve_profiles(ROOT, C.FRAMES_PER_CELL)}
        plan = C.build_cell_plan(
            tiers, ports=self.config["traffic"]["ports"],
            seed=int(self.config["campaign"]["cell_order_seed"]))
        plan_audit = C.audit_cell_plan(plan)
        cell_records: list[dict[str, Any]] = []
        calibration: dict[str, Any] = {}
        protected_after: dict[str, Any] = {}

        try:
            self.preflight()
            require(C.plan_is_registered_design(plan_audit),
                    f"cell plan is not the registered balanced design: {plan_audit}")
            require(all(p in profiles for p in C.CONTRAST_PROFILE_IDS),
                    "registered contrast profiles did not resolve")
            self.out("plan.json").write_text(json.dumps({
                "contract_id": C.CONTRACT_ID,
                "claim_boundary": C.CLAIM_BOUNDARY,
                "design": {
                    "permutations": [list(o) for o in C.PERMUTATIONS],
                    "permutation_labels": [C.permutation_label(o)
                                           for o in C.PERMUTATIONS],
                    "partitions": list(C.PARTITIONS),
                    "fit_permutations": [list(o) for o in C.FIT_PERMUTATIONS],
                    "validation_permutations": [list(o) for o
                                                in C.VALIDATION_PERMUTATIONS],
                    "contrast_profiles": list(C.CONTRAST_PROFILE_IDS),
                    "frames_per_block": C.FRAMES_PER_BLOCK,
                    "frames_per_cell": C.FRAMES_PER_CELL,
                    "expected_cells": C.EXPECTED_CELLS,
                    "expected_decisions": C.EXPECTED_DECISIONS,
                    "transient_decisions": C.TRANSIENT_DECISIONS,
                    "steady_state_decisions": C.STEADY_STATE_DECISIONS,
                },
                "plan_audit": plan_audit,
                "cells": [c.to_json() for c in plan],
                "tiers": [t.to_json() for t in tiers],
                "measured_capacity_mbps": dict(C.MEASURED_CAPACITY_MBPS),
                "capacity_source": C.CAPACITY_SOURCE,
                "expected_load_ratios": C.expected_load_ratios(),
                "cell_order_seed": self.config["campaign"]["cell_order_seed"],
                "payload_seed": self.config["campaign"]["payload_seed"],
                "source_hashes": C.resolved_source_hashes(ROOT),
                "oai_citations": dict(C.OAI_CITATIONS),
                "protected_evidence_before": protected_before,
                "calibration_tier_note": CALIBRATION_TIER_NOTE,
            }, indent=2) + "\n")

            cal_dir = self.output_dir / "calibration"
            cal_dir.mkdir(parents=True, exist_ok=False)
            cal_log: list[dict[str, Any]] = []
            self.assert_cold_ran("calibration")
            gnb_cfg, ue_cfg = self.materialize_configs(cal_dir)
            self.start_ran(gnb_cfg, ue_cfg, "calibration")
            self.wait_attach("calibration")
            self.verify_radio_path(cal_dir)
            self.start_telemetry("calibration")
            self.open_telnet(cal_dir)
            self.start_live_pusch("calibration")
            calibration = self.calibrate_upper_anchor(cal_dir, cal_log)
            (cal_dir / "command_log.json").write_text(
                json.dumps(cal_log, indent=2) + "\n")
            self.teardown_ran()

            for cell in plan:
                if self.aborted:
                    self.notes.append(f"campaign aborted before {cell.cell_id}")
                    break
                cell_records.append(self.run_cell(cell, profiles[cell.profile_id]))
            require(not self.aborted, "campaign was aborted by signal")

            status = (STATUS_OK
                      if all(r["status"] == "CAPTURED" for r in cell_records)
                      else STATUS_PARTIAL)
        except Exception as exc:  # noqa: BLE001
            failure = f"{type(exc).__name__}: {exc}"
        finally:
            self.notes.extend(self.teardown_ran())
            cold = self.final_cold_state()
            try:
                protected_after = PE.require_unchanged("after_campaign", ROOT)
            except PE.ProtectedEvidenceError as exc:
                protected_after = {"all_unchanged": False, "error": str(exc)}
                failure = failure or f"{type(exc).__name__}: {exc}"
                status = STATUS_FAILED
            self.manifest(status, {
                "failure": failure, "notes": self.notes,
                "calibration": calibration, "plan_audit": plan_audit,
                "cells": cell_records, "final_cold_state": cold,
                "protected_evidence_before": protected_before,
                "protected_evidence_after": protected_after,
            })
        print(json.dumps({
            "status": status, "failure": failure,
            "cells_captured": sum(1 for r in cell_records
                                  if r["status"] == "CAPTURED"),
            "cells_planned": len(plan),
            "output_dir": str(self.output_dir),
            "cold": cold.get("cold"),
            "protected_run_unchanged": protected_after.get("all_unchanged"),
        }, indent=2))
        return 0 if status == STATUS_OK else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    output = args.output_dir or (ROOT / config["paths"]["output_root"]
                                 / datetime.now().strftime("%Y%m%d_%H%M%S"))
    PE.assert_outside_protected_run(output, ROOT)
    PE.require_unchanged("before_mkdir", ROOT)
    # Create-only: an existing root is never reused, extended or overwritten.
    output.mkdir(parents=True, exist_ok=False)
    return Runner(args.config, output).run()


if __name__ == "__main__":
    raise SystemExit(main())
