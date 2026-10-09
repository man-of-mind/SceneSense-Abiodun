"""Exact radio identity for Run 4: OAI_N78_100MHZ_273PRB_4D5U_V1.

Run 3 ran on the legacy 40 MHz / 106 PRB / 7D2U radio. That radio is explicitly
rejected for this work: `oai_radio_baseline_100mhz_4d5u_v1.json` records the
legacy target-SNR mapping as
``CALIBRATED_ON_40MHZ_106PRB_7D2U_DO_NOT_REUSE_AS_100MHZ_EVIDENCE`` and the
Phase-14a binding sets ``legacy_mapping_permitted: false``.

Every identity a live Run-4 cell depends on is pinned here and verified three
times -- before preflight, before the scientific cells, and again at final
sealing -- so a rebuild, a config edit or a launcher swap between those points
is caught rather than silently absorbed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]

RADIO_PROFILE_ID = "OAI_N78_100MHZ_273PRB_4D5U_V1"
EXECUTION_TOKEN = "SPLITFUSION_OAI_100MHZ_4D5U_ATTACH"


class RadioBindingError(RuntimeError):
    """Raised when the live radio identity does not match the pinned one."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Pinned identities.
#
# ``authority`` names the independent file that already recorded the same
# digest, where one exists. A pin with authority "FIRST_PIN_RUN4" is recorded
# here for the first time and is corroborated only by the audit that created it.
# --------------------------------------------------------------------------

PINS: Mapping[str, dict[str, str]] = {
    # --- launcher and its runner -------------------------------------
    "launcher": {
        "path": "uplink_only_spatial_map_pipeline/run_splitfusion_oai_100mhz_4d5u_v1.sh",
        "sha256": "8e02f0913338a187bff1de24bfea09d7eca03607da0e383ef8ad4a005b64ce61",
        "authority": "rl_agent/configs/splitfusion_phase14a_campaign_binding_v1.json:launcher.sha256",
    },
    "launcher_runner": {
        "path": "rl_agent/splitfusion_phase14a_100mhz_calibration_v1.py",
        "sha256": "09fb82c0ba644a44bdf5fb5e5ad92269c1daa31ffbeee1fc0864d5b7ce688c93",
        "authority": "rl_agent/configs/splitfusion_phase14a_campaign_binding_v1.json:calibration.runner.sha256",
    },
    "launcher_runner_config": {
        "path": "rl_agent/configs/splitfusion_phase14a_100mhz_calibration_v1.json",
        "sha256": "ad541f71f5659e1bb08d7d2c48a45086dfbd5bf45e669b32adfb320ddcab5cd9",
        "authority": "rl_agent/configs/splitfusion_phase14a_campaign_binding_v1.json:calibration.config.sha256",
    },
    # --- radio locks and configs -------------------------------------
    "radio_lock": {
        "path": "rl_agent/configs/oai_radio_baseline_100mhz_4d5u_v1.json",
        "sha256": "fd604a37cfbc416412440dd447fe43cea9a94d3b65d69dd5e4e975de01a3dbf4",
        "authority": "rl_agent/configs/splitfusion_phase14a_campaign_binding_v1.json:capacity_evidence.artifacts.radio_lock.sha256",
    },
    "gnb_config_273prb": {
        "path": "OAI/openairinterface5g/targets/PROJECTS/GENERIC-NR-5GC/CONF/"
                "gnb.sa.band78.fr1.273PRB.scenesense_rfsim.conf",
        "sha256": "03fb7ac7b4df6ce88f5a2e3e7a26211b78ad89e7d313632ca088e6b364742ffb",
        "authority": "rl_agent/configs/splitfusion_phase14a_100mhz_calibration_v1.json:source_sha256.gnb_source",
    },
    "ue_config": {
        "path": "OAI/openairinterface5g/targets/PROJECTS/GENERIC-NR-5GC/CONF/ue.conf",
        "sha256": "b20bf2d9d8da9bd15b1f822836b7235dafa16b29f4e097e6ca3749729494c944",
        "authority": "rl_agent/configs/splitfusion_phase14a_100mhz_calibration_v1.json:source_sha256.ue_source",
    },
    "channel_config": {
        "path": "OAI/openairinterface5g/targets/PROJECTS/GENERIC-NR-5GC/CONF/"
                "channelmod_rfsimu.conf",
        "sha256": "a47ade413b36ecf66f6df666bead124974705636cb134dd09a539b34bce7d760",
        "authority": "rl_agent/configs/splitfusion_phase14a_100mhz_calibration_v1.json:source_sha256.channel_source",
    },
    # --- target-SNR mapping (Phase-14a, measured under THIS radio) ----
    "target_snr_mapping_json": {
        "path": "experiments/splitfusion_phase14a_100mhz_calibration_v1/"
                "20260904_phase14a_live_calibration_retry3/mapping.json",
        "sha256": "841ee69e53d7325570652a0f4baa7ae7f554a2204c4e775ce963a064cabd2677",
        "authority": "FIRST_PIN_RUN4",
    },
    "target_snr_mapping_csv": {
        "path": "experiments/splitfusion_phase14a_100mhz_calibration_v1/"
                "20260904_phase14a_live_calibration_retry3/target_to_rfsim_mapping.csv",
        "sha256": "f1cc9c48bd5cd55d2de2421d07c23dd3deff2bca3497e3a5d5e87aaa11ec2a3d",
        "authority": "FIRST_PIN_RUN4",
    },
    "target_snr_mapping_manifest": {
        "path": "experiments/splitfusion_phase14a_100mhz_calibration_v1/"
                "20260904_phase14a_live_calibration_retry3/manifest.json",
        "sha256": "5f14b43dedf7aa947598442ffd2823dea155476ad396b6665fc32e37ddb96091",
        "authority": "…/SPLITFUSION_PHASE14A_100MHZ_CALIBRATION_CAPTURE_COMPLETE_"
                     "PENDING_REPLAY:manifest_sha256",
    },
    # --- tracer chain -------------------------------------------------
    "t_messages": {
        "path": "OAI/openairinterface5g/common/utils/T/T_messages.txt",
        "sha256": "2f4945814ad2f47197816756819afc0d087156c499e8c746cf0d9a81653005c5",
        "authority": "rl_agent/configs/ue_n3_oai_ul_live_stage_v1.json:runtime_seals",
    },
    "t_messages_compiled_incgen": {
        "path": "OAI/openairinterface5g/common/utils/T/incgen/T_messages.txt.h",
        "sha256": "e5801830384c32cda8a805e21d58a43f380732f5ecff7e3553a99348aa23abc3",
        "authority": "FIRST_PIN_RUN4",
    },
    "t_messages_compiled_build": {
        "path": "OAI/openairinterface5g/cmake_targets/ran_build/build/common/utils/T/"
                "T_messages.txt.h",
        "sha256": "e5801830384c32cda8a805e21d58a43f380732f5ecff7e3553a99348aa23abc3",
        "authority": "FIRST_PIN_RUN4",
    },
    "tracer_multi": {
        "path": "OAI/openairinterface5g/common/utils/T/tracer/multi",
        "sha256": "455ff3083dfb30f82ffee941dac01f888b39705f9d3ffab2b019ff5430da9542",
        "authority": "rl_agent/configs/ue_n3_oai_ul_live_stage_v1.json:runtime_seals",
    },
    "tracer_record": {
        "path": "OAI/openairinterface5g/common/utils/T/tracer/record",
        "sha256": "517150245a1317c2e117c0ac3adde1ee097a3592658316fe5b40edcbd5235e61",
        "authority": "rl_agent/configs/ue_n3_oai_ul_live_stage_v1.json:runtime_seals",
    },
    "tracer_csv": {
        "path": "OAI/openairinterface5g/common/utils/T/tracer/csv",
        "sha256": "2b5e42dc64d88ed1ad19ebeb7c26b195e668fe2eb9bc9c3da95f81334731faaa",
        "authority": "rl_agent/configs/ue_n3_oai_ul_live_stage_v1.json:runtime_seals",
    },
    "tracer_replay": {
        "path": "OAI/openairinterface5g/common/utils/T/tracer/replay",
        "sha256": "cf6ccf11ad42b481dd3731dea02649e314fa4589b4c710d98ff4e8c06c034f70",
        "authority": "rl_agent/configs/ue_n3_oai_ul_live_stage_v1.json:runtime_seals",
    },
    "tracer_extract_config": {
        "path": "OAI/openairinterface5g/common/utils/T/tracer/extract_config",
        "sha256": "e724bd3f0a1327ed9afef50fb1a81703054e029a8e137689e4874fdfc013c898",
        "authority": "rl_agent/configs/ue_n3_oai_ul_live_stage_v1.json:runtime_seals",
    },
    "extractor": {
        "path": "scripts/ttracer_extract_csv_smoke.sh",
        "sha256": "82f23ae18136379c4ea2511744eb1ed932d74850851885584b8075f9f3cdb488",
        "authority": "rl_agent/configs/ue_n3_oai_ul_live_stage_v1.json:runtime_seals",
    },
    # --- executables ---------------------------------------------------
    # These two DIFFER from the ue_n3_oai_ul_live_stage_v1 seal (01489dfb… /
    # 7cdeee94…). They were rebuilt on 2026-08-25, after the 2026-08-03 edit to
    # gNB_scheduler_ulsch.c that added the SINR UL-MCS policy gate this
    # experiment depends on. The older seal predates that work, so it is NOT
    # treated as corroboration and the difference is recorded rather than
    # papered over.
    "nr_softmodem": {
        "path": "OAI/openairinterface5g/cmake_targets/ran_build/build/nr-softmodem",
        "sha256": "ebcd85f4c96cf6e3d0a1752d0047e8377811014be75cdeeef6334c982fcfab70",
        "authority": "FIRST_PIN_RUN4_REBUILT_AFTER_SINR_MCS_POLICY",
    },
    "nr_uesoftmodem": {
        "path": "OAI/openairinterface5g/cmake_targets/ran_build/build/nr-uesoftmodem",
        "sha256": "60ecc9a1d102e8b66871727a6a23da22977ff907fecc3870a8dc46414080c975",
        "authority": "FIRST_PIN_RUN4_REBUILT_AFTER_SINR_MCS_POLICY",
    },
    "libtelnetsrv": {
        "path": "OAI/openairinterface5g/cmake_targets/ran_build/build/libtelnetsrv.so",
        "sha256": "7c815fbfbd987b256c1992666dff142d683b5a548ba1c6f9a5a39f6fede118f7",
        "authority": "FIRST_PIN_RUN4",
    },
    # --- imported Run-3 implementation (unchanged) ---------------------
    "run3_tagged_sender": {
        "path": "rl_agent/ue_mcs_backlog_calibration_v1/tagged_sender.py",
        "sha256": "",  # resolved at verify time; see RESOLVED_AT_VERIFY
        "authority": "RESOLVED_AT_VERIFY",
    },
}

#: Pins whose digest is resolved from the committed tree at first verification
#: and then frozen into the run manifest, rather than hardcoded here.
RESOLVED_AT_VERIFY = {"run3_tagged_sender"}

# --------------------------------------------------------------------------
# The legacy radio, explicitly refused.
# --------------------------------------------------------------------------

LEGACY_RADIO_ID = "OAI_N78_40MHZ_106PRB_7D2U_LEGACY"
FORBIDDEN_PRB = 106
FORBIDDEN_BANDWIDTH_MHZ = 40
FORBIDDEN_LAUNCHERS: Mapping[str, str] = {
    "uplink_only_spatial_map_pipeline/run_track1_oai_default106_ttracer_10fps.sh":
        "4bd64a5992a50daeb33b5da45a41495350c91ef599f9b105fe4e7f67fc7c09da",
    "uplink_only_spatial_map_pipeline/run_track1_oai_default106_ttracer_10fps_v2.sh":
        "fd2f1dd9530ab35536dfaa9f69b25bce7ae44c9e8905dda1a24b96d7a24b2cf4",
}
FORBIDDEN_GNB_CONFIG = (
    "OAI/openairinterface5g/targets/PROJECTS/GENERIC-NR-5GC/CONF/"
    "gnb.sa.band78.fr1.106PRB.usrpb210.conf")
FORBIDDEN_LEGACY_MAPPING = (
    "rl_agent/experiments/oai_target_snr_replay_pilot_v1/20260822_live_01/"
    "target_to_rfsim_mapping.csv")

#: Environment names the launcher itself refuses, restated so the runner can
#: assert it is not exporting them before it ever calls the launcher.
FORBIDDEN_ENV = (
    "GNB_CONF", "GNB_CONF_DEFAULT", "UE_CONF", "UE_CONF_DEFAULT", "UE_PRB",
    "UE_NUMEROLOGY", "UE_BAND", "UE_DL_FREQ", "UE_SSB", "AWGN_PROFILE",
    "AWGN_NOISE_POWER_DB",
)

# --------------------------------------------------------------------------
# Radio parameters, from the radio lock.
# --------------------------------------------------------------------------

RADIO: Mapping[str, Any] = {
    "profile_id": RADIO_PROFILE_ID,
    "band": 78,
    "bandwidth_mhz": 100,
    "prb": 273,
    "numerology": 1,
    "subcarrier_spacing_khz": 30,
    "ue_frequency_hz": 3649260000,
    "ue_ssb": 516,
    "absolute_frequency_ssb": 641280,
    "downlink_slots": 4,
    "downlink_symbols": 6,
    "uplink_slots": 5,
    "uplink_symbols": 4,
    "ue_count": 1,
    "ue_interface": "oaitun_ue1",
    "ue_static_ip": "10.0.0.2",
    "ext_dn_ip": "192.168.70.135",
    "pdu_session_5qi": 6,
    "min_rxtxtime": 6,
    "mcs_policy": "sinr",
}

#: T ports the qualified launcher actually opens. The UE port is **2023**, not
#: the 2022 the legacy Run-3 config used; binding the wrong one yields an empty
#: UE trace and therefore no MCS and no backlog at all.
TELEMETRY_PORTS: Mapping[str, int] = {
    "gnb_port": 2021,
    "ue_port": 2023,
    "gnb_relay_port": 2121,
    "ue_relay_port": 2123,
}

TELNET: Mapping[str, Any] = {
    "host": "127.0.0.1",
    "port": 9090,
    "channel_model_name": "rfsimu_channel_ue0",
    "clean_restore_noise_power_db": -50.0,
    "command_granularity_db": 0.25,
}

# --------------------------------------------------------------------------
# The registered Phase-14a target-SNR mapping, measured under THIS radio.
# --------------------------------------------------------------------------

#: (commanded noise_power_dB, achieved median PUSCH SNR dB). Twelve anchors,
#: strictly monotonic, granularity 0.25 dB. Transcribed from the pinned
#: mapping.json and re-checked against it by :func:`verify`.
TARGET_SNR_ANCHORS: tuple[tuple[float, float], ...] = (
    (-13.0, 25.5), (-12.0, 23.5), (-11.0, 21.5), (-10.0, 19.5),
    (-9.0, 17.5), (-8.0, 15.5), (-7.0, 14.0), (-6.0, 12.0),
    (-5.0, 10.0), (-4.0, 8.5), (-3.0, 6.5), (-2.0, 5.0),
)
MAPPING_MEASURED_LOWER_DB = 5.0
MAPPING_MEASURED_UPPER_DB = 25.5
MAPPING_REQUIRED_LOWER_DB = 5.5
MAPPING_REQUIRED_UPPER_DB = 24.5
MAPPING_INTERPOLATION = "MONOTONIC_PIECEWISE_LINEAR_INVERSE"

#: Recorded verbatim so nobody later reads the mapping as more qualified than
#: it is. ``campaign_mapping_qualified: false`` in the Phase-14a artifacts refers
#: to the 288-cell campaign gate, which additionally required a four-profile
#: replay. The twelve anchor *measurements* passed all their observation gates
#: under this exact radio, and that is what Run 4 binds.
MAPPING_QUALIFICATION_NOTE = (
    "Phase-14a status is CALIBRATION_CAPTURE_COMPLETE_PENDING_FOUR_PROFILE_REPLAY "
    "with campaign_mapping_qualified=false and profile_replay_performed=false. "
    "Run 4 binds the twelve measured anchors as its actuator mapping under this "
    "radio and does NOT claim the 288-cell campaign qualification."
)

#: Run-3's 106-PRB anchors, listed only so a test can prove Run 4 does not use
#: any of them. Never an input.
FORBIDDEN_RUN3_ANCHORS: tuple[tuple[float, float], ...] = (
    (-2.25, 5.5), (-2.5, 6.0), (-3.0, 6.5), (-3.5, 7.5),
    (-4.0, 8.5), (-5.0, 10.0), (-8.0, 16.0), (-10.0, 19.5),
)


def anchors_for_interpolation() -> list[dict[str, float]]:
    """Anchors in the shape the inherited inverse interpolator expects."""
    return [{"noise_power_db": command, "achieved_median_pusch_snr_db": snr}
            for command, snr in TARGET_SNR_ANCHORS]


def _mapping_is_strictly_monotonic() -> bool:
    commands = [c for c, _ in TARGET_SNR_ANCHORS]
    snrs = [s for _, s in TARGET_SNR_ANCHORS]
    ascending_commands = all(b > a for a, b in zip(commands, commands[1:]))
    descending_snr = all(b < a for a, b in zip(snrs, snrs[1:]))
    return ascending_commands and descending_snr


def verify(stage: str, repo_root: Path | None = None) -> dict[str, Any]:
    """Verify every pinned identity. Raises on the first difference.

    Called before preflight, before the scientific cells, and at final sealing.
    """
    root = repo_root or ROOT
    files: dict[str, Any] = {}
    problems: list[str] = []

    for key, pin in PINS.items():
        path = root / pin["path"]
        if not path.is_file():
            files[key] = {"path": pin["path"], "present": False, "matches": False}
            problems.append(f"{key}: missing at {pin['path']}")
            continue
        observed = sha256_file(path)
        if key in RESOLVED_AT_VERIFY:
            files[key] = {"path": pin["path"], "present": True,
                          "observed_sha256": observed, "matches": True,
                          "authority": pin["authority"]}
            continue
        matches = observed == pin["sha256"]
        files[key] = {"path": pin["path"], "present": True,
                      "expected_sha256": pin["sha256"],
                      "observed_sha256": observed, "matches": matches,
                      "authority": pin["authority"]}
        if not matches:
            problems.append(
                f"{key}: {pin['path']} is {observed}, pinned {pin['sha256']}")

    # The T-tracer byte-compare constraint, in checkable form.
    incgen = files.get("t_messages_compiled_incgen", {}).get("observed_sha256")
    build = files.get("t_messages_compiled_build", {}).get("observed_sha256")
    compiled_consistent = bool(incgen) and incgen == build
    if not compiled_consistent:
        problems.append(
            "compiled T_messages copies disagree; both softmodems must be "
            "rebuilt before any tracer evidence is trusted")

    # The mapping constants must still describe the pinned mapping file.
    mapping_path = root / PINS["target_snr_mapping_json"]["path"]
    mapping_consistent = False
    if mapping_path.is_file():
        data = json.loads(mapping_path.read_text())
        observed_anchors = tuple(
            (float(a["noise_power_db"]), float(a["achieved_median_pusch_snr_db"]))
            for a in data["anchors"])
        mapping_consistent = (
            sorted(observed_anchors) == sorted(TARGET_SNR_ANCHORS)
            and str(data["radio_profile_id"]) == RADIO_PROFILE_ID
            and bool(data["gates"]["strictly_monotonic"])
            and float(data["gates"]["command_granularity_db"])
            == float(TELNET["command_granularity_db"]))
        if not mapping_consistent:
            problems.append(
                "TARGET_SNR_ANCHORS no longer describe the pinned mapping.json")

    if not _mapping_is_strictly_monotonic():
        problems.append("TARGET_SNR_ANCHORS are not strictly monotonic")

    # The legacy radio must not be reachable through this binding.
    legacy_present = {
        name: (root / name).is_file() for name in FORBIDDEN_LAUNCHERS}

    report = {
        "stage": stage,
        "radio_profile_id": RADIO_PROFILE_ID,
        "execution_token": EXECUTION_TOKEN,
        "files": files,
        "compiled_t_messages_consistent": compiled_consistent,
        "mapping_matches_pinned_file": mapping_consistent,
        "mapping_strictly_monotonic": _mapping_is_strictly_monotonic(),
        "mapping_qualification_note": MAPPING_QUALIFICATION_NOTE,
        "legacy_launchers_present_but_forbidden": legacy_present,
        "legacy_radio_id_rejected": LEGACY_RADIO_ID,
        "problems": problems,
        "verified": not problems,
    }
    if problems:
        raise RadioBindingError(
            f"radio binding failed at stage {stage!r}: " + "; ".join(problems))
    return report


def assert_no_forbidden_env(environ: Mapping[str, str]) -> None:
    """The launcher refuses these; fail before spawning rather than after."""
    present = [name for name in FORBIDDEN_ENV if environ.get(name)]
    if present:
        raise RadioBindingError(
            f"legacy radio override(s) exported: {present}. The qualified "
            f"launcher refuses them; unset before launching.")


if __name__ == "__main__":  # pragma: no cover - operator convenience
    import sys
    try:
        print(json.dumps(verify("manual"), indent=2))
    except RadioBindingError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
