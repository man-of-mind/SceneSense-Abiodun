"""Data-only source inventory identity for slim-edge verification."""

SCHEMA = "splitfusion.m1_fixed_live_v2.exact_source_manifest.v1"
EXPECTED_SOURCE_COUNT = 461
EXPECTED_PATHS_SHA256 = "901b3e352f542c39c54ec0a27c5862279bad93c0131dc44818ea047ada3a7276"

REQUIRED_SOURCES = (
    "rl_agent/splitfusion_run5_full_live_retention_v1/contract.py",
    "rl_agent/splitfusion_run5_full_live_retention_v1/entries.py",
    "rl_agent/splitfusion_run5_full_live_retention_v1/retention.py",
    "rl_agent/splitfusion_paired_retention_v2/observer.py",
    "rl_agent/splitfusion_paired_retention_v2/writer.py",
    "rl_agent/splitfusion_exact_gt_live_v2/contract.py",
    "rl_agent/splitfusion_numeric_gt_runtime_v1/replay.py",
    "rl_agent/splitfusion_live_parity_repair_v1/source_geometry.py",
    "rl_agent/splitfusion_ue_input_prep_v2/candidate.py",
)

LAUNCHER_SOURCES = (
    "rl_agent/splitfusion_m1_fixed_live_v2/__init__.py",
    "rl_agent/splitfusion_m1_fixed_live_v2/child_ae32.py",
    "rl_agent/splitfusion_m1_fixed_live_v2/child_mode11.py",
    "rl_agent/splitfusion_m1_fixed_live_v2/contracts.py",
    "rl_agent/splitfusion_m1_fixed_live_v2/deploy.py",
    "rl_agent/splitfusion_m1_fixed_live_v2/edge_ae32.py",
    "rl_agent/splitfusion_m1_fixed_live_v2/edge_mode11.py",
    "rl_agent/splitfusion_m1_fixed_live_v2/entries.py",
    "rl_agent/splitfusion_m1_fixed_live_v2/fixed.py",
    "rl_agent/splitfusion_m1_fixed_live_v2/host_ae32.py",
    "rl_agent/splitfusion_m1_fixed_live_v2/host_common.py",
    "rl_agent/splitfusion_m1_fixed_live_v2/host_mode11.py",
    "rl_agent/splitfusion_m1_fixed_live_v2/partition_ae32.py",
    "rl_agent/splitfusion_m1_fixed_live_v2/partition_common.py",
    "rl_agent/splitfusion_m1_fixed_live_v2/partition_mode11.py",
    "rl_agent/splitfusion_m1_fixed_live_v2/prove.py",
    "rl_agent/splitfusion_m1_fixed_live_v2/runner_ae32.py",
    "rl_agent/splitfusion_m1_fixed_live_v2/runner_common.py",
    "rl_agent/splitfusion_m1_fixed_live_v2/runner_mode11.py",
    "rl_agent/splitfusion_m1_fixed_live_v2/source_manifest.py",
    "rl_agent/splitfusion_m1_fixed_live_v2/source_verify.py",
    "rl_agent/splitfusion_m1_fixed_live_v2/test_fixed.py",
)

