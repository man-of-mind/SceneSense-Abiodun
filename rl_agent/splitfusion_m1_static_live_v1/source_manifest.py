"""Import-safe identity of the complete M1 static exact-source inventory.

Only path identity lives here.  Source digests remain sealed in each fresh
registration and are checked byte-for-byte by :mod:`source_verify`.  Keeping
this module data-only prevents edge startup verification from importing the
offline replay/geometry stack (and therefore SciPy).
"""

SCHEMA = "splitfusion.m1_static_live_v1.exact_source_manifest.v1"

# SHA-256 of json.dumps(sorted(exact_source_paths), separators=(",", ":"),
# ensure_ascii=True).  This is the retained, prerequisite-qualified 461-source
# set.  M1 launcher sources are independently sealed below so the prerequisite
# identity remains unchanged.
EXPECTED_SOURCE_COUNT = 461
EXPECTED_PATHS_SHA256 = "901b3e352f542c39c54ec0a27c5862279bad93c0131dc44818ea047ada3a7276"

# These are structural completeness checks in addition to the full path-set
# fingerprint.  They make omissions in the selection, retained runtime,
# retention, input preparation, and former SciPy-bearing geometry chain
# individually legible.
REQUIRED_SOURCES = (
    "rl_agent/splitfusion_run5_full_live_retention_v1/contract.py",
    "rl_agent/splitfusion_run5_full_live_retention_v1/entries.py",
    "rl_agent/splitfusion_run5_full_live_retention_v1/retention.py",
    "rl_agent/splitfusion_paired_retention_v2/observer.py",
    "rl_agent/splitfusion_paired_retention_v2/writer.py",
    "rl_agent/splitfusion_exact_gt_live_v2/contract.py",
    "rl_agent/splitfusion_numeric_gt_runtime_v1/live_contract.py",
    "rl_agent/splitfusion_numeric_gt_runtime_v1/replay.py",
    "rl_agent/splitfusion_live_gt_optimization_v1/source_geometry.py",
    "rl_agent/splitfusion_live_parity_repair_v1/source_geometry.py",
    "rl_agent/splitfusion_ue_input_prep_v2/candidate.py",
)

# Exact top-level launcher set sealed separately in registration.  It includes
# the fixed-selection seam and the import-safe verifier itself.
LAUNCHER_SOURCES = (
    "rl_agent/splitfusion_m1_static_live_v1/__init__.py",
    "rl_agent/splitfusion_m1_static_live_v1/contract.py",
    "rl_agent/splitfusion_m1_static_live_v1/deploy.py",
    "rl_agent/splitfusion_m1_static_live_v1/entries.py",
    "rl_agent/splitfusion_m1_static_live_v1/fixed.py",
    "rl_agent/splitfusion_m1_static_live_v1/gc_policy_live.py",
    "rl_agent/splitfusion_m1_static_live_v1/gc_seams.py",
    "rl_agent/splitfusion_m1_static_live_v1/host.py",
    "rl_agent/splitfusion_m1_static_live_v1/partition.py",
    "rl_agent/splitfusion_m1_static_live_v1/prove.py",
    "rl_agent/splitfusion_m1_static_live_v1/retention.py",
    "rl_agent/splitfusion_m1_static_live_v1/runner.py",
    "rl_agent/splitfusion_m1_static_live_v1/service_entry.py",
    "rl_agent/splitfusion_m1_static_live_v1/source_manifest.py",
    "rl_agent/splitfusion_m1_static_live_v1/source_verify.py",
    "rl_agent/splitfusion_m1_static_live_v1/test_fixed.py",
    "rl_agent/splitfusion_m1_static_live_v1/worker_entry.py",
)
