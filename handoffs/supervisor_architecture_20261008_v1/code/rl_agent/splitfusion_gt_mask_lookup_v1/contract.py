"""Identity of the mask-lookup candidate v1 (narrow, in place, registered).

Diagnosis (valid_02): target admission (scorer TARGET request, ``integration_v2.ready.decode_target`` ->
``_admit``) validates both 768x432 masks with ``set(np.unique(mask)).issubset(labels)`` (a full sort): 2.62 ms per
admission on one pinned core. Candidate: an exact per-uint8 lookup table + reduction (``allowed[v] = v in labels``),
0.997 ms; type/length check first and unchanged, same exception and messages, no GT-eligibility change.
Changed file: exactly ``CONTRACT``; original preserved byte-exact under ``EVIDENCE``/original.
"""
import hashlib
from pathlib import Path

from rl_agent.splitfusion_gt_archive_capacity_v1 import contract as AC

CONTRACT = "rl_agent/splitfusion_numeric_gt_integration_v1/contract.py"
ORIGINAL = {CONTRACT: "57deee2c34e4e77acce25e4aadee218129c5b1451a42cdcef7ed71002df70bee"}
REPAIRED = {CONTRACT: "79b1d4c3b4ebf795d13583329b76fc75f6eb497e18aecf93aa2cd9eb6b07cdbb"}
EVIDENCE = "experiments/splitfusion_gt_mask_lookup_v1/20261006_v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def repaired(root):
    root = Path(root).resolve(strict=True)
    for relative, digest in REPAIRED.items():
        path = root / relative
        if not path.is_file() or path.is_symlink() or sha(path) != digest:
            raise ValueError("mask-lookup source differs from the registered candidate: " + relative)
    return dict(REPAIRED)


def substitution_ok(proof_sources, current_sources, root):
    """A retained proof is reusable iff every differing file is a registered ORIGINAL->REPAIRED transition of the
    archive-capacity repair or this candidate, and the current tree holds both repaired states."""
    if proof_sources == current_sources:
        return True
    transitions = {k: (AC.ORIGINAL[k], v) for k, v in AC.repaired(root).items()}
    transitions.update({k: (ORIGINAL[k], v) for k, v in repaired(root).items()})
    if set(proof_sources) != set(current_sources):
        return False
    differing = {k for k in current_sources if proof_sources[k] != current_sources[k]}
    return bool(differing) and all(k in transitions and (proof_sources[k], current_sources[k]) == transitions[k]
                                   for k in differing)
