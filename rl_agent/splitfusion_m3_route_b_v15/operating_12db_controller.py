"""M3.5 v15 OPERATING_12DB target-SNR controller: versioned copy of the v11 GOOD_ALL controller (itself the v3 mechanism)
reading the v15 contract, whose registered constant schedule is OPERATING_12DB = 12.0 dB nominal -> RFsim noise_power_dB -6.0
(the existing Phase-14A anchor 07 of the pinned 100-MHz mapping; historical measured PUSCH median 12.0 / P05 11.5 /
P95 12.0 dB, MCS 15 -- the MCS is NOT forced).  Only the contract (and the record schema) differ from v11; the
mechanism below is unchanged.

    python -m rl_agent.splitfusion_m3_route_b_v15.operating_12db_controller --controller ... --profile-id FAVORABLE_STABLE ...

Launched by the retained ``runtime_policy.start_target`` (CONTROLLER_MODULE substituted in the UE child). Changes:
  * ``--profile-id`` must be the carrier FAVORABLE_STABLE and is rewritten to ``OPERATING_12DB`` so every radio-trace row
    names the schedule actually commanded;
  * ``ue_target_snr_cell_runtime_v1.prepare_sequence`` is replaced for this process: it first runs the ORIGINAL for
    the carrier profile (every pinned design/trace/digest check still executes), then returns the constant OPERATING_12DB
    sequence (state 1 = INTERMEDIATE label, 12.0 dB every step, prefix of the registered length, frozen identity with the
    registered digest).
Everything else -- 100-ms schedule, mapping/inverse interpolation (12.0 -> -6.0), telnet command and response
validation, SKIP_OBSOLETE_NEVER_BURST, causal-SNR publication to the actor, the -50 dB restore and its read-back --
is the unchanged ``splitfusion_run5_live_avo_v1.snr_live.controller_main`` / ``live_pilot_target_snr_runtime``.
"""
import json
import sys
import time
from pathlib import Path

from . import contract as C


class ConstantSequence:
    """Same interface as DeterministicTargetSnrSequence: next_sample() -> (state_index, target_snr_db)."""

    def __init__(self, state_index, target_snr_db):
        self.state_index, self.target_snr_db = int(state_index), float(target_snr_db)
        self.samples = 0

    def next_sample(self):
        self.samples += 1
        return self.state_index, self.target_snr_db


def build_prepare(original, registration):
    schedule = registration["schedule"]

    def prepare_sequence(*, campaign, profile_id):
        if profile_id != schedule["schedule_id"]:
            raise ValueError("OPERATING_12DB controller received a foreign profile id: " + repr(profile_id))
        original(campaign=campaign, profile_id=C.ORIGINAL_PROFILE)      # every retained pin check still runs
        count = int(campaign["network"]["prefix_samples"])
        prefix, digest = C.schedule_prefix(count)
        if count != schedule["prefix_samples"] or digest != schedule["trace_sha256"]:
            raise ValueError("OPERATING_12DB prefix differs from the registered schedule")
        frozen = dict(profile_id=schedule["schedule_id"], trace_id=schedule["trace_id"], seed=int(schedule["seed"]),
                      trace_sha256=digest)
        return ConstantSequence(schedule["state_index"], schedule["target_snr_db"]), prefix, frozen

    return prepare_sequence


def rewrite_profile(argv):
    argv = list(argv)
    hits = [i for i, value in enumerate(argv) if value == "--profile-id"]
    if len(hits) != 1 or hits[0] + 1 >= len(argv) or argv[hits[0] + 1] != C.ORIGINAL_PROFILE:
        raise ValueError("expected exactly one --profile-id FAVORABLE_STABLE (carrier) argument")
    argv[hits[0] + 1] = C.SCHEDULE["schedule_id"]
    return argv


def main(argv=None):
    from rl_agent import ue_target_snr_cell_runtime_v1 as base
    from rl_agent.splitfusion_run5_live_avo_v1 import snr_live
    root = Path(__file__).resolve().parents[2]
    registration = C.channel_registration(root)
    argv = rewrite_profile(sys.argv[1:] if argv is None else argv)
    original = base.prepare_sequence
    base.prepare_sequence = build_prepare(original, registration)
    print(json.dumps(dict(schema="splitfusion.m3_route_b_v15.operating_12db_controller.v1", wall_ns=time.time_ns(),
                          schedule=registration["schedule"]["schedule_id"],
                          trace_id=registration["schedule"]["trace_id"],
                          target_snr_db=registration["schedule"]["target_snr_db"],
                          expected_rfsim_command_db=registration["schedule"]["expected_rfsim_command_db"])), flush=True)
    try:
        return snr_live.controller_main(argv)
    finally:
        base.prepare_sequence = original


if __name__ == "__main__":
    raise SystemExit(main())
