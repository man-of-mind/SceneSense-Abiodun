"""M3.1 v11 GOOD_ALL target-SNR controller: versioned copy of the v3 controller reading the v11 contract, whose
registered constant schedule is 23.5 dB nominal -> RFsim noise_power_dB -12.0 (Phase-14A anchor 01, qualified in
experiments/splitfusion_m3_good_all_snr_qual_v1/20261008_run01).  Only the contract (and the record schema) differ;
the v3 header below describes the mechanism (its 21.5 dB / -11.0 values are the v3 schedule).

GOOD_ALL target-SNR controller: the retained Run-5 controller with a constant 21.5-dB schedule.

    python -m rl_agent.splitfusion_m3_route_b_v11.good_all_controller --controller ... --profile-id FAVORABLE_STABLE ...

Launched by the retained ``runtime_policy.start_target`` (CONTROLLER_MODULE substituted in the UE child). Changes:
  * ``--profile-id`` must be the carrier FAVORABLE_STABLE and is rewritten to ``GOOD_ALL`` so every radio-trace row
    names the schedule actually commanded;
  * ``ue_target_snr_cell_runtime_v1.prepare_sequence`` is replaced for this process: it first runs the ORIGINAL for
    the carrier profile (every pinned design/trace/digest check still executes), then returns the constant GOOD_ALL
    sequence (state 2, 21.5 dB every step, prefix of the registered length, frozen identity with the registered
    digest).
Everything else -- 100-ms schedule, mapping/inverse interpolation (21.5 -> -11.0), telnet command and response
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
            raise ValueError("GOOD_ALL controller received a foreign profile id: " + repr(profile_id))
        original(campaign=campaign, profile_id=C.ORIGINAL_PROFILE)      # every retained pin check still runs
        count = int(campaign["network"]["prefix_samples"])
        prefix, digest = C.schedule_prefix(count)
        if count != schedule["prefix_samples"] or digest != schedule["trace_sha256"]:
            raise ValueError("GOOD_ALL prefix differs from the registered schedule")
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
    print(json.dumps(dict(schema="splitfusion.m3_route_b_v11.good_all_controller.v1", wall_ns=time.time_ns(),
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
