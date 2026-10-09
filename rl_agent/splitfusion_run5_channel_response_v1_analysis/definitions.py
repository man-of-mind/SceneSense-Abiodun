"""Analysis definitions of the ONE Run-5 channel-adaptation validation, FROZEN BEFORE ANY FRAME (registered by digest in
ACCEPTANCE.json). Pure constants and pure functions; nothing here reads live outcomes.

Channel: the pinned FADE_RECOVERY trace (seed 2026082104), replayed unchanged from step 0 at the first-capture gate.
Phases are fixed on the trace STEP INDEX (100 ms per step) from the design Markov states of that trace, so they cannot
depend on the actor. A decision belongs to the step of the causal SNR command it admitted (``active_command_id``).
"""
import math

STEP_MS = 100

# ---- phases on the FADE_RECOVERY design states (state runs verified against traces.csv by test_definitions)
# [start, end) step intervals
PHASES = (
    dict(name="P0_INITIAL_MIXED", start=0, end=361, role="descriptive: INT-dominated with ADV fades 242-252, 303-357"),
    dict(name="P1_GOOD_PRE", start=361, end=460, role="good reference: FAV 361-459"),
    dict(name="P2_MIXED", start=460, end=658, role="descriptive: INT with FAV 512-517/546-549/606-657, ADV 558-590"),
    dict(name="P3_DEGRADED_SUSTAINED", start=658, end=972,
         role="sustained degraded: no FAV; INT 658-745, ADV 746-787, INT 788-827, ADV 828-835, INT 836-866, ADV 867-914, INT 915-971"),
    dict(name="P4_RECOVERY_GOOD", start=972, end=1114, role="recovery: FAV 972-1113"),
    dict(name="P5_LATE", start=1114, end=10**9, role="descriptive: INT 1114-1219, FAV 1220-1229, INT 1230-1260, ADV from 1261"),
)
ADV_CORE = dict(name="P3b_ADV_CORE", start=746, end=915, role="sub-phase of P3 from the first to the end of the last ADV run")
# condition labels for 'by channel condition' tables: the design state of the decision's admitted step
CONDITIONS = ("FAVORABLE", "INTERMEDIATE", "ADVERSE")

# ---- transitions (the recurring FAV<->INT<->ADV transitions of the pinned trace are ALL analysed; these are primary)
PRIMARY_TRANSITIONS = (
    dict(name="T_DEG_ONSET", step=658, direction="deterioration", note="FAV 606-657 -> INT; start of P3"),
    dict(name="T_DEG", step=746, direction="deterioration", note="first ADV run of the sustained degraded block"),
    dict(name="T_REC", step=972, direction="recovery", note="INT -> FAV; start of P4"),
)
# every design-state change of the pinned trace inside the run is also analysed (event table), same rules
RECURRING_TRANSITION_RULE = "every step s where design state(s) != state(s-1); direction by state order ADV<INT<FAV"

# ---- action response (frozen policy; response, not learning)
PRE_WINDOW_DECISIONS = 20          # pre-transition reference: the last 20 decisions with admitted step < transition step
BAND = (10, 90)                    # payload band percentiles of the pre-window
SUSTAIN = dict(window=5, required=4)   # sustained response: >= 4 of decisions d..d+4 outside the band, expected side
EXPECTED_SIDE = dict(deterioration="payload below pre-window P10", recovery="payload above pre-window P90")
SEARCH_LIMIT_DECISIONS = 40        # beyond this (or the next transition) -> NO_RESPONSE_WITHIN_WINDOW
ACTION_RESPONSE_CLASS = dict(QUICK_MAX_DECISIONS=3, SLOW_MIN_DECISIONS=11)   # counted from the first decision that
#                                    admitted the changed SNR; 0-3 QUICK, 4-10 MODERATE, >=11 SLOW, none -> NONE
# also reported: q_e4 and mode at the same decisions (unchanged mode is NOT 'no adaptation': q/payload can move in mode 11)

# ---- delays separated (all CLOCK_MONOTONIC_RAW unless stated)
DELAYS = dict(
    command_to_effective="radio_trace scheduled (CLOCK_MONOTONIC) -> command ACK (CLOCK_MONOTONIC); per step",
    effective_to_admitted="ACK effective_raw_ns of the first command with step >= s_T -> state_commit of the first "
                          "decision admitting that command or later (observation delay)",
    admitted_to_action="that decision's action_open (decision availability); then decisions/time to first and "
                       "sustained action response",
    action_to_kpi="action response -> KPI recovery (rolling on-time)")

# ---- timeliness KPI
DEADLINE_MS = 170.0                # inclusive; missing reply = not on time
ROLLING_WINDOW_DECISIONS = 20      # trailing window (~5 s); a 10-decision window is reported alongside, labelled
KPI_RECOVERY = dict(reference="on-time fraction of all decisions in P1_GOOD_PRE (this run)", margin=0.10,
                    secondary_absolute=0.80, hold_decisions=10,
                    rule="first decision d >= d(T_REC) whose trailing-20 on-time fraction >= threshold and stays >= "
                         "threshold for the next 10 decisions")
KPI_RECOVERY_CLASS = dict(QUICK_MAX_DECISIONS=20, SLOW_MIN_DECISIONS=41)   # >41 or not within P4 -> SLOW / NOT_RECOVERED

# ---- attribution guards (no policy-failure claim without them)
ATTRIBUTION = dict(
    channel_feasibility="per phase: measured uplink duration (edge_reassembly_complete - ue_first_send) vs L2 bytes; "
                        "effective uplink rate; implied L2 byte budget = rate x (170 - median non-uplink time); share of "
                        "late decisions whose uplink alone exceeds the slack; whether smaller-payload decisions in the "
                        "same phase were timely. Derived, not a fixed-action measurement",
    runtime="per phase: GT ready (gt_published), prediction at service, scoring/return (result_sent -> ue_receipt) delays",
    scene="per phase: actor count, eligible vehicles/persons, camera SI, radar P40",
    comparator="the FAVORABLE_STABLE retention capture at the SAME trace-step range (same route, same time index) is "
               "attached as a DESCRIPTIVE scene-matched context only; different session, traffic and channel",
    one_route="one route, one trace, one session: no causal claim")

# ---- quality (agent vs full-local on identical retained frames; never conditioned on success in the main result)
TOLERANCE_M = dict(vehicle=1.0, person=1.2)
RETENTION_SHARES = (0.80, 0.85, 0.90, 0.95)      # DESCRIPTIVE shares, not acceptance gates
QUALITY_WINDOW_RETAINED = 5                      # short-window trend over consecutive retained samples (~25 decisions)
BOOTSTRAP = dict(block=5, resamples=10000, seed=20261006, ci=0.95)
QUALITY_RULES = dict(
    recall="within-tolerance hits / eligible GT (misses count); conservative CDF puts misses at +inf",
    retention_pooled="sum agent hits / sum full-local hits over the same frames (pooled), plus absolute difference",
    per_frame_ratio="agent/full-local recall per frame where full-local recall > 0; full-local == 0 reported separately "
                    "with absolute outcomes; no eligible GT -> excluded from that class but kept in latency accounting",
    precision="TP/(TP+FP) at the 3 m gate; segmentation pooled IoU sum I / sum U",
    views="ALL_SCORED (main) and ON_TIME_ONLY (secondary); late/missed never hidden",
    small_n="person n small -> intervals and counts only; never pass/fail")

# ---- bytes (identical retained frames, same layer for both branches)
BYTE_LAYERS = dict(
    L1_codec_payload="ExecutionEnvelopeV3.inner_payload (agent: from unpack_sfd4 of the transmitted wire; baseline: "
                     "codec.encode output of the offline re-encode = REPLAY baseline.payload_bytes)",
    L2_application_frame="complete SFD4 frame = head + SFD1-v2 context extension + v3 action envelope + inner + "
                         "SHA-256 (agent: transmitted wire = live payload_bytes; baseline: the same frame re-packed with "
                         "the baseline inner payload and baseline action identity, else agent L2 - agent L1 + baseline L1 "
                         "if re-packing is refused; the method used is reported)",
    L3_packetized="phase2_map_sharing.transport.chunk_payload(L2, chunk_bytes=runtime.udp_chunk_bytes): sum of UDP "
                  "payload bytes (incl. chunk headers) and datagram count; L3_ip adds 28 B IPv4+UDP per datagram (inner "
                  "IP before GTP-U). GTP-U/PDCP/RLC/MAC/radio overhead NOT measured",
    full_local="no feature-offload traffic; its FP32 tensor size is NEVER reported as transmitted bytes",
    labels="baseline sizes = OFFLINE_REENCODED; baseline network latency = UNMEASURED",
    whole_run_agent_traffic="actual transmitted SFD4 bytes and datagrams of every transmitted frame in the run "
                            "(policy decisions, holds and fallbacks), separately from the retained-sample comparison",
    quantiles="P50/P95/P99 with n; P99 on ~100 samples (and any channel subset) labelled SPARSE_TAIL_ESTIMATE")


def phase_of(step):
    for p in PHASES:
        if p["start"] <= step < p["end"]:
            return p["name"]
    raise ValueError(f"step {step} outside every phase")


def transition_steps(states):
    """Recurring transitions: every design-state change (states = list of state names by step)."""
    order = dict(ADVERSE=0, INTERMEDIATE=1, FAVORABLE=2)
    return [dict(step=s, from_state=states[s - 1], to_state=states[s],
                 direction="deterioration" if order[states[s]] < order[states[s - 1]] else "recovery")
            for s in range(1, len(states)) if states[s] != states[s - 1]]


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)
