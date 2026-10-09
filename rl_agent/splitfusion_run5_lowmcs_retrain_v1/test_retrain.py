"""Focused checks for the low-MCS retraining adapter (CPU). Earlier evidence reused unchanged: the reward-trial
orchestrator/collector/trainer tests (``splitfusion_run5_reward_trials_v1.test_trials``) and the candidate's own
boundary/composition tests (``splitfusion_run5_uplink_lowmcs_v1.test_lowmcs``).

Run from abiodun/:  env -u PYTHONPATH CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 python3 -m unittest -v \
    rl_agent.splitfusion_run5_lowmcs_retrain_v1.test_retrain
"""
from __future__ import annotations

import functools
import json
import random
import tempfile
import unittest
from pathlib import Path

from rl_agent.splitfusion_hybrid_sac_run4_v1 import modeled_smoke_orchestrator as orch
from rl_agent.splitfusion_run5_full_v1 import support as S
from rl_agent.splitfusion_run5_reward_trials_v1 import collector as C
from rl_agent.splitfusion_run5_uplink_lowmcs_v1 import latency as LC
from rl_agent.splitfusion_run5_uplink_lowmcs_v1 import model as M

from . import campaign as CAMP
from . import registration as REG


@functools.lru_cache(maxsize=None)
def shared():
    return C.load_shared_sources(REG.ARM, "fit")


def _actions(n, seed=5):
    rng = random.Random(seed)
    for _ in range(n):
        mode = rng.choice((11, 11, 11, 10, 8, 6, 3, 0))
        lo, hi = S.BOUNDS[mode]
        yield mode, rng.randint(lo, hi)


def _run(latency, n, seed=17):
    col = C.TrialRun5CollectorV1(latency=latency, seed=seed, shared_sources=shared(), arm=REG.ARM)
    for i, (m, q) in enumerate(_actions(n)):
        col.collect_transition(orch.ModeledActionRequestV1(decision_ordinal=i, mode_id=m, q_e4=q,
                                                           source="STOCHASTIC_ACTOR", warmup_q_bin_index=None))
    return col.diagnostics


class RetrainAdapterTest(unittest.TestCase):
    def test_registration_binds_sources_evidence_and_rejects_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "REGISTRATION.json"
            path.write_text(json.dumps(REG.document()))
            REG.verify(path, seed=29)
            with self.assertRaises(REG.RegistrationError):
                REG.verify(path, seed=11)
            for key, value in (("source_digest", "0" * 64), ("lowmcs_sources", {"model.py": "x"}),
                               ("candidate", dict(REG.document()["candidate"], sha256="0" * 64))):
                doc = dict(REG.document(), **{key: value})
                path.write_text(json.dumps(doc))
                with self.assertRaises(REG.RegistrationError):
                    REG.verify(path, seed=17)

    def test_correction_is_frozen_at_the_fitted_values(self):
        c = CAMP.load_correction()
        fit = json.loads((REG.LOWMCS_RUN / "fit_01/FIT.json").read_text())["estimate"]
        self.assertEqual((c.kappa_ms_per_mb_mcs, c.lambda_low), (fit["kappa"], fit["lambda_low"]))
        self.assertEqual((c.fit_wire_max, c.fit_mcs_min), (145_556, 9))

    def test_comparator_and_deployed_actors_match_their_hashes(self):
        for spec in list(REG.COMPARATORS.values()) + [REG.DEPLOYED]:
            self.assertEqual(REG.sha(REG.ROOT / spec["path"]), spec["sha256"])

    def test_identity_wiring_reproduces_the_c0_training_environment_exactly(self):
        _, old = CAMP.components(corrected=False)
        ident = LC.CandidatePoolLatencyModelV1(REG.BINDING_02, shared()["model"], pool="TRAIN_GCV2", correction=M.IDENTITY)
        a, b = _run(old, 400), _run(ident, 400)
        strip = lambda d: {k: v for k, v in d.items() if k != "floors"}
        self.assertEqual([strip(x) for x in a], [strip(x) for x in b])

    def test_candidate_runs_flags_unsupported_and_keeps_the_lateness_rule(self):
        _, cand = CAMP.components()
        diag = _run(cand, 1500, seed=29)
        flagged = [d for d in diag if any(f in d["floors"] for f in CAMP.FLAGS)]
        for d in diag:
            low = d["prior_ul_mcs"] <= 15
            expect = cand.correction.flags(backlog_bytes=d["backlog_bytes"], wire_bytes=d["wire_bytes"],
                                           prior_ul_mcs=d["prior_ul_mcs"])
            self.assertEqual(tuple(f for f in d["floors"] if f in CAMP.FLAGS), expect)
            if not low:
                self.assertFalse(any(f in d["floors"] for f in CAMP.FLAGS))
            p_d = cand.transport.predict(pre_enqueue_backlog_bytes=float(d["backlog_bytes"]), wire_bytes=int(d["wire_bytes"]),
                                         prior_ul_mcs=int(d["prior_ul_mcs"])).on_time_probability
            p_b = cand._base_p[d["base_chain_id"]]
            self.assertAlmostEqual(d["uplink_late_probability"], min(1.0, max(0.0, 1.0 - p_d / p_b)), places=12)
        self.assertGreater(len(flagged), 0)                    # large-payload modes at low MCS reach the flags
        ex = CAMP.exposure(diag)
        self.assertEqual(ex["all"]["any"]["count"], len(flagged))


if __name__ == "__main__":
    unittest.main()
