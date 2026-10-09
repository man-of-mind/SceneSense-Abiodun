"""Constrained objective (registered before training; see REGISTRATION.json for the frozen numbers).

Per decision t (scene s, executed (mode, q); D = 1[receipt <= 170 ms inclusive]):

  primary     r^P = D * Y,   Y = (0.4 Rv [if eligible vehicles] + 0.6 Rp [if eligible persons]) / (weights of eligible
              classes), R_c = within-tolerance hits / eligible GT (1.0 m vehicle, 1.2 m person; misses count). Y is the
              registered reward-trial Y with NO segmentation multiplier (segmentation moves into the constraints). Off-grid
              q: Y linear between the bracketing grid anchors, undefined if either is (registered rule). Y is undefined
              exactly when Q_perc is undefined, so outcome semantics are unchanged (timely undefined: excluded + session
              reset; late undefined: timeout, r^P = 0).
  constraints for c in {vehicle, person} (expected counts at off-grid q; ref = the same scene's grid cell (0, 0)):
              g_loc  = hits - tau_loc  * ref_hits
              g_seg  = inter - tau_seg * rho_c * union          rho_c = FIT aggregate reference IoU
              g_prec = tp - tau_prec * pi_c * (tp + fp)         pi_c  = FIT aggregate reference precision
              normalised by kappa_k (FIT mean per-scene reference hits / inter / tp), so mean(g_k)/kappa_k ~= retention_k - tau_k.
  training reward r = r^P + sum_k lambda_k * g_k / kappa_k      (recomputed at replay sampling time with the current lambda)
  dual        lambda_k <- clip(lambda_k - ETA * mean_window(g_k / kappa_k), 0, LAMBDA_MAX) every DUAL_EVERY stochastic
              decisions over the last WINDOW stochastic decisions (all decisions, including excluded ones: the registered
              retention is all-scored).
"""
from __future__ import annotations

import hashlib
import json
from typing import Optional

import numpy as np

from rl_agent.splitfusion_run5_reward_trials_v1 import catalog as K
from rl_agent.splitfusion_run5_reward_trials_v1 import rewards as R
from rl_agent.ue_production_transport_model_v2 import scene_source as SS

OBJECTIVE_ID = "CLOC"
CONSTRAINTS = ("loc_vehicle", "loc_person", "seg_vehicle", "seg_person", "precision_vehicle", "precision_person")
ETA, LAMBDA_MAX, DUAL_EVERY, WINDOW, WARMUP = 1.0, 5.0, 400, 2000, 288
TRAIN_MARGIN = 0.015
REGISTERED_TIERS = {"loc_vehicle": 0.95, "loc_person": 0.90, "seg_vehicle": 0.85, "seg_person": 0.80,
                    "precision_vehicle": 0.95, "precision_person": 0.90}


def y_from_components(comp: dict) -> Optional[float]:
    num = den = 0.0
    for cls in R.TOL:                                      # same classes, order and expression as rewards.value_from_row
        if comp[cls]["eligible"]:
            num += R.WEIGHT[cls] * comp[cls]["hits"] / comp[cls]["eligible"]
            den += R.WEIGHT[cls]
    return None if den == 0 else num / den


def definition(constants: dict) -> dict:
    return dict(schema="splitfusion.run5_constrained_v1.objective.v1", objective=OBJECTIVE_ID,
                primary="r^P = D * Y (registered Y, no segmentation multiplier)", tolerances_m=dict(R.TOL),
                class_weights=dict(R.WEIGHT), deadline_ms_inclusive=170, timeout=0.0,
                timely_undefined="EXCLUDED_EVALUATOR_FAULT_SESSION_RESET", late_undefined="TIMEOUT",
                off_grid_q="LINEAR_BETWEEN_GRID_ANCHORS_UNDEFINED_IF_EITHER", constraints=list(CONSTRAINTS),
                dual=dict(eta=ETA, lambda_max=LAMBDA_MAX, every_stochastic_decisions=DUAL_EVERY, window=WINDOW,
                          warmup_decisions_excluded=WARMUP), constants=constants)


def sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


class LocCatalogV1(K.TrialSceneCatalogV1):
    """The reward-trial catalogue (FIT or held_scene) serving r^P's Y instead of an arm value. Loaded through the
    registered loader (arm C0 only to pass its arm check); the arm label is then replaced so nothing is called C0."""

    def __init__(self, split: str, objective_sha256: str) -> None:
        super().__init__(split, "C0")
        self.arm = OBJECTIVE_ID
        self.reward_sha256 = objective_sha256
        self.reward_binding_sha256 = sha({"catalog": self.binding_sha256, "reward": objective_sha256})
        self._y = {}
        for key, entry in self._scenes.items():
            ys = {}
            for cell, comp in entry["components"].items():
                y = y_from_components(comp)
                if (y is None) != (entry["grid"][cell][0] is None):
                    raise ValueError(f"{key} {cell}: Y and Q_perc definedness differ")
                ys[cell] = y
            self._y[key] = ys

    def reward_value(self, scene_key: str, *, mode_id: int, q_e4: int) -> Optional[float]:
        ys = self._y[scene_key]
        if q_e4 in SS.Q_E4_GRID:
            return ys[(mode_id, q_e4)]
        parts = self._bracket(q_e4)
        (lo, _), (hi, w) = parts
        a, b = ys[(mode_id, lo)], ys[(mode_id, hi)]
        if a is None or b is None:
            return None
        return float(a) + w * (float(b) - float(a))

    def expected(self, scene_key: str, *, mode_id: int, q_e4: int) -> dict:
        out = {c: {f: 0.0 for f in ("eligible", "hits", "tp", "fp", "inter", "union")} for c in R.TOL}
        for w, comp in self.components(scene_key, mode_id=mode_id, q_e4=q_e4):
            for c in R.TOL:
                for f in out[c]:
                    out[c][f] += w * comp[c][f]
        return out


def fit_constants(cat: LocCatalogV1) -> dict:
    """FIT reference aggregates (uniform over catalogue scenes, as the scene draw is uniform)."""
    ref = [cat.reference_components(k) for k in cat.keys]
    out = {}
    for c in R.TOL:
        s = lambda f: float(sum(r[c][f] for r in ref))
        out[c] = dict(rho=s("inter") / s("union"), pi=s("tp") / (s("tp") + s("fp")),
                      kappa_loc=s("hits") / len(ref), kappa_seg=s("inter") / len(ref), kappa_prec=s("tp") / len(ref))
    return out


def g_vector(cat: LocCatalogV1, scene_key: str, mode_id: int, q_e4: int, targets: dict, consts: dict) -> np.ndarray:
    """Normalised constraint terms g_k / kappa_k for one decision (order = CONSTRAINTS)."""
    e = cat.expected(scene_key, mode_id=mode_id, q_e4=q_e4)
    r = cat.reference_components(scene_key)
    out = []
    for name in CONSTRAINTS:
        kind, cls = name.split("_")
        k, tau = consts[cls], targets[name]
        if kind == "loc":
            out.append((e[cls]["hits"] - tau * r[cls]["hits"]) / k["kappa_loc"])
        elif kind == "seg":
            out.append((e[cls]["inter"] - tau * k["rho"] * e[cls]["union"]) / k["kappa_seg"])
        else:
            out.append((e[cls]["tp"] - tau * k["pi"] * (e[cls]["tp"] + e[cls]["fp"])) / k["kappa_prec"])
    return np.asarray(out, dtype=np.float64)
