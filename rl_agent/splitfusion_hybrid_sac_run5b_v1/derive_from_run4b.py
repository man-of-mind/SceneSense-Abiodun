#!/usr/bin/env python3
"""Derive ``run5b_learner.py`` and ``run5b_runner.py`` from Run-4B, mechanically.

Run 5B must be Run 4B (same SAC configuration, replay, warm-up, seed plan,
checkpoint cadence, bundle format, smoke gate, resume test and campaign
driver) on the shared joint SNR/MCS channel with the causal SNR as feature 21.
To make that auditable, the two modules are not hand-written: they are the
Run-4B sources with exactly the substitutions listed below.  Every
substitution must match the stated number of times or derivation fails, and a
test regenerates both files and requires byte equality.

    python -m rl_agent.splitfusion_hybrid_sac_run5b_v1.derive_from_run4b [--check]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
RUN4B = PACKAGE.parent / "splitfusion_hybrid_sac_run4b_v1"
RUN4B_SOURCE_SHA256 = {
    "learner.py": "a7969146f45d3360811590763f76331d71b91b7559dfc003b20cea352de38ce5",
    "runner.py": "d8f7f63ea66b8b09e1a168cdd6826f631998c263f964f4c1b7d376425fe04538",
}

LEARNER_SUBSTITUTIONS = (
    ('"""Run-4B replay buffer and Hybrid-SAC trainer.\n',
     '"""Run-5B replay buffer and Hybrid-SAC trainer (21-feature state).\n\n'
     'DERIVED MECHANICALLY from ``splitfusion_hybrid_sac_run4b_v1/learner.py`` by\n'
     '``derive_from_run4b.py``; the only differences are the listed substitutions.\n', 1),
    ("from . import contract as C\nfrom .models import validate_models\n",
     "from . import run5b_state_contract as C\nfrom .run5b_models import validate_models\n", 1),
    ('"replay rows must carry 20 features"', '"replay rows must carry 21 features"', 1),
)

RUNNER_SUBSTITUTIONS = (
    ('"""Run-4B offline Hybrid-SAC runner, durable checkpoints and campaign CLI.\n',
     '"""Run-5B offline Hybrid-SAC runner, durable checkpoints and campaign CLI.\n\n'
     'DERIVED MECHANICALLY from ``splitfusion_hybrid_sac_run4b_v1/runner.py`` by\n'
     '``derive_from_run4b.py``; every difference is one listed substitution.\n'
     'Run 5B is Run 4B on the shared joint SNR/MCS channel\n'
     '(``splitfusion_joint_channel_v1``) with the causal UL-SNR proxy as feature 21.\n'
     'The campaign records, but does not enforce, the Run-4B family-dominance window.\n', 1),
    ("import argparse\nimport hashlib\n", "import argparse\nimport dataclasses\nimport hashlib\n", 1),
    ("from . import contract as C\nfrom . import environment as E\n"
     "from . import learner as L\nfrom . import models as M\n",
     "from rl_agent.splitfusion_joint_channel_v1 import joint_channel as E\n\n"
     "from . import run5b_checks as K\nfrom . import run5b_learner as L\n"
     "from . import run5b_models as M\nfrom . import run5b_registration as REG\n"
     "from . import run5b_state_contract as C\n", 1),
    ('ROOT = Path(__file__).resolve().parents[2]\n',
     'ROOT = Path(__file__).resolve().parents[2]\n'
     'EVIDENCE_ROOT = Path(os.environ.get("RUN5B_EVIDENCE_ROOT",\n'
     '                                    ROOT.parent / "abiodun")).resolve()\n', 1),
    ('RUNNER_SCHEMA = "splitfusion.run4b.runner.v1"', 'RUNNER_SCHEMA = "splitfusion.run5b.runner.v1"', 1),
    ('BUNDLE_SCHEMA = "splitfusion.run4b.checkpoint_bundle.v1"',
     'BUNDLE_SCHEMA = "splitfusion.run5b.checkpoint_bundle.v1"', 1),
    ('EXPORT_SCHEMA = "splitfusion.run4b.actor_export.v1"',
     'EXPORT_SCHEMA = "splitfusion.run5b.actor_export.v1"', 1),
    ("        states.append(C.build_features(observation, prior, scaling))\n",
     "        states.append(C.build_features(observation, prior, scaling,\n"
     "                                       (index % 20) / 19.0))\n", 1),
    ("Run4BRunnerV1", "Run5BRunnerV1", 9),
    ("E.SharedSourcesV1.load()", "E.load_sources(EVIDENCE_ROOT)", 3),
    ("E.SharedSourcesV1", "E.JointSourcesV1", 2),
    ("        self.env = E.Run4BEnvironmentV1(sources, seed=seed)\n",
     "        self.env = E.JointChannelEnvironmentV1(sources, seed=seed)\n"
     "        self.snr = C.ModeledSnrObserverV1(seed)\n"
     "        self.registration_sha256 = REG.sealed_sha256()\n"
     "        _require(sources.binding_sha256 == REG.sealed_joint_channel_binding_sha256(),\n"
     "                 \"joint-channel binding differs from the Run-5B registration\")\n", 1),
    ('            "feature_schema_sha256": C.FEATURE_SCHEMA_SHA256,\n'
     '            "feature_order": list(C.FEATURE_ORDER),\n'
     '            "reward_schema_sha256": C.REWARD_SCHEMA_SHA256,\n',
     '            "feature_schema_sha256": C.FEATURE_SCHEMA_SHA256,\n'
     '            "feature_order": list(C.FEATURE_ORDER),\n'
     '            "feature_order_sha256": C.FEATURE_ORDER_SHA256,\n'
     '            "joint_channel_binding_sha256": self.sources.binding_sha256,\n'
     '            "registration_sha256": self.registration_sha256,\n'
     '            "reward_schema_sha256": C.REWARD_SCHEMA_SHA256,\n', 1),
    ("    def collect_one(self) -> tuple[dict[str, Any], E.TransitionV1]:\n"
     "        features = self.env.current_features()\n",
     "    def collect_one(self) -> tuple[dict[str, Any], E.E4.TransitionV1]:\n"
     "        features = C.features_for(self.env, self.snr)\n", 1),
    ('        _require(transition.state == features, "state changed during step")\n',
     '        _require(transition.state == features[:C.SNR_FEATURE_INDEX],\n'
     '                 "state changed during step")\n'
     '        next_features = C.features_for(self.env, self.snr)\n'
     '        _require(transition.next_state == next_features[:C.SNR_FEATURE_INDEX],\n'
     '                 "successor Run-4B prefix differs")\n'
     '        transition = dataclasses.replace(transition, state=features,\n'
     '                                         next_state=next_features)\n', 1),
    ('                   "prior_ul_mcs", "pre_action_backlog_bytes",\n'
     '                   "reward_wire_bytes")},\n',
     '                   "prior_ul_mcs", "pre_action_backlog_bytes",\n'
     '                   "reward_wire_bytes", "snr_db", "successor_mcs",\n'
     '                   "successor_snr_db", "generated_ticks_after_observed")},\n'
     '               "snr_feature": transition.state[C.SNR_FEATURE_INDEX],\n', 1),
    ("        prior_ok = transitions[0].state[4:] == (0.0,) * 16\n",
     "        prior_ok = transitions[0].state[C.PRIOR_SLICE] == (0.0,) * 16\n", 1),
    ("            expected = C.build_features(\n", "            expected = C.C4.build_features(\n", 1),
    ("            prior_ok = prior_ok and current.state[4:] == expected\n",
     "            prior_ok = prior_ok and current.state[C.PRIOR_SLICE] == expected\n", 1),
    ('        rewards = [r["reward"] for r in rows]\n        report = {\n'
     '            "schema": "splitfusion.run4b.preflight_288.v1",\n',
     '        checks.update(K.preflight_snr_checks(self, transitions))\n'
     '        rewards = [r["reward"] for r in rows]\n        report = {\n'
     '            "schema": "splitfusion.run5b.preflight_288.v1",\n', 1),
    ('            "model_binding_sha256": M.MODEL_BINDING_SHA256,\n'
     '            "operational_latency_provider_sha256":\n'
     '                runner.sources.provider.binding_sha256,\n'
     '            "runner_binding": runner.binding,\n',
     '            "model_binding_sha256": M.MODEL_BINDING_SHA256,\n'
     '            "operational_latency_provider_sha256":\n'
     '                runner.sources.provider.binding_sha256,\n'
     '            "feature_schema_id": C.FEATURE_SCHEMA["schema_id"],\n'
     '            "feature_order_sha256": C.FEATURE_ORDER_SHA256,\n'
     '            "registration_sha256": runner.registration_sha256,\n'
     '            "joint_channel_binding_sha256": runner.sources.binding_sha256,\n'
     '            "actor_tree_sha256": R4O._tree_sha256(runner.actor.state_dict()),\n'
     '            "runner_binding": runner.binding,\n', 1),
    ("    manifest = json.loads(manifest_bytes)\n    _require(manifest.get(\"schema\") == BUNDLE_SCHEMA,\n",
     "    manifest = json.loads(manifest_bytes)\n"
     "    M.require_run5b_identity(manifest, registration_sha256=runner.registration_sha256)\n"
     "    _require(manifest.get(\"schema\") == BUNDLE_SCHEMA,\n", 1),
    ('        "runner_binding_sha256": runner.binding_sha256,\n'
     '        "operational_latency_provider_sha256":\n',
     '        "runner_binding_sha256": runner.binding_sha256,\n'
     '        "feature_schema_id": C.FEATURE_SCHEMA["schema_id"],\n'
     '        "feature_order_sha256": C.FEATURE_ORDER_SHA256,\n'
     '        "registration_sha256": runner.registration_sha256,\n'
     '        "joint_channel_binding_sha256": runner.sources.binding_sha256,\n'
     '        "operational_latency_provider_sha256":\n', 1),
    ('    manifest = json.loads(manifest_path.read_text())\n'
     '    _require(manifest.get("schema") == EXPORT_SCHEMA, "foreign export schema",\n',
     '    manifest = json.loads(manifest_path.read_text())\n'
     '    M.require_run5b_identity(manifest, registration_sha256=REG.sealed_sha256())\n'
     '    _require(manifest.get("schema") == EXPORT_SCHEMA, "foreign export schema",\n', 1),
    ('"no Run-4B ACTOR_EXPORT.json"', '"no Run-5B ACTOR_EXPORT.json"', 1),
    ('"feature order differs from Run-4B"', '"feature order differs from Run-5B"', 1),
    ('f"not a Run-4B bundle directory: {path}"', 'f"not a Run-5B bundle directory: {path}"', 1),
    ('        [sys.executable, "-m", "rl_agent.splitfusion_hybrid_sac_run4b_v1.runner",\n',
     '        [sys.executable, "-m", "rl_agent.splitfusion_hybrid_sac_run5b_v1.run5b_runner",\n', 1),
    ("def cmd_smoke(args) -> int:\n    out = Path(args.out)\n",
     "def cmd_smoke(args) -> int:\n    out = Path(args.out)\n"
     "    K.require_disk(out, seeds=(17,), target=SMOKE_UPDATE)\n", 1),
    ('    report = {"schema": "splitfusion.run4b.smoke_500_gate.v1",\n',
     '    final_runner = restore_runner(seed_dir / "checkpoints" / bundle_name(SMOKE_UPDATE),\n'
     '                                  sources, 17)\n'
     '    gates.update(K.smoke_snr_gates(seed_dir, final_runner))\n'
     '    report = {"schema": "splitfusion.run5b.smoke_500_gate.v1",\n', 1),
    ('              "actor_family_occupancy_updates_1_500": window}\n',
     '              "actor_family_occupancy_updates_1_500": window,\n'
     '              "snr_diagnostics_at_500": K.snr_diagnostics(final_runner, 17)}\n', 1),
    ('    report = {"schema": "splitfusion.run4b.resume_equivalence.v1",\n',
     '    report = {"schema": "splitfusion.run5b.resume_equivalence.v1",\n', 1),
    ('    start = {"schema": "splitfusion.run4b.campaign.v1", "status": "RUNNING",\n',
     '    K.require_disk(out, seeds=SEEDS, target=FINAL_UPDATE)\n'
     '    start = {"schema": "splitfusion.run5b.campaign.v1", "status": "RUNNING",\n', 1),
    ('                               "--target", str(FINAL_UPDATE),\n'
     '                               "--enforce-dominance"],\n',
     '                               "--target", str(FINAL_UPDATE)],\n', 1),
    ('    final = {**start, "status": "COMPLETE", "exit_codes": exit_codes,\n',
     '    fresh = K.fresh_process_verify(out)\n'
     '    final = {**start, "status": "COMPLETE", "exit_codes": exit_codes,\n'
     '             "fresh_process_actor_verification": fresh,\n', 1),
    # Last: the shared sources now carry the Run-4B sources under ``run4b``.
    ("sources.provider.", "sources.run4b.provider.", 7),
    ("self.sources.action_catalog", "self.sources.run4b.action_catalog", 1),
)


class DerivationError(RuntimeError):
    pass


def derive(source: str, substitutions) -> str:
    out = source
    for old, new, count in substitutions:
        found = out.count(old)
        if found != count:
            raise DerivationError(f"expected {count} occurrence(s), found {found}: {old[:70]!r}")
        out = out.replace(old, new)
    return out


def derived_files() -> dict[str, str]:
    files = {}
    for name, subs, target in (("learner.py", LEARNER_SUBSTITUTIONS, "run5b_learner.py"),
                               ("runner.py", RUNNER_SUBSTITUTIONS, "run5b_runner.py")):
        data = (RUN4B / name).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != RUN4B_SOURCE_SHA256[name]:
            raise DerivationError(f"Run-4B {name} differs from the pinned source ({digest})")
        files[target] = derive(data.decode("utf-8"), subs)
    return files


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    for target, text in derived_files().items():
        path = PACKAGE / target
        if args.check:
            if path.read_text(encoding="utf-8") != text:
                print(f"DRIFT {target}")
                return 1
        else:
            path.write_text(text, encoding="utf-8")
        print(target, hashlib.sha256(text.encode()).hexdigest())
    return 0


if __name__ == "__main__":
    sys.exit(main())
