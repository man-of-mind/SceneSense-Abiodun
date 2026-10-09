"""Qualification 1+2 (offline): resolve and freeze the 12 static q=0 identities; prove q=0 keeps every cell and
bypasses ranking through the real continuous UE -> edge codec path.

For every mode 0..11 at q_e4 = 0:
  * ``DynamicExecutionContract.resolve_q_e4`` (the resolver the live engine calls) -> exact profile; the catalogue
    anchor must be action 6*mode with its catalogue profile_id; keep 21,504 / drop 0; ranker bypassed; no active
    ranker checkpoint; execution-bundle sha256 recorded (and equal to the two previously recorded M1 bundles for
    modes 0 and 9).
  * ``ContinuousUERuntimeV2.prepare`` with the pinned AE encoders and a ranker that RAISES if consulted, on one seeded
    synthetic C2 (the deployed front is not needed: C2 is the codec input); then ``ContinuousEdgeRuntimeV2.process``
    with the pinned AE decoders and an identity tail.  Checks: envelope identity, inner header keep_count 21,504,
    decoded keep mask all ones (AE / low-bit paths) or every cell present (noAE UINT8), finite dense C2 of the frozen
    shape.
  * Negatives: q=0 encode WITH a ranker is refused; a tampered execution-bundle digest, a q mismatch and a fabricated
    anchor are refused by the edge.
Nothing live is started; the GPU is used only for the codec/AE tensors if available.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[2]
RECORDED = {0: "583cbab6b68449495e779259507fdf7dcdfba56b5d31894ab9f8955da4deb10a",   # m1_static_live_v1/contract.py:53
            9: "dbe6a1e5ee1b0c0e2ab8617a35ad6388f2325aff3b1761c1cf958a8a1ee976cc"}   # m1_fixed_live_v2/contracts.py:166
FULL_GRID = 112 * 192
SEED = 20261010


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _jsonable(value):
    if dataclasses.is_dataclass(value):
        return {f.name: _jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, Path):
        return str(value)
    return value if isinstance(value, (str, int, float, bool, type(None))) else repr(value)


def _verified(contract, profile):
    contract.verify_profile(profile)          # raises on any disagreement with the contract's own resolution
    return True


class RefusingRanker:
    calls = 0

    def score_cells(self, c2):
        RefusingRanker.calls += 1
        raise AssertionError("ranker consulted at q=0")


def resolve_all():
    from rl_agent.splitfusion_live_dispatch_v1.dynamic_execution_contract import (
        MEASURED_ANCHOR, load_dynamic_execution_contract)
    contract = load_dynamic_execution_contract()
    catalogue = json.loads((ROOT / "rl_agent/splitfusion_action_catalog_v1/splitfusion_72_action_catalog.json")
                           .read_bytes())
    rows = {}
    for item in catalogue["profiles"]:
        rows[int(item["action_id"])] = item
    out = []
    for mode in range(12):
        p = contract.resolve_q_e4(mode, 0)
        anchor = rows.get(6 * mode, {})
        checks = dict(
            anchor_action_is_6_times_mode=p.action_id == 6 * mode,
            catalogue_profile_id=p.profile_id == anchor.get("profile_id"),
            catalogue_family_quantizer=(p.family, p.quantizer) == (anchor.get("family"), anchor.get("quantizer")),
            measured_anchor=p.measurement_status == MEASURED_ANCHOR,
            keep_all=p.keep_count == FULL_GRID and p.drop_count == 0,
            ranker_bypassed=p.execution_bundle.ranker_bypassed is True
            and p.execution_bundle.active_ranker_checkpoint is None,
            recorded_bundle_matches=(p.execution_bundle_sha256 == RECORDED[mode]) if mode in RECORDED else True,
            verify_profile=_verified(contract, p))
        out.append(dict(
            mode_id=p.mode_id, family=p.family, quantizer=p.quantizer, bit_width=p.bit_width, q_e4=p.q_e4,
            keep_count=p.keep_count, drop_count=p.drop_count, action_id=p.action_id, profile_id=p.profile_id,
            measurement_status=p.measurement_status, ranker_bypassed=p.execution_bundle.ranker_bypassed,
            active_ranker_checkpoint=_jsonable(p.execution_bundle.active_ranker_checkpoint),
            ae_checkpoint=_jsonable(p.ae_checkpoint), perception_checkpoint=_jsonable(p.perception_checkpoint),
            transported_channels=p.transported_channels, latent_width=p.latent_width, routing_tag=p.routing_tag,
            wire=_jsonable(p.wire), execution_bundle_sha256=p.execution_bundle_sha256,
            recorded_bundle_sha256=RECORDED.get(mode), checks=checks))
    return contract, out


def load_autoencoders(device):
    from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_ae_v1 import (
        ae_phase11b_gpu_qualification as phase11b)
    import torch
    from rl_agent.splitfusion_live_dispatch_v1.live_pilot_runtime import _AE, _Ledger
    ledger, models = _Ledger(), {}
    for family, _family_id, bottleneck in phase11b.FAMILIES:
        if bottleneck is None:
            continue
        item = phase11b.FROZEN_INPUTS[family]
        payload = torch.load(phase11b._repository_path(item["path"]), map_location="cpu", weights_only=False)
        models[family] = _AE(family, phase11b._load_selected_autoencoder(family, bottleneck, item, payload, device),
                             ledger)
        del payload
    return models, ledger


def exercise(contract, identities, device_name):
    import torch
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import continuous_execution_v2 as X
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2.phase6_prewarm_v2 import C2_SHAPE
    from rl_agent.splitfusion_live_dispatch_v1.transport import ae_uint8_transport, lowbit_transport, uint8_codec
    from rl_agent.splitfusion_live_dispatch_v1.transport import DispatchContractError, ProductionSplitCodec
    from rl_agent.splitfusion_live_dispatch_v1.timing import EDGE_STAGES, UE_STAGES, StageRecorder
    device = torch.device(device_name)
    encoders, ledger = load_autoencoders(device)
    codec = ProductionSplitCodec()
    generator = torch.Generator(device=device)
    generator.manual_seed(SEED)
    c2 = torch.rand(C2_SHAPE, generator=generator, device=device, dtype=torch.float32)
    captured = {}

    def tail(decoded, _result):
        captured["c2"] = decoded
        return None
    ue = X.ContinuousUERuntimeV2(contract, front=lambda _input: c2, ranker=RefusingRanker(), ae_encoders=encoders,
                                 codec=codec)
    edge = X.ContinuousEdgeRuntimeV2(contract, tail=tail, ae_decoders=encoders, codec=codec, tail_device=device)
    results = []
    for row in identities:
        profile = contract.resolve_q_e4(row["mode_id"], 0)
        frame = X.FrameIdentityV2(reward_requested=True, session_uuid="00000000-0000-4000-8000-000000000000",
                                  controller_lineage_sha256="0" * 64, decision_seq=0, ticket_seq=0,
                                  frame_id=row["mode_id"], tensor_seq=0, capture_timestamp_ns=1)
        before = RefusingRanker.calls
        prepared = ue.prepare(profile, None, frame)
        result = edge.process(prepared.wire_bytes)
        inspected = codec.inspect(prepared.envelope.inner_payload, timing=StageRecorder(EDGE_STAGES))
        if inspected.kind == "noae_uint8":
            decoded, _q = uint8_codec.decode(inspected.sparse_bytes)
            kept = int((decoded.reshape(decoded.shape[0], -1).abs().amax(dim=0) > 0).sum())
            mask_kind = "every cell present in the decoded dense grid"
        elif inspected.kind == "ae_uint8":
            _decoded, mask, _q, _ = ae_uint8_transport.decode_sparse(inspected.sparse_bytes)
            kept, mask_kind = int(mask.reshape(-1).bool().sum()), "keep mask"
        else:
            _decoded, mask, _q = lowbit_transport.decode_inspected(inspected.parsed)
            kept, mask_kind = int(mask.reshape(-1).bool().sum()), "keep mask"
        out_c2 = captured.pop("c2")
        checks = dict(
            ranker_not_consulted=RefusingRanker.calls == before,
            envelope_identity=(prepared.envelope.mode_id, prepared.envelope.q_e4, prepared.envelope.keep_count,
                               prepared.envelope.anchor_action_id, prepared.envelope.execution_bundle_sha256)
            == (row["mode_id"], 0, FULL_GRID, row["action_id"], row["execution_bundle_sha256"]),
            inner_header_keep_all=inspected.identity.keep_count == FULL_GRID and inspected.identity.q_e4 == 0,
            every_cell_kept=kept == FULL_GRID,
            edge_reresolved_same_bundle=result.profile.execution_bundle_sha256 == row["execution_bundle_sha256"],
            dense_finite_c2=tuple(out_c2.shape) == tuple(C2_SHAPE) and bool(torch.isfinite(out_c2).all()))
        negatives = {}
        try:
            codec.encode(X.codec_view(profile), c2, ranker=RefusingRanker(),
                         ae_encoder=None if profile.family == "noAE" else encoders[profile.family],
                         timing=StageRecorder(UE_STAGES))
            negatives["q0_with_ranker_refused"] = False
        except DispatchContractError as exc:
            negatives["q0_with_ranker_refused"] = "bypass" in str(exc)
        for label, field, value in (("tampered_bundle_refused", "execution_bundle_sha256", "f" * 64),
                                    ("q_mismatch_refused", "q_e4", 100),
                                    ("fabricated_anchor_refused", "anchor_action_id", (row["action_id"] + 1) % 72)):
            forged = dataclasses.replace(prepared.envelope, **{field: value})
            try:
                edge.process(X.pack_envelope_v3(forged))
                negatives[label] = False
            except Exception:                         # any typed refusal counts; acceptance is the failure
                negatives[label] = True
        results.append(dict(mode_id=row["mode_id"], inner_kind=inspected.kind, kept_cells=kept,
                            kept_cells_evidence=mask_kind, inner_payload_bytes=len(prepared.envelope.inner_payload),
                            wire_bytes=len(prepared.wire_bytes), checks=checks, negatives=negatives))
    return results, dict(ledger.__dict__) if hasattr(ledger, "__dict__") else {}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--freeze", action="store_true", help="write the package's FIXED_IDENTITIES.json (create-only)")
    args = parser.parse_args(argv)
    import torch
    args.out.mkdir(parents=True, exist_ok=False)
    contract, identities = resolve_all()
    device = args.device if (not args.device.startswith("cuda") or torch.cuda.is_available()) else "cpu"
    exercised, _ = exercise(contract, identities, device)
    passed = (all(all(r["checks"].values()) for r in identities)
              and all(all(r["checks"].values()) and all(r["negatives"].values()) for r in exercised))
    document = dict(
        schema="splitfusion.static_modes_v13.identities.v1", status="PASS" if passed else "FAIL",
        root=str(ROOT), q_e4=0, full_grid_cells=FULL_GRID, identities=identities, exercised=exercised,
        contract=dict(runtime_binding_sha256=contract.runtime_binding_sha256,
                      runtime_source_base_commit=contract.runtime_source_base_commit,
                      behavioral_source_binding_sha256=contract.behavioral_source_binding_sha256),
        sources={rel: sha(ROOT / rel) for rel in (
            "rl_agent/splitfusion_live_dispatch_v1/dynamic_execution_contract.py",
            "rl_agent/splitfusion_live_dispatch_v1/transport.py",
            "rl_agent/splitfusion_live_dispatch_v1/runtime_binding.json",
            "rl_agent/splitfusion_hybrid_sac_live_route_b_v2/continuous_execution_v2.py",
            "rl_agent/splitfusion_action_catalog_v1/splitfusion_72_action_catalog.json",
            "rl_agent/splitfusion_static_modes_v13/identities.py")},
        environment=dict(python=platform.python_version(), torch=torch.__version__, device=device),
        live_services_started=False)
    (args.out / "IDENTITIES.json").write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
    (args.out / "OUTPUT_SHA256.json").write_text(json.dumps({"IDENTITIES.json": sha(args.out / "IDENTITIES.json")},
                                                            indent=1) + "\n")
    if passed and args.freeze:
        table = ROOT / "rl_agent/splitfusion_static_modes_v13/FIXED_IDENTITIES.json"
        if table.exists():
            raise SystemExit("refusing to overwrite the frozen identity table: " + str(table))
        table.write_text(json.dumps(dict(
            schema="splitfusion.static_modes_v13.fixed_identities.v1",
            source=str(args.out / "IDENTITIES.json"), source_sha256=sha(args.out / "IDENTITIES.json"),
            q_e4=0, modes={str(r["mode_id"]): r["execution_bundle_sha256"] for r in identities},
            profiles={str(r["mode_id"]): [r["family"], r["quantizer"], r["action_id"], r["profile_id"]]
                      for r in identities}), indent=1, sort_keys=True) + "\n")
    print(json.dumps(dict(status=document["status"], rows=[(r["mode_id"], r["action_id"], r["profile_id"],
                                                            r["execution_bundle_sha256"][:12]) for r in identities],
                          exercised=[(r["mode_id"], r["kept_cells"], r["inner_payload_bytes"],
                                      all(r["checks"].values()), all(r["negatives"].values())) for r in exercised])))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
