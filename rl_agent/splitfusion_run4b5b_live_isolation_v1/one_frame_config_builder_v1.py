"""Create one sealed, create-only B one-frame engineering configuration.

Every host path is supplied explicitly.  Network identities are filled from
the already-registered W10275/L10319 authority and then validated by the
target schema.  The builder neither launches nor probes a service.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from . import final_actor_gate_v2 as F
from . import one_frame_engineering_v1 as O


def _write_create_only(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True, indent=2,
                          ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def build(*, run_id: str, cell_id: str, variant: str,
          actor_manifest_path: Path, actor_weights_path: Path,
          actor_evidence_root: Path, local_repository: Path,
          remote_repository: Path, local_attempt_root: Path,
          remote_attempt_root: Path, edge_campaign_config: Path,
          route_config: Path) -> O.OneFrameConfigV1:
    if variant not in {F.RUN4B_VARIANT, F.RUN5B_VARIANT}:
        raise O.OneFrameEngineeringError("unknown final B actor variant")
    actor_manifest_path = Path(actor_manifest_path)
    actor_weights_path = Path(actor_weights_path)
    if not actor_manifest_path.is_file() or actor_manifest_path.is_symlink():
        raise O.OneFrameEngineeringError("actor manifest is not a regular file")
    if not actor_weights_path.is_file() or actor_weights_path.is_symlink():
        raise O.OneFrameEngineeringError("actor weights are not a regular file")
    return O.OneFrameConfigV1(
        run_id=run_id, cell_id=cell_id, variant=variant,
        actor_manifest_path=actor_manifest_path,
        actor_manifest_sha256=O._digest(actor_manifest_path),
        actor_weights_path=actor_weights_path,
        actor_weights_sha256=O._digest(actor_weights_path),
        actor_evidence_root=Path(actor_evidence_root),
        local_repository=Path(local_repository),
        remote_repository=Path(remote_repository),
        local_attempt_root=Path(local_attempt_root),
        remote_attempt_root=Path(remote_attempt_root),
        edge_campaign_config=Path(edge_campaign_config),
        route_config=Path(route_config),
        network=O.NetworkBindingV1(
            local_host=O.LOCAL_HOST, remote_host=O.REMOTE_HOST,
            remote_ssh=O.REMOTE_SSH, local_lan_ip=O.LOCAL_LAN_IP,
            remote_lan_ip=O.REMOTE_LAN_IP, cn_subnet=O.CN_SUBNET,
            edge_ip=O.EDGE_IP, ext_dn_ip=O.EXT_DN_IP,
            ue_tunnel_ip=O.UE_TUNNEL_IP,
            ue_tunnel_interface=O.UE_TUNNEL_INTERFACE,
            ue_policy_table=O.UE_POLICY_TABLE, edge_route=O.EDGE_ROUTE,
            edge_feature_port=O.EDGE_FEATURE_PORT, ack_port=O.ACK_PORT,
            direct_map_host=O.LOCAL_LAN_IP,
            direct_map_port=O.DIRECT_MAP_PORT,
            carla_rpc_host="127.0.0.1", carla_rpc_port=O.CARLA_RPC_PORT),
        transmitted_budget=O.TRANSMITTED_BUDGET,
        policy_decision_budget=O.POLICY_DECISION_BUDGET,
        deadline_ns=O.DEADLINE_NS, clock_domain=O.CLOCK_DOMAIN,
        ack_semantics=O.ACK_SEMANTICS,
        postrun_semantics=O.POSTRUN_SEMANTICS,
        purpose=O.PURPOSE, policy_performance_claim=False,
        factory_module=O.FACTORY_MODULE)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--variant", required=True,
                        choices=(F.RUN4B_VARIANT, F.RUN5B_VARIANT))
    for option in (
        "actor-manifest-path", "actor-weights-path", "actor-evidence-root",
        "local-repository", "remote-repository", "local-attempt-root",
        "remote-attempt-root", "edge-campaign-config", "route-config",
    ):
        parser.add_argument("--" + option, type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = build(
        run_id=args.run_id, cell_id=args.cell_id, variant=args.variant,
        actor_manifest_path=args.actor_manifest_path,
        actor_weights_path=args.actor_weights_path,
        actor_evidence_root=args.actor_evidence_root,
        local_repository=args.local_repository,
        remote_repository=args.remote_repository,
        local_attempt_root=args.local_attempt_root,
        remote_attempt_root=args.remote_attempt_root,
        edge_campaign_config=args.edge_campaign_config,
        route_config=args.route_config)
    O.require_create_only_targets(config)
    _write_create_only(args.output, O.seal(config))
    print(json.dumps({
        "status": "ONE_FRAME_CONFIG_CREATED",
        "path": str(args.output),
        "binding_sha256": config.binding_sha256(),
        "variant": config.variant, "services_launched": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

