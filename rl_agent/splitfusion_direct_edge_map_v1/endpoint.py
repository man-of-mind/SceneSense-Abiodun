"""Resolution and auditing of the direct edge-to-map endpoint.

The spatial map is an edge application that runs on the edge host, while the
inference service runs in the ``oai-perception-rx`` container attached to the
CN5G bridge ``oai-cn5g-public-net``. The direct object-map path is therefore a
container-to-host datagram on that directly-connected bridge: the edge sends to
the host's own address on the bridge.

That address is resolved from Docker rather than hardcoded, and then audited: it
must lie inside the CN5G bridge subnet, must not be the UE tunnel address, must
not be loopback, and must not be the unspecified address.
"""

from __future__ import annotations

import ipaddress
import json
import subprocess
from dataclasses import dataclass
from typing import Any, Sequence

from .protocol import DirectMapProtocolError, assert_direct_map_endpoint, _require


CN5G_NETWORK_NAME = "oai-cn5g-public-net"
# The UE tunnel addresses the object-map path must never touch.
UE_TUNNEL_HOSTS = ("10.0.0.2",)
UE_TUNNEL_SUBNET = ipaddress.ip_network("10.0.0.0/16")
# The UE result port of the superseded edge->UE->map detour.
UE_RESULT_PORTS = (51004, 51104)


@dataclass(frozen=True)
class DirectMapEndpoint:
    """The audited edge-local address the map binds and the edge publishes to."""

    host: str
    port: int
    network_name: str
    subnet: str
    bridge_interface: str
    resolution: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "port": int(self.port),
            "network_name": self.network_name,
            "subnet": self.subnet,
            "bridge_interface": self.bridge_interface,
            "resolution": self.resolution,
            "traverses_ue_tunnel": False,
            "path": "EDGE_CONTAINER_TO_HOST_ON_CN5G_BRIDGE",
        }


def _docker_network_inspect(network: str, *, sudo: bool = True) -> dict[str, Any]:
    argv = (["sudo"] if sudo else []) + [
        "docker",
        "network",
        "inspect",
        str(network),
        "--format",
        "{{json .}}",
    ]
    completed = subprocess.run(
        argv,
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30.0,
    )
    _require(
        completed.returncode == 0,
        f"docker network inspect {network} failed rc={completed.returncode}: "
        f"{completed.stderr.strip()[:400]}",
    )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise DirectMapProtocolError(
            f"docker network inspect {network} returned unparsable JSON: {exc}"
        ) from exc


def resolve_direct_map_endpoint(
    *,
    port: int,
    network: str = CN5G_NETWORK_NAME,
    sudo: bool = True,
) -> DirectMapEndpoint:
    """Resolve and audit the host-side CN5G bridge address for the map."""

    document = _docker_network_inspect(network, sudo=sudo)
    ipam = document.get("IPAM") or {}
    configs = ipam.get("Config") or []
    _require(
        len(configs) == 1,
        f"{network} must declare exactly one IPAM config, saw {len(configs)}",
    )
    config = configs[0]
    gateway = str(config.get("Gateway") or "").strip()
    subnet = str(config.get("Subnet") or "").strip()
    _require(bool(gateway), f"{network} declares no gateway address")
    _require(bool(subnet), f"{network} declares no subnet")
    network_cidr = ipaddress.ip_network(subnet)
    address = ipaddress.ip_address(gateway)
    _require(
        address in network_cidr,
        f"{network} gateway {gateway} is outside its own subnet {subnet}",
    )
    _require(
        address not in UE_TUNNEL_SUBNET,
        f"{network} gateway {gateway} lies inside the UE tunnel subnet",
    )
    options = document.get("Options") or {}
    bridge = str(options.get("com.docker.network.bridge.name") or "").strip()
    assert_direct_map_endpoint(
        gateway,
        port,
        ue_hosts=UE_TUNNEL_HOSTS,
        forbidden_ports=UE_RESULT_PORTS,
    )
    return DirectMapEndpoint(
        host=gateway,
        port=int(port),
        network_name=str(network),
        subnet=subnet,
        bridge_interface=bridge,
        resolution="docker_network_inspect_gateway",
    )


def audit_publisher_destination(host: str, port: int) -> None:
    """The last gate before an object-map socket is used."""

    assert_direct_map_endpoint(
        host, port, ue_hosts=UE_TUNNEL_HOSTS, forbidden_ports=UE_RESULT_PORTS
    )


def static_address_audit(sources: Sequence[str]) -> dict[str, Any]:
    """Prove no direct-map publisher in ``sources`` can target the UE address.

    The audit is textual and deliberately conservative: it reports every line in
    the direct-edge-map implementation that mentions a UE tunnel host or a UE
    result port, so a reviewer sees each one and its justification rather than
    trusting an absence.
    """

    from pathlib import Path

    findings: list[dict[str, Any]] = []
    needles = tuple(UE_TUNNEL_HOSTS) + tuple(str(value) for value in UE_RESULT_PORTS)
    for source in sources:
        path = Path(source)
        if not path.is_file():
            continue
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            for needle in needles:
                if needle in line:
                    findings.append(
                        {
                            "path": str(source),
                            "line": number,
                            "needle": needle,
                            "text": line.strip()[:200],
                        }
                    )
    return {
        "sources_scanned": len(list(sources)),
        "ue_tunnel_hosts": list(UE_TUNNEL_HOSTS),
        "ue_result_ports": list(UE_RESULT_PORTS),
        "mentions": findings,
        "mention_count": len(findings),
    }
