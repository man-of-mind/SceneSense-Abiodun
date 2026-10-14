"""Fail-closed two-logical-UE binding. No fixed lease/RNTI and no NAT identity."""
import ipaddress
import json
from pathlib import Path
import subprocess

from .identity import Source, Sources, require

EXPECTED = (
    dict(ego_id='m4-ego-0', ue_id=0, imsi='001010000000001', tunnel='oaitun_ue1',
         stream_id='m4-ue0', feedback_port=52010, ingress_port=52000),
    dict(ego_id='m4-ego-1', ue_id=1, imsi='001010000000002', tunnel='oaitun_ue2',
         stream_id='m4-ue1', feedback_port=52011, ingress_port=52001))


def resolve(*, expected, run_id, cell_id, attach_sessions, uicc_text, ue_log, interfaces, rlc_rows):
    from rl_agent.multiue_oai.runner import parse_interface_ipv4, parse_runtime_uicc_imsis, parse_uicc_profiles
    require(list(expected) == list(EXPECTED), 'subscriber/source roster differs')
    profiles, running = parse_uicc_profiles(uicc_text), parse_runtime_uicc_imsis(ue_log)
    require(set(profiles) == {0, 1} and len(running) == 2, 'two and only two logical subscribers required')
    pairs = {i: set() for i in (0, 1)}
    for row in rlc_rows:
        uid, rnti = int(row['ue_id']), int(row['rnti'])
        require(uid in pairs and 0 < rnti <= 65535, 'foreign UE/invalid RNTI in attach epoch')
        pairs[uid].add(rnti)
    require(all(len(v) == 1 for v in pairs.values()), 'missing/ambiguous RNTI; never majority-vote')
    sources = []
    for item in expected:
        i = item['ue_id']
        require(profiles[i]['imsi'] == running[i] == item['imsi'], 'configured/runtime IMSI mismatch')
        lease = parse_interface_ipv4(interfaces[item['tunnel']], item['tunnel'])
        require(lease is not None and ipaddress.ip_address(lease) in ipaddress.ip_network('10.0.0.0/24'),
                'missing/non-DNN lease')
        sources.append(Source(**{k: v for k, v in item.items() if k != 'ingress_port'}, ip=lease,
            rnti=next(iter(pairs[i])), run_id=run_id, cell_id=cell_id, session_id=attach_sessions[str(i)]))
    return Sources(sources)


def discover_interfaces(expected=EXPECTED, run=subprocess.run):
    result = {}
    for item in expected:
        value = run(['ip', '-j', '-4', 'addr', 'show', 'dev', item['tunnel']],
                    capture_output=True, text=True, check=True, timeout=10)
        result[item['tunnel']] = value.stdout
    return result


def check_routes(sources, *, edge_container, edge_ip, run=subprocess.run):
    """Executed only by future live preflight, after real attach."""
    rows = []
    for source in sources.sources:
        front = run(['ip', '-j', 'route', 'get', edge_ip, 'from', source.ip,
                     'oif', source.tunnel], capture_output=True, text=True, check=True, timeout=10)
        route = json.loads(front.stdout)
        require(len(route) == 1 and route[0]['dev'] == source.tunnel, 'front route bypasses UE tunnel')
        back = run(['sudo', '-n', 'docker', 'exec', edge_container, 'ip', '-j', 'route', 'get', source.ip],
                   capture_output=True, text=True, check=True, timeout=10)
        down = json.loads(back.stdout)
        require(len(down) == 1 and down[0].get('gateway') == '192.168.70.134',
                'edge feedback route bypasses UPF')
        rows.append(dict(source=source.tag, front=route, feedback=down))
    return rows


def achieved_channel(sources, rows):
    """Per-transmission PUSCH, NOT the MAC TPC statistic or commanded SNR."""
    import statistics
    result = {}
    for source in sources.sources:
        selected = [r for r in rows if int(r['rnti']) == source.rnti]
        snr = [float(r['snrx10']) / 10. for r in selected]
        result[source.stream_id] = dict(samples=len(snr), commanded_snr_db=23.5,
            measured_pusch_median_db=statistics.median(snr) if snr else None,
            measured_mcs_median=statistics.median(float(r['mcs']) for r in selected) if selected else None,
            calibration_claim=False, missing_measurement=not bool(snr))
    return result
