# Preserved failed run — 20260924_124747

**Status: `UE_MCS_BACKLOG_CALIBRATION_FAILED`, 0/12 cells. Retained, not deleted.**

Failed at the calibration attach step:
`RunFailure: UE did not attach and reach the edge host before timeout`.

## Cause

Harness defect, not a radio or design problem. The attach gate required an ICMP round trip to the
edge host (the host's own address on the CN bridge, `192.168.70.129`). That round trip cannot
succeed here: the host routes `10.0.0.0/16` via the corporate LAN (`10.0.0.2 via 10.21.16.1 dev
wlp130s0f0`), so an ICMP echo *reply* to the UE is misrouted. Only `oai-ext-dn` carries the correct
route (`10.0.0.0/16 via 192.168.70.134`).

The **data** path was never shown to be broken. The edge reassembly endpoint is receive-only and
never replies, and `rp_filter` is `2` (loose) on this host, so inbound datagrams from the UE are
accepted on the bridge interface. The gate was testing the wrong thing.

## Repair

1. The attach gate now pings `oai-ext-dn` (`192.168.70.135`), the one CN-side address with a working
   return route, which is what actually proves the PDU session is up.
2. A real **UDP path probe** was added: before any cell runs, datagrams are sent from the UE tunnel
   address to the edge host over the radio and the UPF, and at least one must arrive. The
   receive-only path is now verified rather than assumed, and a cell refuses to start without it.

No OAI source change, no host routing change, and no change to the experimental design. The campaign
was relaunched into a fresh directory; this one is kept as the failure record.

Channel state was restored and the host verified cold at exit (`final_cold_state.json`, `cold: true`).
