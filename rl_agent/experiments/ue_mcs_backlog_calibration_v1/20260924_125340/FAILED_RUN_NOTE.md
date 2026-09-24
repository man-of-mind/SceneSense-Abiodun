# Preserved failed run — 20260924_125340 (traffic bypassed the radio)

**Status: INVALID. Retained as the failure record; its measurements must not be used.**

Aborted by operator signal after 9 of 12 cells. The cells that completed are structurally clean —
450/450 decisions each, MCS coverage 450/450, HARQ round 0 only, MCS table 0 only, zero negative
ages, clock-bridge residual P95 ≈ 1 µs — but they measure the **wrong path**.

## Defect

The edge reassembly endpoint was set to the CN bridge **gateway**, `192.168.70.129`, which is an
address **local to this host**:

```
local 192.168.70.129 dev oai-cn5g proto kernel scope host src 192.168.70.129
```

`ip rule` priority 0 is `from all lookup local`. A datagram addressed to a host-local address is
therefore matched and delivered locally **before** the UE policy rule
(`from 10.0.0.x lookup 9999`, priority ≈ 31895) is ever consulted. The traffic never entered
`oaitun_ue1`, never reached the gNB, and never crossed RFsim. Binding the socket source to the UE
tunnel address does not change this: the source does not select the route when the destination is
local.

The symptom that exposed it: the **high** tier offers 880,567 B × 10 fps ≈ **70 Mbps** into a
~6 Mbps uplink, yet every block reassembled **150/150 complete with zero socket drops and zero
backlog** in every cell. That is impossible over the radio — 15 s of high-tier offered load is
~132 MB, which needs ~176 s at 6 Mbps. Pre-enqueue backlog was identically 0 at every tier, which is
what finally made the bypass undeniable.

The earlier UDP probe (`5/5 received`) did not catch this, because it only asked whether a datagram
*arrived*, not whether it went over the air. It arrived by loopback.

## Repair

1. The edge endpoint is now the **ext-DN container** (`oai-ext-dn`, `192.168.70.135`), which is not
   host-local, so the UE policy rule selects the tunnel.
2. Receivers run inside the ext-DN **network namespace** via
   `sudo nsenter -t <ext-dn pid> -n python3 …`, using the host python3 and the host filesystem. The
   production receiver is still executed unmodified, and no container is created or altered.
3. The probe is replaced by `verify_radio_path()`, which refuses a cell unless **both** hold:
   the edge host is absent from the host's `local` routing table, **and**
   `ip route get <edge> from <ue_ip> iif lo` resolves through `oaitun_ue1`.
4. A signal now aborts the whole campaign rather than being absorbed into one cell's failure handler.

## Scope of the defect

This same bypass applies to the uplink iperf3 traffic in the earlier
`ue_snr_bridge_qualification_v1` run, whose UL client targeted a CN-side address from the host. Its
reported "4.000 Mbps, 0.00 % loss in every profile" therefore did **not** measure the radio. That
does not change that study's conclusion — it rested on `UE_PHY_MEAS` being degenerate and on gNB
PUSCH SNR, which did respond to the channel actuator — but the uplink *delivery* numbers in it
should be treated as loopback, not radio, and the report's "did not reach the delivery frontier"
limitation understates the cause.

## Teardown

The campaign was killed mid-cell, so its own `finally` block did not run and no manifest was written.
The host was cleaned manually and verified: no `nr-softmodem`, no `nr-uesoftmodem`, no tracer, no
receiver or sender, and no `oaitun_ue1`. RFsim channel state lives inside the gNB process and is
destroyed with it; every RAN start re-materialises `noise_power_dB = -50` from config.
