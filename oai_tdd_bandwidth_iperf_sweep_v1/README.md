# OAI TDD and bandwidth iperf sweep v1

This runner compares one UE over four RFsim configurations while changing only
bandwidth and the TDD slot pattern:

| ID | Bandwidth | PRBs | TDD |
|---|---:|---:|---:|
| `bw40_tdd7d2u` | 40 MHz | 106 | 7 DL / 2 UL |
| `bw40_tdd4d5u` | 40 MHz | 106 | 4 DL / 5 UL |
| `bw100_tdd7d2u` | 100 MHz | 273 | 7 DL / 2 UL |
| `bw100_tdd4d5u` | 100 MHz | 273 | 4 DL / 5 UL |

The last configuration is generated inside each experiment from the validated
273-PRB configuration. The generator asserts that only
`nrofDownlinkSlots=7->4` and `nrofUplinkSlots=2->5` change. It never edits the
OAI submodule.

Every measurement cell gets a fresh lifecycle: stop UE/gNB, tear down the core
containers (without deleting volumes), verify cleanup, recreate the core, start
gNB and UE, verify `oaitun_ue1=10.0.0.2`, run traffic, save atomically, and tear
the complete stack down again. The core database currently binds IMSI
`001010000000001` to `10.0.0.2`; the runner checks both the source SQL and the
live tunnel on every cell.

The current subscriber SQL advertises 5QI 6 for this DNN. The runner records
that value and deliberately does not change it: QoS must remain identical in
all four cells so this experiment isolates bandwidth and TDD. It must not be
described as a 5QI-9 experiment unless the separately versioned core contract
is changed and requalified.

The host does not need an iperf package. The client runs in a short-lived
host-network container made from the same image as `oai-ext-dn`, while the
server runs freshly inside `oai-ext-dn`. Traffic is explicitly bound to
`10.0.0.2`.

## Run

First qualify all four configurations:

```bash
python3 oai_tdd_bandwidth_iperf_sweep_v1/run_oai_tdd_bandwidth_iperf_sweep_v1.py --smoke
```

After reviewing the smoke result, start the full sweep in `tmux` so it survives
terminal or VS Code restarts:

```bash
tmux new-session -s oai-tdd-sweep
python3 oai_tdd_bandwidth_iperf_sweep_v1/run_oai_tdd_bandwidth_iperf_sweep_v1.py --full
```

Resume an interrupted experiment without rerunning cells that already have an
atomic `COMPLETE.json`:

```bash
python3 oai_tdd_bandwidth_iperf_sweep_v1/run_oai_tdd_bandwidth_iperf_sweep_v1.py \
  --resume experiments/oai_tdd_bandwidth_iperf_sweep_v1/<run-directory>
```

The script asks for `sudo` once and refreshes that credential while it runs.
Do not use `sudo python3`; experiment artifacts should remain owned by the
calling user.

## Measurement contract

- UDP rates: 5, 10, 15, 20, 30, 45, 70 and 100 Mbps.
- 150, 200 and 300 Mbps are attempted sequentially only while the preceding
  point delivers at least 95% of offered throughput with at most 1% loss.
- Three full repetitions; rate order alternates to limit order effects.
- 1200-byte UDP datagrams avoid application-layer IP-fragmentation confounding.
- One single-stream TCP capacity cell per configuration and repetition.
- Concurrent loaded RTT and compact UE/gNB T-tracer capture.

`iperf3` delivery is datagram delivery, not split-inference frame completion.
The selected network configurations still require a later burst-shaped test
using the real feature framing and 100 ms frame deadline.

Artifacts are create-only under
`experiments/oai_tdd_bandwidth_iperf_sweep_v1/`; that directory is gitignored.
The runner does not launch CARLA, access model data, change QoS, rebuild OAI, or
modify the OAI submodule.
