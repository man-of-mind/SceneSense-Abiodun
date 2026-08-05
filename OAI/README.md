# OAI 5G SA RFsim setup and validation guide

This README is for setting up an OpenAirInterface
5G SA RFsim environment:

- OAI 5G Core in Docker Compose
- OAI gNB in RF simulator mode
- OAI nrUE in RF simulator mode
- UE user-plane validation with ping and iperf
- application deployment patterns for using OAI as the network transport

Primary OAI references:

- Official CN5G tutorial: <https://gitlab.eurecom.fr/oai/openairinterface5g/-/blob/develop/doc/NR_SA_Tutorial_OAI_CN5G.md>
- Official OAI nrUE tutorial: <https://gitlab.eurecom.fr/oai/openairinterface5g/-/blob/develop/doc/NR_SA_Tutorial_OAI_nrUE.md>

The commands below assume one Ubuntu machine running CN, gNB, and nrUE in RFsim
mode. If gNB and UE are on separate machines, replace the RFsim server address
accordingly.

## 1. Recommended machine setup

Use a Linux machine with enough CPU headroom. The official OAI tutorial currently
targets Ubuntu 24.04 LTS. Ubuntu 22.04 may work, but if reproducing from scratch,
24.04 is the cleanest choice.

Recommended baseline:

- Ubuntu 24.04 LTS
- x86_64 CPU, 8 cores or more
- 32 GB RAM for CN + gNB builds/runs; 8 GB is usually enough for a UE-only host
- Docker Engine + Docker Compose plugin
- `sudo` access

Useful base packages:

```bash
sudo apt update
sudo apt install -y \
  git curl wget unzip ca-certificates \
  build-essential cmake ninja-build ccache \
  net-tools iproute2 iputils-ping iperf3 iperf \
  python3 python3-pip python3-venv \
  libsctp-dev lksctp-tools \
  cpufrequtils
```

Install Docker Engine and the Compose plugin:

```bash
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

For stable real-time behavior, set CPU governor to performance:

```bash
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
  echo performance | sudo tee "$cpu"
done
```

## 2. Clone OAI

Optional: Create a working directory:

```bash
mkdir -p ~/oai-rfsim
cd ~/oai-rfsim
```

Clone OAI RAN:

```bash
git clone https://gitlab.eurecom.fr/oai/openairinterface5g.git
cd openairinterface5g
git checkout develop
```

For reproducibility, record the exact commit:

```bash
git rev-parse HEAD
```

If sharing results with another machine, both machines should use the same commit.

## 3. Prepare the OAI 5G Core

The OAI repository includes the tutorial CN5G files. Copy them next to the RAN
tree so the CN config matches the OAI commit:

```bash
cd ~/oai-rfsim
cp -a openairinterface5g/doc/tutorial_resources/oai-cn5g ./oai-cn5g
cd oai-cn5g
docker compose pull
```

Start the core:

```bash
cd ~/oai-rfsim/oai-cn5g
docker compose up -d
```

Check status:

```bash
docker compose ps
docker ps --filter "name=oai-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

Expected important containers include:

- `oai-amf`
- `oai-smf`
- `oai-upf`
- `oai-ext-dn`
- `mysql`
- `oai-nrf`, `oai-udm`, `oai-udr`, `oai-ausf`

Stop the core:

```bash
cd ~/oai-rfsim/oai-cn5g
docker compose down
```

## 4. Build OAI gNB and nrUE for RFsim

Install OAI dependencies:

```bash
cd ~/oai-rfsim/openairinterface5g/cmake_targets
./build_oai -I
```

Build gNB and nrUE with RFsim support:

```bash
cd ~/oai-rfsim/openairinterface5g/cmake_targets
./build_oai --gNB --nrUE -w SIMU --ninja
```

Expected binaries:

```bash
ls -lh ~/oai-rfsim/openairinterface5g/cmake_targets/ran_build/build/nr-softmodem
ls -lh ~/oai-rfsim/openairinterface5g/cmake_targets/ran_build/build/nr-uesoftmodem
ls -lh ~/oai-rfsim/openairinterface5g/cmake_targets/ran_build/build/librfsimulator.so
```

If the build fails after changing branches or patches, rebuild cleanly:

```bash
cd ~/oai-rfsim/openairinterface5g/cmake_targets
./build_oai --gNB --nrUE -w SIMU --ninja -c
```

## 5. Confirm UE subscription credentials

The default RFsim UE configuration should match the CN database.

UE config:

```bash
grep -nE "imsi|key|opc|dnn" \
  ~/oai-rfsim/openairinterface5g/targets/PROJECTS/GENERIC-NR-5GC/CONF/ue.conf
```

Expected first UE defaults:

- IMSI: `001010000000001`
- DNN: `oai`
- UE tunnel IP after attach: usually `10.0.0.2`

The CN tutorial database normally maps the same IMSI to `10.0.0.2`.

```bash
grep -n "001010000000001" ~/oai-rfsim/oai-cn5g/database/oai_db.sql
grep -n "10.0.0.2" ~/oai-rfsim/oai-cn5g/database/oai_db.sql
```

If you edit the database SQL after the CN has already started, fully recreate the
CN containers/volumes before expecting the DB change to take effect.

## 6. Run CN, gNB, and UE

Use three terminals.

### Terminal 1: 5G Core

```bash
cd ~/oai-rfsim/oai-cn5g
docker compose up -d
docker compose ps
```

### Terminal 2: gNB

```bash
cd ~/oai-rfsim/openairinterface5g/cmake_targets/ran_build/build
sudo ./nr-softmodem \
  -O ../../../targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.sa.band78.fr1.106PRB.usrpb210.conf \
  --gNBs.[0].min_rxtxtime 6 \
  --rfsim
```

Useful signs in the gNB log:

- NG setup succeeds with AMF
- UE random access starts after nrUE launches
- RNTI appears
- UL/DL scheduler messages appear

### Terminal 3: nrUE

Start the UE after the gNB is running:

```bash
cd ~/oai-rfsim/openairinterface5g/cmake_targets/ran_build/build
sudo ./nr-uesoftmodem \
  --rfsim \
  --rfsimulator.[0].serveraddr 127.0.0.1 \
  -r 106 \
  --numerology 1 \
  --band 78 \
  -C 3619200000 \
  -O ../../../targets/PROJECTS/GENERIC-NR-5GC/CONF/ue.conf
```

Useful signs in the UE log:

- registration succeeds
- PDU session is established
- Linux interface `oaitun_ue1` appears

On recent OAI tags, SA mode is the default. On older OAI versions, add `--sa` to
both gNB and UE commands.

## 7. Basic connectivity checks

In a fourth terminal:

```bash
ip -br addr show oaitun_ue1
```

Expected: `oaitun_ue1` has an address such as `10.0.0.2`.

Ping from the UE tunnel toward the external data network:

```bash
ping -I oaitun_ue1 -c 5 192.168.70.135
```

Ping from the external data network back to the UE:

```bash
docker exec -it oai-ext-dn ping -c 5 10.0.0.2
```

Useful log checks:

```bash
docker logs oai-amf 2>&1 | grep -Ei "registered|5GMM|IPV4|UE Address" | tail -20
docker logs oai-smf 2>&1 | grep -Ei "UE Address|10\\.0\\.0|IPv4" | tail -20
docker exec -it oai-ext-dn ip route
```

The `oai-ext-dn` route should include:

```text
10.0.0.0/16 via 192.168.70.134
```

where `192.168.70.134` is the UPF container.

## 8. iperf validation

### Uplink: UE tunnel to external data network

Start an iperf3 server inside `oai-ext-dn`:

```bash
docker exec -it oai-ext-dn iperf3 -s -B 192.168.70.135
```

In another terminal, run an uplink UDP client from the UE tunnel:

```bash
iperf3 -c 192.168.70.135 \
  -u -b 10M -t 30 \
  -B 10.0.0.2
```

For TCP uplink:

```bash
iperf3 -c 192.168.70.135 -t 30 -B 10.0.0.2
```

### Downlink: external data network to UE tunnel

Start an iperf3 server on the UE tunnel address:

```bash
iperf3 -s -B 10.0.0.2
```

In another terminal, run a downlink UDP client from `oai-ext-dn`:

```bash
docker exec -it oai-ext-dn iperf3 \
  -c 10.0.0.2 \
  -u -b 10M -t 30 \
  -B 192.168.70.135
```

For TCP downlink:

```bash
docker exec -it oai-ext-dn iperf3 \
  -c 10.0.0.2 \
  -t 30 \
  -B 192.168.70.135
```

Interpretation:

- Uplink means traffic enters the UE tunnel, traverses nrUE -> RFsim -> gNB -> UPF -> external data network.
- Downlink means traffic starts in the external data network and returns through UPF -> gNB -> RFsim -> nrUE -> UE tunnel.
- Always bind the UE-side process to `10.0.0.2` or `oaitun_ue1`; otherwise Linux may choose a non-OAI interface.

## 9. Using OAI as transport for an application similar to the CARLA Split-inference traffic

There are two clean deployment patterns.

### Pattern A: application server on the CN external network

This is the easiest pattern for an edge/server application.

1. Attach the application container to the OAI Docker network:

   ```yaml
   services:
     my-edge-app:
       image: my-edge-app:latest
       container_name: my-edge-app
       privileged: true
       networks:
         public_net:
           ipv4_address: 192.168.70.140

   networks:
     public_net:
       external: true
       name: oai-cn5g-public-net
   ```

2. Add a route from the application container to the UE subnet:

   ```bash
   ip route add 10.0.0.0/16 via 192.168.70.134 dev eth0
   ```

   In Docker Compose, this can be placed in the container entrypoint before the
   application starts.

3. The edge application listens on its CN-side IP:

   ```text
   bind = 0.0.0.0:<server_port>
   or
   bind = 192.168.70.140:<server_port>
   ```

4. The UE-side client sends to the edge server:

   ```text
   source/bind = 10.0.0.2:<client_source_port>
   destination = 192.168.70.140:<server_port>
   ```

5. If the edge app needs to send a response to the UE, send to:

   ```text
   10.0.0.2:<ue_result_port>
   ```

This is the pattern we used for the split-inference back-half container: the
front half runs on the UE side and binds to `10.0.0.2`; the back half lives on
the CN/external network and listens on a `192.168.70.x` address.

### Pattern B: application process reuses the UE tunnel address

This pattern is for applications running on the same host as `nr-uesoftmodem`.
The application must explicitly bind to the OAI UE address so packets enter the
OAI user plane.

Example Python UDP client skeleton:

```python
import socket

ue_ip = "10.0.0.2"
edge_ip = "192.168.70.140"
local_port = 51001
remote_port = 51002

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((ue_ip, local_port))
sock.sendto(b"hello over OAI", (edge_ip, remote_port))
```

Example Python UDP server on the UE side for downlink messages:

```python
import socket

ue_ip = "10.0.0.2"
listen_port = 51004

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((ue_ip, listen_port))

while True:
    data, addr = sock.recvfrom(65535)
    print("received", len(data), "bytes from", addr)
```

Key rule: do not bind UE-side traffic to `127.0.0.1`, the host LAN IP, or
`0.0.0.0` unless you have confirmed routing. Use `10.0.0.2` when the goal is to
force the application traffic over OAI.

## 10. Troubleshooting checklist

### CN containers are not healthy

Check:

```bash
cd ~/oai-rfsim/oai-cn5g
docker compose ps
docker logs oai-amf --tail 100
docker logs oai-smf --tail 100
docker logs oai-upf --tail 100
```

Common causes:

- Docker was not restarted after installation.
- Old containers/volumes still hold stale DB state.
- Another Docker network already uses `192.168.70.128/26`.

Clean restart:

```bash
cd ~/oai-rfsim/oai-cn5g
docker compose down
docker compose up -d
```

If you changed the database SQL and need to remove old DB state, remove the
Compose volume too. Be careful: this deletes CN database state.

### gNB cannot connect to AMF

Check:

```bash
docker inspect oai-amf | grep -n "IPAddress"
grep -n "amf" ~/oai-rfsim/openairinterface5g/targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.sa.band78.fr1.106PRB.usrpb210.conf
```

The gNB config should point to the AMF on the Docker network, normally
`192.168.70.132`. Also confirm Docker CN is already up before starting gNB.

### UE does not attach or `oaitun_ue1` is missing

Check:

```bash
ip -br addr show oaitun_ue1
docker logs oai-amf --tail 100
docker logs oai-smf --tail 100
```

Common causes:

- gNB was not fully up before starting UE.
- UE `--rfsimulator.[0].serveraddr` is wrong.
- UE IMSI/key/OPC does not match the CN database.
- On older OAI releases, `--sa` is required.
- Stale CN database state after changing subscriber SQL.

### Ping works one way but not the other

Check the route inside `oai-ext-dn`:

```bash
docker exec -it oai-ext-dn ip route
```

You should see a route for `10.0.0.0/16` via the UPF. If running a custom
application container on the CN network, it also needs a route to the UE subnet.

### Application traffic does not traverse OAI

Most common cause: the UE-side application did not bind to the UE tunnel IP.

Check:

```bash
ip -br addr show oaitun_ue1
ss -uapn | grep <your_port>
ip route get 192.168.70.140 from 10.0.0.2
```

The source IP should be `10.0.0.2`. If the source is a host LAN IP, the traffic
is not following the OAI UE path.

### Build problems

Try:

```bash
cd ~/oai-rfsim/openairinterface5g/cmake_targets
./build_oai -I
./build_oai --gNB --nrUE -w SIMU --ninja -c
```

If the machine is low on memory, reduce parallelism:

```bash
export MAKEFLAGS="-j4"
```

Then rebuild.

### Real-time or heavy-traffic instability

For RFsim debugging, use performance governor and keep the machine lightly
loaded. OAI also exposes runtime knobs such as:

```bash
--gNBs.[0].min_rxtxtime 6
--MACRLCs.[0].ulsch_max_frame_inactivity 0
--MACRLCs.[0].ul_max_mcs 14
--L1s.[0].max_ldpc_iterations 4
```

Use these as diagnostic knobs, not as default scientific settings, unless the
experiment explicitly requires them.
