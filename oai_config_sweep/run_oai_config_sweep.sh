#!/usr/bin/env bash
# Autonomous OAI config sweep. Model FIXED at no-AE u8 (config is the only variable). Phases in priority order:
#   1) TDD DL:UL slot ratio (biggest uplink lever)   2) 5QI QoS profile   3) bandwidth/PRB (riskiest, last)
# Per config: (re)start gNB+UE with the variant, health-check UE tunnel (skip+log on failure), run the fixed
# 300-frame pole front over OAI, sample network, extract RTT/delivery/payload -> results TSV. Continue-on-failure.
# Restores original config.yaml + removes variant confs at the end. Assumes: CN up, back-half container up
# (no-AE u8, single worker), CARLA up. Passwordless sudo + docker-without-sudo confirmed.
set -uo pipefail
AB="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"
cd "$AB"; source scripts/config.env
PY="/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python"
export PYTHONPATH="$AB/pole_lraspp_multimodal_fusion:$AB:$AB/rl_agent/feature_ae"
CKPT="experiments/ae_integrated_20260710/noae_baseline/checkpoints/mprime_joint_noae/best.pt"
OUT="$AB/oai_config_sweep"; LOGDIR="$OUT/logs"; mkdir -p "$LOGDIR"
RESULTS="$OUT/oai_config_results.tsv"
FRAMES="${FRAMES:-300}"
CFGYAML="$OAI_CN_DIR/conf/config.yaml"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGDIR/sweep.log"; }

restore(){
  say "RESTORE: config.yaml 5QI -> 9, remove variant confs"
  cp "$CFGYAML.sweepbak" "$CFGYAML" 2>/dev/null && (cd "$OAI_CN_DIR" && sudo docker compose restart oai-smf >/dev/null 2>&1)
  rm -f "$OAI_RAN_CONF"/gnb_sweep_*.conf
}
trap restore EXIT
cp "$CFGYAML" "$CFGYAML.sweepbak"

stop_ran(){ sudo pkill -f nr-uesoftmodem 2>/dev/null; sudo pkill -f nr-softmodem 2>/dev/null; sleep 4; }
start_gnb(){ ( cd "$OAI_RAN_BUILD"; setsid nohup sudo ./nr-softmodem -O "$OAI_RAN_CONF/$1" --gNBs.[0].min_rxtxtime 6 --rfsim > "$LOGDIR/gnb_$2.log" 2>&1 & ); }
start_ue(){  ( cd "$OAI_RAN_BUILD"; setsid nohup sudo ./nr-uesoftmodem --rfsim --rfsimulator.[0].serveraddr 127.0.0.1 -r "$1" --numerology 1 --band 78 -C "$2" -O "$OAI_RAN_CONF/ue.conf" > "$LOGDIR/ue_$3.log" 2>&1 & ); }
wait_tunnel(){ for i in $(seq 1 40); do ip -4 addr show oaitun_ue1 2>/dev/null | grep -q "10.0.0.2" && { sleep 3; return 0; }; sleep 2; done; return 1; }

# bring RAN up with a given gNB conf + UE PRB/freq; returns 0 iff tunnel attaches at 10.0.0.2
bring_up(){ # $1=gnbconf $2=prb $3=freq $4=tag
  stop_ran
  # Reset core NF session/IP state: stale PDU sessions make the SMF hand out incrementing IPs (10.0.0.4,...),
  # but the back-half container returns results to 10.0.0.2. Restarting amf/smf/upf clears the pool -> UE re-gets .2.
  # (Also applies any config.yaml 5QI edit made just before this call.)
  ( cd "$OAI_CN_DIR" && sudo docker compose restart oai-amf oai-smf oai-upf >/dev/null 2>&1 )
  for i in $(seq 1 30); do
    [ "$(docker ps --format '{{.Names}} {{.Status}}' | grep -E 'oai-(amf|smf|upf)' | grep -c healthy)" -ge 3 ] && break
    sleep 2
  done
  sleep 3
  start_gnb "$1" "$4"; sleep 22; start_ue "$2" "$3" "$4"
  if wait_tunnel; then
    say "  tunnel UP ($4) ip=$(ip -4 addr show oaitun_ue1 2>/dev/null | grep -oE '10.0.0.[0-9]+' | head -1)"; return 0
  else say "  !! tunnel FAILED / not 10.0.0.2 ($4) — skipping"; return 1; fi
}

run_point(){ # $1=label $2=run_group
  say "  run front 300f: $1 ($2)"
  setsid nohup "$PY" scripts/sample_oai_network_metrics.py --run-group "$2" --ping-host 192.168.70.135 > "$LOGDIR/sampler_$2.log" 2>&1 &
  local sp=$!
  timeout 700 "$PY" carla_split_inference_udp_fusion_object_pole_client_spatial_stream_oai.py \
    --role front --bind-host 10.0.0.2 --remote-host 192.168.70.140 --sync-world \
    --traffic-light-id 14 --camera-x 9 --camera-y 2 --camera-pitch -30 --camera-yaw-offset 50 --camera-roll 0 --camera-fov 100 \
    --fusion-checkpoint "$CKPT" --quantization-mode per_channel_uint8 --entropy-coder zlib --roi-threshold 0.0 \
    --no-spatial-map-stream --headless --max-frames "$FRAMES" --result-timeout 1.5 \
    --transport-label oai_sweep --run-group "$2" \
    --camera-source-port 51001 --remote-port 51002 --remote-source-port 51003 --camera-result-port 51004 \
    --front-device cuda > "$LOGDIR/front_$2.log" 2>&1
  local rc=$?; kill "$sp" 2>/dev/null; sleep 2
  "$PY" "$OUT/oai_extract_metrics.py" "$1" "$2" "$RESULTS"
  say "  done $1 (front rc=$rc)"
}

mk_tdd(){ local vc="gnb_sweep_tdd_$1_$2.conf"; cp "$OAI_RAN_CONF/$GNB_CONF" "$OAI_RAN_CONF/$vc"
  sed -i "s/\(nrofDownlinkSlots[[:space:]]*=[[:space:]]*\)[0-9]\+/\1$1/" "$OAI_RAN_CONF/$vc"
  sed -i "s/\(nrofUplinkSlots[[:space:]]*=[[:space:]]*\)[0-9]\+/\1$2/" "$OAI_RAN_CONF/$vc"; echo "$vc"; }
mk_prb(){ local vc="gnb_sweep_prb_$1.conf"
  # NR locationAndBandwidth RIV against N=275 (RBstart=0): L-1<=137 -> 275*(L-1); else 275*(276-L)+274.
  local riv; if [ $(($1-1)) -le 137 ]; then riv=$((275*($1-1))); else riv=$((275*(276-$1)+274)); fi
  cp "$OAI_RAN_CONF/$GNB_CONF" "$OAI_RAN_CONF/$vc"
  sed -i "s/\(dl_carrierBandwidth[[:space:]]*=[[:space:]]*\)[0-9]\+/\1$1/" "$OAI_RAN_CONF/$vc"
  sed -i "s/\(ul_carrierBandwidth[[:space:]]*=[[:space:]]*\)[0-9]\+/\1$1/" "$OAI_RAN_CONF/$vc"
  sed -i "s/\(initialDLBWPlocationAndBandwidth[[:space:]]*=[[:space:]]*\)[0-9]\+/\1$riv/" "$OAI_RAN_CONF/$vc"
  sed -i "s/\(initialULBWPlocationAndBandwidth[[:space:]]*=[[:space:]]*\)[0-9]\+/\1$riv/" "$OAI_RAN_CONF/$vc"; echo "$vc"; }

say "===== OAI CONFIG SWEEP START (frames=$FRAMES, model=no-AE u8) ====="

# ---------- PHASE 1: TDD (DL:UL). DL+UL must = 9 (10 slots - 1 special). ----------
say "--- PHASE 1: TDD DL:UL ---"
for pair in "7 2" "4 5" "2 7"; do
  set -- $pair; DL=$1; UL=$2; LB="tdd_${DL}dl_${UL}ul"; RG="oaicfg_$LB"
  say "TDD $DL:$UL"
  VC=$(mk_tdd "$DL" "$UL")
  if bring_up "$VC" 106 "$UE_DL_FREQ" "$LB"; then run_point "$LB" "$RG"; fi
done

# ---------- PHASE 2: 5QI (QoS). Baseline 9 already covered by tdd_7dl_2ul. Try delay/GBR profiles. ----------
say "--- PHASE 2: 5QI ---"
BEST_TDD_CONF=$(mk_tdd 2 7)   # use an uplink-favored TDD so QoS effect is visible on a non-saturated link
for q in 5 1; do
  LB="5qi_${q}"; RG="oaicfg_$LB"
  say "5QI=$q"
  cp "$CFGYAML.sweepbak" "$CFGYAML"; sed -i "s/5qi: 9/5qi: $q/g" "$CFGYAML"   # bring_up's smf restart applies it
  if bring_up "$BEST_TDD_CONF" 106 "$UE_DL_FREQ" "$LB"; then run_point "$LB" "$RG"; fi
done
cp "$CFGYAML.sweepbak" "$CFGYAML"   # restore 5QI=9 (bring_up on next phase's config re-applies via smf restart)

# ---------- PHASE 3: bandwidth/PRB (riskiest, last). RIV=275*(PRB-1); freq unchanged (rfsim-tolerant). ----------
say "--- PHASE 3: bandwidth/PRB ---"
for prb in 162 217 273; do
  LB="prb_${prb}"; RG="oaicfg_$LB"
  say "PRB=$prb"
  VC=$(mk_prb "$prb")
  if bring_up "$VC" "$prb" "$UE_DL_FREQ" "$LB"; then run_point "$LB" "$RG"; fi
done

say "===== SWEEP DONE. Results: $RESULTS ====="
column -t -s $'\t' "$RESULTS" 2>/dev/null | tee -a "$LOGDIR/sweep.log"
stop_ran
say "RAN stopped. CN + back-half container left up. Configs restored on exit."
echo "OAI_SWEEP_COMPLETE"
