#!/usr/bin/env bash
# Fix + re-run the two phases that failed overnight, appending to the same results TSV:
#   A) 5QI {9-ref via baseline already, 5, 1} on a WORKING TDD base (7:2, K2=6). (Overnight bug: it ran on the
#      broken 2:7 conf so 5QI was never actually tested.)
#   B) UL-heavy TDD {7:2, 4:5, 2:7} at min_rxtxtime=2 (K2=2). (Overnight: 2:7 crashed the gNB — N_dl1 >= k2-1
#      fails at K2=6 with only 2 DL slots. Lower K2 makes the extreme UL ratio valid so we can finally test it.)
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
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGDIR/rerun.log"; }

restore(){ say "RESTORE config.yaml + rm variant confs"; cp "$CFGYAML.rrbak" "$CFGYAML" 2>/dev/null && (cd "$OAI_CN_DIR" && sudo docker compose restart oai-smf >/dev/null 2>&1); rm -f "$OAI_RAN_CONF"/gnb_rerun_*.conf; }
trap restore EXIT
cp "$CFGYAML" "$CFGYAML.rrbak"

stop_ran(){ sudo pkill -f nr-uesoftmodem 2>/dev/null; sudo pkill -f nr-softmodem 2>/dev/null; sleep 4; }
start_gnb(){ ( cd "$OAI_RAN_BUILD"; setsid nohup sudo ./nr-softmodem -O "$OAI_RAN_CONF/$1" --gNBs.[0].min_rxtxtime "$3" --rfsim > "$LOGDIR/gnb_$2.log" 2>&1 & ); }
start_ue(){  ( cd "$OAI_RAN_BUILD"; setsid nohup sudo ./nr-uesoftmodem --rfsim --rfsimulator.[0].serveraddr 127.0.0.1 -r 106 --numerology 1 --band 78 -C "$UE_DL_FREQ" -O "$OAI_RAN_CONF/ue.conf" > "$LOGDIR/ue_$2.log" 2>&1 & ); }
wait_tunnel(){ for i in $(seq 1 40); do ip -4 addr show oaitun_ue1 2>/dev/null | grep -q "10.0.0.2" && { sleep 3; return 0; }; sleep 2; done; return 1; }

bring_up(){ # $1=gnbconf $2=tag $3=min_rxtxtime
  stop_ran
  ( cd "$OAI_CN_DIR" && sudo docker compose restart oai-amf oai-smf oai-upf >/dev/null 2>&1 )
  for i in $(seq 1 30); do [ "$(docker ps --format '{{.Names}} {{.Status}}' | grep -E 'oai-(amf|smf|upf)' | grep -c healthy)" -ge 3 ] && break; sleep 2; done
  sleep 3
  start_gnb "$1" "$2" "$3"; sleep 22; start_ue "$2"
  if wait_tunnel; then say "  tunnel UP ($2) ip=$(ip -4 addr show oaitun_ue1 2>/dev/null | grep -oE '10.0.0.[0-9]+' | head -1) K2=$3"; return 0
  else say "  !! tunnel FAILED ($2) — check $LOGDIR/gnb_$2.log"; return 1; fi
}

run_point(){ # $1=label $2=run_group
  say "  run front 300f: $1"
  setsid nohup "$PY" scripts/sample_oai_network_metrics.py --run-group "$2" --ping-host 192.168.70.135 > "$LOGDIR/sampler_$2.log" 2>&1 &
  local sp=$!
  timeout 700 "$PY" carla_split_inference_udp_fusion_object_pole_client_spatial_stream_oai.py \
    --role front --bind-host 10.0.0.2 --remote-host 192.168.70.140 --sync-world \
    --traffic-light-id 14 --camera-x 9 --camera-y 2 --camera-pitch -30 --camera-yaw-offset 50 --camera-roll 0 --camera-fov 100 \
    --fusion-checkpoint "$CKPT" --quantization-mode per_channel_uint8 --entropy-coder zlib --roi-threshold 0.0 \
    --no-spatial-map-stream --headless --max-frames "$FRAMES" --result-timeout 1.5 \
    --transport-label oai_rerun --run-group "$2" \
    --camera-source-port 51001 --remote-port 51002 --remote-source-port 51003 --camera-result-port 51004 \
    --front-device cuda > "$LOGDIR/front_$2.log" 2>&1
  local rc=$?; kill "$sp" 2>/dev/null; sleep 2
  "$PY" "$OUT/oai_extract_metrics.py" "$1" "$2" "$RESULTS"; say "  done $1 (rc=$rc)"
}

mk_tdd(){ local vc="gnb_rerun_tdd_$1_$2.conf"; cp "$OAI_RAN_CONF/$GNB_CONF" "$OAI_RAN_CONF/$vc"
  sed -i "s/\(nrofDownlinkSlots[[:space:]]*=[[:space:]]*\)[0-9]\+/\1$1/" "$OAI_RAN_CONF/$vc"
  sed -i "s/\(nrofUplinkSlots[[:space:]]*=[[:space:]]*\)[0-9]\+/\1$2/" "$OAI_RAN_CONF/$vc"; echo "$vc"; }

say "===== OAI RE-RUN START (frames=$FRAMES) ====="

# ---- PHASE A: 5QI on a WORKING base (TDD 7:2, K2=6) ----
say "--- PHASE A: 5QI (base TDD 7:2, K2=6) ---"
BASE72=$(mk_tdd 7 2)
for q in 5 1; do
  say "5QI=$q"
  cp "$CFGYAML.rrbak" "$CFGYAML"; sed -i "s/5qi: 9/5qi: $q/g" "$CFGYAML"
  if bring_up "$BASE72" "5qi_${q}" 6; then run_point "5qi_${q}_tdd72" "oaicfg_5qi_${q}"; fi
done
cp "$CFGYAML.rrbak" "$CFGYAML"

# ---- PHASE B: UL-heavy TDD at K2=2 (min_rxtxtime=2) ----
say "--- PHASE B: TDD ratio at K2=2 ---"
for pair in "7 2" "4 5" "2 7"; do
  set -- $pair; DL=$1; UL=$2
  say "TDD $DL:$UL @K2=2"
  VC=$(mk_tdd "$DL" "$UL")
  if bring_up "$VC" "tddk2_${DL}dl_${UL}ul" 2; then run_point "tddk2_${DL}dl_${UL}ul" "oaicfg_tddk2_${DL}dl_${UL}ul"; fi
done

say "===== RE-RUN DONE ====="
column -t -s $'\t' "$RESULTS" 2>/dev/null | tee -a "$LOGDIR/rerun.log"
stop_ran; say "RAN stopped. CN + back-half left up. Configs restored."
echo "OAI_RERUN_COMPLETE"
