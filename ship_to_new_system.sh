#!/usr/bin/env bash
# Ship abiodun/ (code + docs + configs + checkpoints + OAI source) and the Claude memory to a new
# system. NON-DESTRUCTIVE: no --delete, nothing removed locally. DRY-RUN by default.
#
#   DRY_RUN=1 (default)  -> rsync -n, shows what WOULD transfer, moves nothing
#   DRY_RUN=0            -> actually transfers
#
# Usage:
#   bash ship_to_new_system.sh              # dry run
#   DRY_RUN=0 bash ship_to_new_system.sh    # real transfer
set -uo pipefail

DEST_HOST="${DEST_HOST:-shr_aisvcs@L10319.idcc.lab}"
# Same absolute paths on both machines (new box mirrors the local structure).
ABI="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"
DEST_PARENT="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab"
CLAUDE_PROJ="/home/shr_aisvcs/.claude/projects/-home-shr-aisvcs-workarea-carla-0-10-env-Carla-0-10-0-Linux-Shipping-PythonAPI-neu-collab"

DRY_RUN="${DRY_RUN:-1}"
NFLAG=""; [ "$DRY_RUN" = "1" ] && NFLAG="-n" && echo "=== DRY RUN (set DRY_RUN=0 to transfer for real) ==="

# -a archive, -H hardlinks, -z compress, --partial resume-safe, progress2 overall bar. NO --delete.
RSYNC="rsync -aHz --partial --info=progress2 $NFLAG"

echo ">>> [1/3] abiodun code+docs+configs+checkpoints+OAI-source -> ${DEST_HOST}"
echo "    (excludes: $ABI/.rsync-transfer-excludes)"
$RSYNC --exclude-from="$ABI/.rsync-transfer-excludes" \
  "$ABI" "${DEST_HOST}:${DEST_PARENT}/"

echo ">>> [2/3] Claude project memory (durable session context) -> same path on new box"
ssh "$DEST_HOST" "mkdir -p '$CLAUDE_PROJ'" 2>/dev/null || true
$RSYNC "$CLAUDE_PROJ/memory" "${DEST_HOST}:${CLAUDE_PROJ}/"

echo ">>> [3/3] user Claude settings (permissions) -> new box"
$RSYNC "/home/shr_aisvcs/.claude/settings.json" "${DEST_HOST}:/home/shr_aisvcs/.claude/" 2>/dev/null || \
  echo "    (settings.json skipped)"

cat <<'NEXT'

=== after the transfer completes, on L10319 ===
 1. Rebuild OAI (required — T-tracer byte-compares T_messages.txt vs the compiled copy):
      cd .../abiodun/OAI/openairinterface5g/cmake_targets && ./build_oai ... (gNB + nr-UE)
 2. Re-clone V2Xverse if needed:  git clone https://github.com/CollaborativePerception/V2Xverse
 3. Smoke-check: `git -C .../abiodun status` (should match), run a small offline script.
 4. Optional — the offline eval dataset was NOT shipped (it is 75k symlinks -> 90 GB of raw).
    If you need to reproduce offline evals, materialize the merged set (deref symlinks):
      rsync -aL <local>:.../fusion_training_data/moving_ego_pps200000_merged_8loops_stride2 \
               .../fusion_training_data/
    or regenerate via the CARLA collector.
 5. Start Claude from the SAME directory you use locally so the memory hash matches and context auto-loads.
NEXT
